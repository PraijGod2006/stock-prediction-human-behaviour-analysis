"""
========================================================================================
STANDALONE AUX DATA INFERENCE RUNNER
========================================================================================
Loads the newly retrained 3-model XGBoost artifacts and executes inference on any 
OHLCV CSV from Aux Data 1 (Stocks) or Aux Data 2 (Indices).

Features:
---------
- Uses Optuna-tuned hyperparameters & newly trained model weights:
    * artifacts/model_1_direction.json (27 features: 9 technical + 18 peer momentum)
    * artifacts/model_2_price.json     (27 features)
    * artifacts/model_3_exhaustion.json (12 features)
- Enforces Typical Price = (High + Low + Close) / 3 for all reference prices and returns
- Computes shift(1) leakage-free Z-scores and technical indicators
- Injects 18 peer features from artifacts/peer_map.json (across raw, binary, spike dimensions)
- Applies multi-model signal combiner and ATR-based risk management
========================================================================================
"""

import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd

# Setup path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from feature_engine.pipeline import build_features_1min, build_features_5min
from feature_engine.reference_price import compute_reference_price
from correlation.cross_asset import DEFAULT_PEER_COLS
from models.model_1_direction import DirectionalModel
from models.model_2_price import PriceModel
from models.model_3_exhaustion import ExhaustionModel
from models.model_3b_runup import RunupModel
from backtest.risk_manager import calculate_atr, calculate_position_size, check_kill_switch

# Paths
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
MODEL_1_PATH = os.path.join(ARTIFACTS_DIR, "model_1_direction.json")
MODEL_2_PATH = os.path.join(ARTIFACTS_DIR, "model_2_price.json")
MODEL_3_PATH = os.path.join(ARTIFACTS_DIR, "model_3_exhaustion.json")
MODEL_3B_PATH = os.path.join(ARTIFACTS_DIR, "model_3b_runup.json")
PEER_MAP_PATH = os.path.join(ARTIFACTS_DIR, "peer_map.json")

# Schema definitions
EXPECTED_FEATURES_5MIN = [
    "return_zscore",
    "log_return_zscore",
    "rolling_zscore",
    "volume_zscore",
    "volatility_zscore",
    "price_vs_ma_zscore",
    "price_vs_ema_zscore",
    "range_zscore",
    "momentum_zscore",
    "bollinger_pctb",
    "bollinger_bandwidth",
] + DEFAULT_PEER_COLS

EXPECTED_FEATURES_1MIN = [
    "return_zscore",
    "log_return_zscore",
    "rolling_zscore",
    "volume_zscore",
    "dollar_volume_zscore",
    "volatility_zscore",
    "range_zscore",
    "price_vs_ma_zscore",
    "price_vs_ema_zscore",
    "volume_zscore_20",
    "distance_from_vwap",
    "rsi_14",
    "relative_volume_tod",
    "bollinger_pctb",
    "bollinger_dist_upper",
    "bollinger_bandwidth",
]


def load_models():
    """Loads all four trained models from artifacts."""
    if check_kill_switch():
        raise RuntimeError("Kill Switch is active in artifacts/KILL_SWITCH! Inference halted.")
    
    for p, name in [
        (MODEL_1_PATH, "Model 1"),
        (MODEL_2_PATH, "Model 2"),
        (MODEL_3_PATH, "Model 3"),
        (MODEL_3B_PATH, "Model 3b"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing model artifact: {p}")
            
    m1 = DirectionalModel(MODEL_1_PATH)
    m2 = PriceModel(MODEL_2_PATH)
    m3 = ExhaustionModel(MODEL_3_PATH)
    m3b = RunupModel(MODEL_3B_PATH)
    return m1, m2, m3, m3b


def load_raw_csv(csv_path: str) -> pd.DataFrame:
    """Standardizes input CSV to pure OHLCV format."""
    df = pd.read_csv(csv_path)
    rename_map = {}
    for col in df.columns:
        c_low = col.strip().lower()
        if c_low in ("datetime", "timestamp", "time"):
            rename_map[col] = "date"
        elif c_low in ("open", "high", "low", "close", "volume"):
            rename_map[col] = c_low

    df = df.rename(columns=rename_map)
    required = ["date", "open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")

    clean_df = pd.DataFrame(df[required]).copy()
    clean_df["date"] = pd.to_datetime(clean_df["date"])
    if hasattr(clean_df["date"].dt, "tz") and clean_df["date"].dt.tz is not None:
        clean_df["date"] = clean_df["date"].dt.tz_localize(None)

    return clean_df.dropna(subset=required).sort_values("date").reset_index(drop=True)


def inject_peer_features(df_5min: pd.DataFrame, symbol: str, day_folder: str | None = None) -> pd.DataFrame:
    """Injects 18 peer momentum features across raw, binary, and spike dimensions."""
    if not os.path.exists(PEER_MAP_PATH):
        for col in DEFAULT_PEER_COLS:
            df_5min[col] = 0.0
        return df_5min

    with open(PEER_MAP_PATH) as f:
        peer_map = json.load(f)

    if symbol not in peer_map:
        for col in DEFAULT_PEER_COLS:
            df_5min[col] = 0.0
        return df_5min

    sym_data = peer_map[symbol]
    dimensions = {
        "raw": sym_data.get("correlated_peers_raw", {}),
        "binary": sym_data.get("correlated_peers_binary", {}),
        "spike": sym_data.get("correlated_peers_spike", {}),
    }

    for dim_name, dim_data in dimensions.items():
        pos_peers = dim_data.get("top_positive", [])[:3]
        neg_peers = dim_data.get("top_negative", [])[:3]

        for rank, peer in enumerate(pos_peers):
            col = f"peer_pos_{dim_name}_{rank + 1}_return_lag1"
            ret_series = _get_peer_series(peer, day_folder)
            df_5min[col] = df_5min.index.map(ret_series).fillna(0.0) if ret_series is not None else 0.0

        for rank, peer in enumerate(neg_peers):
            col = f"peer_neg_{dim_name}_{rank + 1}_return_lag1"
            ret_series = _get_peer_series(peer, day_folder)
            df_5min[col] = df_5min.index.map(ret_series).fillna(0.0) if ret_series is not None else 0.0

    for col in DEFAULT_PEER_COLS:
        if col not in df_5min.columns:
            df_5min[col] = 0.0
        df_5min[col] = df_5min[col].fillna(0.0)

    return df_5min


def _get_peer_series(peer: str, day_folder: str | None) -> pd.Series | None:
    """Extracts 5-minute shift(1) lagged returns using Typical Price."""
    if day_folder and os.path.exists(day_folder):
        matches = glob.glob(os.path.join(day_folder, f"{peer}_*.csv"))
        if matches:
            try:
                pdf = load_raw_csv(matches[0])
                if len(pdf) > 10:
                    p_5m = build_features_5min(pdf)
                    ref_p = compute_reference_price(p_5m) if all(c in p_5m.columns for c in ['high', 'low', 'close']) else p_5m['close']
                    return (ref_p / ref_p.shift(1) - 1.0).shift(1)
            except Exception:
                pass
    return None


def run_inference(
    csv_path: str,
    symbol: str,
    model_1: DirectionalModel,
    model_2: PriceModel,
    model_3: ExhaustionModel,
    is_stock: bool = True,
    confidence_hurdle: float = 0.52,
    capital: float = 100000.0,
) -> pd.DataFrame:
    """Executes end-to-end 3-model inference on a given CSV file."""
    raw_df = load_raw_csv(csv_path)
    if len(raw_df) < 25:
        raise ValueError(f"Insufficient bars ({len(raw_df)}) for rolling indicator warmup.")

    day_folder = os.path.dirname(csv_path) if is_stock else None

    # Step 1: Feature Engineering
    df_1min = build_features_1min(raw_df)
    df_5min = build_features_5min(raw_df)

    # Step 2: Peer Features
    if is_stock:
        df_5min = inject_peer_features(df_5min, symbol, day_folder=day_folder)
    else:
        for col in DEFAULT_PEER_COLS:
            df_5min[col] = 0.0

    # Step 3: Feature Matrix Validation
    X_5min = pd.DataFrame(df_5min[EXPECTED_FEATURES_5MIN]).replace([np.inf, -np.inf], np.nan).dropna()
    X_1min = pd.DataFrame(df_1min[EXPECTED_FEATURES_1MIN]).replace([np.inf, -np.inf], np.nan).dropna()

    if len(X_5min) == 0 or len(X_1min) == 0:
        return pd.DataFrame()

    # Step 4: Run Models
    pred_dir, prob_dir = model_1.predict(X_5min)
    pred_price = model_2.predict(X_5min)
    pred_drawdown = model_3.predict(X_1min)

    # Step 5: Decision Logic & Sizing
    atr_series = calculate_atr(df_5min, period=14)
    ref_prices = compute_reference_price(df_5min) if all(c in df_5min.columns for c in ['high', 'low', 'close']) else df_5min['close']

    results = []
    n_bars = min(len(pred_dir), len(pred_price), len(pred_drawdown))

    for i in range(n_bars):
        direction = int(pred_dir[i])
        confidence = float(prob_dir[i])
        price_move = float(pred_price[i])
        drawdown = float(pred_drawdown[i])
        current_price = float(ref_prices.iloc[i])
        timestamp = str(df_5min.index[i])

        signal_type = "HOLD"
        entry_price = current_price

        # Confidence: prob_dir is P(UP). P(DOWN) is 1.0 - confidence
        p_up = confidence
        p_down = 1.0 - confidence

        if direction == 1 and drawdown < -0.003:
            # Mean Reversion Buy on flush
            entry_price = current_price * (1.0 - abs(drawdown))
            signal_type = "MEAN_REVERSION_BUY"
        elif direction == 1 and p_up >= confidence_hurdle and price_move > 0:
            signal_type = "MOMENTUM_BUY"
        elif direction == 0 and p_down >= confidence_hurdle and price_move < 0:
            signal_type = "MOMENTUM_SELL"

        # ATR-based Position Sizing
        current_atr = float(atr_series.iloc[i]) if i < len(atr_series) and not np.isnan(atr_series.iloc[i]) else current_price * 0.005
        shares = 0
        position_value = 0.0

        if signal_type != "HOLD":
            shares = int(calculate_position_size(
                capital=capital,
                risk_pct=0.02,
                atr=current_atr,
                price=entry_price,
                max_position_pct=0.10,
            ))
            position_value = round(shares * entry_price, 2)

        results.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "signal": signal_type,
            "entry_price": round(entry_price, 2),
            "ref_price": round(current_price, 2),
            "m1_direction": direction,
            "m1_confidence": round(confidence, 4),
            "m2_price_move": round(price_move, 6),
            "m3_drawdown": round(drawdown, 6),
            "atr_14": round(current_atr, 4),
            "shares": shares,
            "position_value": position_value,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3-Model XGBoost Inference on Aux Data CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to input raw CSV file")
    parser.add_argument("--symbol", type=str, default=None, help="Stock or index ticker symbol (auto-inferred if omitted)")
    parser.add_argument("--index", action="store_true", help="Set flag if processing index data (disables peer injection)")
    parser.add_argument("--conf", type=float, default=0.52, help="Confidence threshold (default: 0.52)")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path for predictions")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    sym = args.symbol
    if not sym:
        base = os.path.basename(csv_path)
        sym = base.split("_")[0]

    print(f"Loading models and running inference on {sym} ({os.path.basename(csv_path)})...")
    m1, m2, m3 = load_models()
    signals_df = run_inference(
        csv_path=csv_path,
        symbol=sym,
        model_1=m1,
        model_2=m2,
        model_3=m3,
        is_stock=not args.index,
        confidence_hurdle=args.conf,
    )

    print(f"Generated {len(signals_df)} bar predictions.")
    if not signals_df.empty:
        actionable = signals_df[signals_df["signal"] != "HOLD"]
        print(f"Actionable Trades: {len(actionable)} / {len(signals_df)}")
        print("\nRecent 5 Predictions:")
        print(signals_df[["timestamp", "signal", "entry_price", "m1_confidence", "shares"]].tail(5).to_string(index=False))

        out_path = args.out or os.path.join(ARTIFACTS_DIR, "inference_results", f"{sym}_inference_output.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        signals_df.to_csv(out_path, index=False)
        print(f"\nSaved full results to: {out_path}")

