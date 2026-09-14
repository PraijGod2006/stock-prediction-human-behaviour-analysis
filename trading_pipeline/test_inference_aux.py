"""
========================================================================================
TEST INFERENCE PIPELINE: Run Trained Models on Aux Data 1 & Aux Data 2
========================================================================================

PURPOSE:
--------
Validates the fully trained 3-Model XGBoost Trading Pipeline against unseen out-of-sample
data from two external auxiliary sources:

1. AUX DATA 1 (NIFTY 50 Stocks, Day-by-Day):
   - Located at: DATA/aux data 1/
   - Contains 23 trading day directories (e.g., nifty50_01Jun, nifty50_02Jun...).
   - Each folder contains 50 NIFTY50 constituent 1-minute CSVs for that single session.
   - Raw columns: Datetime, Close, High, Low, Open, Volume, [precomputed features], target.
   - We strip the precomputed features and target, keeping ONLY pure raw OHLCV.
   - Contemporaneous peer returns are dynamically extracted across peer CSV files in the
     same daily folder (SBIN, ICICIBANK, AXISBANK, etc.) according to artifacts/peer_map.json.

2. AUX DATA 2 (NIFTY Broad Market & Sectoral Indices):
   - Located at: DATA/aux data 2/
   - Contains 136 minute-level CSVs for indices (NIFTY 50, NIFTY BANK, NIFTY IT, etc.).
   - Indices do not have equity trading volume (volume = 0). Our pipeline gracefully
     handles flat volume with 0.0 z-scores and price-tracking VWAP without dropping data.
   - Peer momentum features are cleanly zeroed out for index-level evaluation.

THE 3-MODEL SYSTEM:
-------------------
- Model 1 (Directional): XGBClassifier -> 5-min binary UP/DOWN prediction.
  Features (27): 9 technical z-scores + 18 cross-asset peer momentum lags (raw, binary, spike).
- Model 2 (Price Movement): XGBRegressor -> 1-min return prediction of next bar.
  Features (27): Exact same 27 features as Model 1.
- Model 3 (Exhaustion): XGBRegressor -> 10-min forward maximum drawdown.
  Features (12): 1-min z-scores, Volume Z(20), VWAP Distance, and RSI(14).

SIGNAL COMBINATION LOGIC:
-------------------------
1. MEAN REVERSION BUY:
   - Model 1 predicts UP (Direction = 1)
   - Model 3 detects sharp incoming flush (Drawdown < -0.3% / -0.003)
   - Action: Place Limit Order at Current_Price * (1.0 - abs(Predicted_Drawdown))
2. MOMENTUM BUY:
   - Model 1 predicts UP (Direction = 1) with Probability > 0.60 (No exhaustion)
   - Action: Enter Market/Mid-Price
3. MOMENTUM SELL:
   - Model 1 predicts DOWN (Direction = 0) with Probability > 0.60
   - Action: Enter Short / Exit Long
4. HOLD:
   - All other conditions.

RISK CONTROLS:
--------------
- Kill Switch: Halts immediately if artifacts/KILL_SWITCH is present.
- Position Sizing: 2% risk per trade scaled by 14-period Average True Range (ATR).
- Exposure Cap: Max 2 correlated positions simultaneously.

USAGE:
------
    # Run test on sample of aux data (3 days of stocks, 3 indices):
    python trading_pipeline/test_inference_aux.py

    # Run full comprehensive test across ALL days and ALL files:
    python trading_pipeline/test_inference_aux.py --all

========================================================================================
"""

import argparse
import gc
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PATH SETUP: Ensure trading_pipeline root is in sys.path
# ---------------------------------------------------------------------------
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)

# ---------------------------------------------------------------------------
# MODULAR PIPELINE IMPORTS (with explicit library attribution)
# ---------------------------------------------------------------------------
# [backtest/risk_manager.py]
# Provides ATR volatility calculations, position sizing, and exposure capping
from backtest.risk_manager import (
    ExposureManager,
    calculate_atr,
    calculate_position_size,
    check_kill_switch,
)

# [feature_engine/pipeline.py]
# Builds leak-free shift(1) Z-scores, RSI, VWAP distance, and resampled 5-min bars
from feature_engine.pipeline import build_features_1min, build_features_5min

# [models/model_1_direction.py, model_2_price.py, model_3_exhaustion.py]
# Core XGBoost trained model wrappers
from models.model_1_direction import DirectionalModel
from models.model_2_price import PriceModel
from models.model_3_exhaustion import ExhaustionModel

from correlation.cross_asset import DEFAULT_PEER_COLS
from feature_engine.reference_price import compute_reference_price

# ---------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(PIPELINE_DIR, "artifacts")
MODEL_1_PATH = os.path.join(ARTIFACTS_DIR, "model_1_direction.json")
MODEL_2_PATH = os.path.join(ARTIFACTS_DIR, "model_2_price.json")
MODEL_3_PATH = os.path.join(ARTIFACTS_DIR, "model_3_exhaustion.json")
PEER_MAP_PATH = os.path.join(ARTIFACTS_DIR, "peer_map.json")
PARQUET_DIR = os.path.join(WORKSPACE_DIR, "DATA", "parquet")

AUX_DATA_1_DIR = os.path.join(WORKSPACE_DIR, "DATA", "aux data 1")
AUX_DATA_2_DIR = os.path.join(WORKSPACE_DIR, "DATA", "aux data 2")

RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "inference_results")

# Mean-Reversion Exhaustion Threshold (Section 5): -0.3% expected drawdown
EXHAUSTION_THRESHOLD = -0.003

# Exact feature names expected by Model 1 (Direction) and Model 2 (Price) (29 columns: 11 technical + 18 peer)
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

# Exact feature names expected by Model 3 (Exhaustion) & Model 3b (Runup) (16 columns)
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


# ============================================================================
# DATA STANDARDIZATION FUNCTIONS
# ============================================================================
def standardize_aux1_csv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Standardizes an Aux Data 1 stock CSV into clean OHLCV format.
    
    The raw CSV contains:
        Datetime, Close, High, Low, Open, Volume, ret_1, ret_lag_1, ..., target
        
    We strip ALL precomputed features and target columns so our pipeline engineers
    features completely from scratch using true shift(1) leakage-free calculations.
    We also strip timezones to ensure pure tz-naive datetime index operations.
    """
    rename_map = {}
    for col in df.columns:
        c_low = col.strip().lower()
        if c_low in ("datetime", "timestamp", "time"):
            rename_map[col] = "date"
        elif c_low in ("open", "high", "low", "close", "volume"):
            rename_map[col] = c_low

    df = df.rename(columns=rename_map)
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"      -> [Data Clean] WARNING: Missing required columns for {symbol}: {missing}")
        return pd.DataFrame()

    clean_df = pd.DataFrame(df[required]).copy()

    # Convert date to tz-naive datetime to ensure seamless merging
    clean_df["date"] = pd.to_datetime(clean_df["date"])
    if hasattr(clean_df["date"].dt, "tz") and clean_df["date"].dt.tz is not None:
        clean_df["date"] = clean_df["date"].dt.tz_localize(None)

    clean_df = clean_df.dropna(subset=["open", "high", "low", "close", "volume"])
    return clean_df


def standardize_aux2_csv(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """
    Standardizes an Aux Data 2 index CSV into clean OHLCV format.
    
    Indices typically have:
        date, open, high, low, close, volume (often 0 for pure indices).
    """
    rename_map = {}
    for col in df.columns:
        c_low = col.strip().lower()
        if c_low in ("datetime", "timestamp", "time", "date"):
            rename_map[col] = "date"
        elif c_low in ("open", "high", "low", "close", "volume"):
            rename_map[col] = c_low

    df = df.rename(columns=rename_map)
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"      -> [Data Clean] WARNING: Missing required columns for {index_name}: {missing}")
        return pd.DataFrame()

    clean_df = pd.DataFrame(df[required]).copy()
    clean_df["date"] = pd.to_datetime(clean_df["date"])
    if hasattr(clean_df["date"].dt, "tz") and clean_df["date"].dt.tz is not None:
        clean_df["date"] = clean_df["date"].dt.tz_localize(None)

    clean_df = clean_df.dropna(subset=["open", "high", "low", "close", "volume"])
    return clean_df


# ============================================================================
# CROSS-ASSET PEER MOMENTUM INJECTION
# ============================================================================
def inject_contemporaneous_peers(
    df_5min: pd.DataFrame,
    symbol: str,
    day_folder: str | None = None,
) -> pd.DataFrame:
    """
    Injects 18 peer momentum features across all 3 correlation dimensions
    (raw returns, binary direction, 0.2% return spike).
    
    Priority:
    1. Check if the peer CSV exists in the same day directory (contemporaneous 2026 data).
    2. Fallback to historical parquet if peer CSV is not found.
    3. Fill missing or unmatched bars with 0.0.
    """
    print(f"    [Step 2/6: Cross-Asset Injection] Checking peer relationships for {symbol}...")
    
    if not os.path.exists(PEER_MAP_PATH):
        print("      -> peer_map.json not found. Setting default 18 peer columns to 0.0.")
        for col in DEFAULT_PEER_COLS:
            df_5min[col] = 0.0
        return df_5min

    with open(PEER_MAP_PATH) as f:
        peer_map = json.load(f)

    if symbol not in peer_map:
        print(f"      -> {symbol} not in peer_map.json. Setting default 18 peer columns to 0.0.")
        for col in DEFAULT_PEER_COLS:
            df_5min[col] = 0.0
        return df_5min

    sym_data = peer_map[symbol]
    if "correlated_peers_raw" in sym_data:
        dimensions = {
            "raw": sym_data["correlated_peers_raw"],
            "binary": sym_data["correlated_peers_binary"],
            "spike": sym_data["correlated_peers_spike"],
        }
    else:
        dimensions = {
            "raw": {"top_positive": sym_data.get("top_positive", []), "top_negative": sym_data.get("top_negative", [])},
            "binary": {"top_positive": [], "top_negative": []},
            "spike": {"top_positive": [], "top_negative": []},
        }

    for dim_name, dim_data in dimensions.items():
        pos_list = dim_data.get("top_positive", [])[:3]
        neg_list = dim_data.get("top_negative", [])[:3]
        print(f"      -> [{dim_name.upper()}] Peers: +{pos_list} | -{neg_list}")

        # Process Positive Peers
        for rank, peer in enumerate(pos_list):
            col_name = f"peer_pos_{dim_name}_{rank + 1}_return_lag1"
            peer_ret = _extract_peer_return(peer, day_folder)
            if peer_ret is not None:
                df_5min = df_5min.join(peer_ret.rename(col_name), how="left")
            else:
                df_5min[col_name] = 0.0

        # Process Negative Peers
        for rank, peer in enumerate(neg_list):
            col_name = f"peer_neg_{dim_name}_{rank + 1}_return_lag1"
            peer_ret = _extract_peer_return(peer, day_folder)
            if peer_ret is not None:
                df_5min = df_5min.join(peer_ret.rename(col_name), how="left")
            else:
                df_5min[col_name] = 0.0

    # Ensure all 18 columns exist and fill NaNs
    for col in DEFAULT_PEER_COLS:
        if col not in df_5min.columns:
            df_5min[col] = 0.0
        else:
            df_5min[col] = df_5min[col].fillna(0.0)

    return df_5min


def _extract_peer_return(peer: str, day_folder: str | None) -> pd.Series | None:
    """
    Helper to extract shift(1)-lagged 5-minute returns for a peer stock using Typical Price.
    Checks the local day folder first, then historical parquet.
    """
    # 1. Search in the day's folder
    if day_folder and os.path.exists(day_folder):
        matches = glob.glob(os.path.join(day_folder, f"{peer}_*.csv"))
        if matches:
            try:
                pdf_raw = pd.read_csv(matches[0])
                pdf_std = standardize_aux1_csv(pdf_raw, peer)
                if len(pdf_std) > 10:
                    p_5m = build_features_5min(pdf_std)
                    # 5-minute lagged return with shift(1) leakage guard (uses Typical Price in mid_price)
                    p_ret = (p_5m["mid_price"] / p_5m["mid_price"].shift(1) - 1.0).shift(1)
                    return p_ret
            except Exception:  # noqa: BLE001, S110
                pass

    # 2. Fallback to historical parquet if needed
    parquet_path = os.path.join(PARQUET_DIR, f"{peer}.parquet")
    if os.path.exists(parquet_path):
        try:
            p_raw = pd.read_parquet(parquet_path, columns=["date", "high", "low", "close"])
            p_raw["date"] = pd.to_datetime(p_raw["date"])
            p_raw = p_raw.set_index("date")
            p_mid = compute_reference_price(p_raw)
            p_ret = (p_mid / p_mid.shift(5) - 1.0).shift(1)
            return p_ret
        except Exception:  # noqa: BLE001, S110
            pass

    return None


# ============================================================================
# VERBOSE 3-MODEL INFERENCE ENGINE
# ============================================================================
def run_inference_verbose(
    df_raw: pd.DataFrame,
    symbol: str,
    model_1: DirectionalModel,
    model_2: PriceModel,
    model_3: ExhaustionModel,
    day_folder: str | None = None,
    is_stock: bool = True,
    portfolio_capital: float = 100000.0,
    conf_threshold: float = 0.52,
) -> pd.DataFrame:
    """
    Executes full 3-model inference with extensive logging of each step.
    
    Steps:
    1. Feature Engineering (1-min and 5-min) via feature_engine/pipeline.py
    2. Peer Momentum Injection via artifacts/peer_map.json
    3. Feature Matrix Assembly and Exact Schema Verification
    4. Parallel 3-Model XGBoost Predictions
    5. Quantitative Signal Combiner (Direction, Price Move, Drawdown)
    6. Risk Manager Enforcement (Kill Switch, ATR Sizing, Exposure Limits)
    """
    t_min = df_raw["date"].min() if "date" in df_raw.columns else "N/A"
    t_max = df_raw["date"].max() if "date" in df_raw.columns else "N/A"
    print(f"\n  {'=' * 75}")
    print(f"  [START INFERENCE] Analyzing Ticker: {symbol} | Raw Bars: {len(df_raw):,}")
    print(f"  [DATA TIMELINE]   Start: {t_min}  -->  End: {t_max}")
    print(f"  {'=' * 75}")
    t_start = time.time()

    # -----------------------------------------------------------------------
    # STEP 1: Feature Engineering
    # -----------------------------------------------------------------------
    print("    [Step 1/6: Feature Engine] Computing Leakage-Free Z-Scores & Indicators...")
    print("      -> Calling feature_engine/pipeline.py: build_features_1min()")
    print("         (Computes return/log_ret/vol/volatility Z-scores, RSI 14, VWAP distance, Volume Z)")
    df_1min = build_features_1min(df_raw)
    print(f"         => 1-Min Feature Matrix: {df_1min.shape[0]:,} bars x {df_1min.shape[1]} columns")

    print("      -> Calling feature_engine/pipeline.py: build_features_5min()")
    print("         (Resamples 1m -> 5m via Aggregator, then computes 5m Z-scores + momentum)")
    df_5min = build_features_5min(df_raw)
    print(f"         => 5-Min Feature Matrix: {df_5min.shape[0]:,} bars x {df_5min.shape[1]} columns")

    if len(df_5min) < 3 or len(df_1min) < 3:
        print(f"      -> [SKIP] Insufficient clean bars after rolling window warmup (5m={len(df_5min)}, 1m={len(df_1min)})")
        return pd.DataFrame()

    # -----------------------------------------------------------------------
    # STEP 2: Peer Momentum Injection
    # -----------------------------------------------------------------------
    if is_stock:
        df_5min = inject_contemporaneous_peers(df_5min, symbol, day_folder=day_folder)
    else:
        print("    [Step 2/6: Cross-Asset Injection] Index Asset: Populating 18 peer columns with 0.0.")
        for col in DEFAULT_PEER_COLS:
            df_5min[col] = 0.0

    # -----------------------------------------------------------------------
    # STEP 3: Feature Matrix Assembly & Strict Validation
    # -----------------------------------------------------------------------
    print("    [Step 3/6: Feature Matrix Assembly] Verifying XGBoost Schema Alignment...")
    
    # Verify Model 1 & 2 Features (15 columns)
    missing_5m = [c for c in EXPECTED_FEATURES_5MIN if c not in df_5min.columns]
    if missing_5m:
        print(f"      -> ERROR: Missing 5-min features for Models 1 & 2: {missing_5m}")
        return pd.DataFrame()
    X_5min = pd.DataFrame(df_5min[EXPECTED_FEATURES_5MIN]).replace([np.inf, -np.inf], np.nan).dropna()

    # Verify Model 3 Features (12 columns)
    missing_1m = [c for c in EXPECTED_FEATURES_1MIN if c not in df_1min.columns]
    if missing_1m:
        print(f"      -> ERROR: Missing 1-min features for Model 3: {missing_1m}")
        return pd.DataFrame()
    X_1min = pd.DataFrame(df_1min[EXPECTED_FEATURES_1MIN]).replace([np.inf, -np.inf], np.nan).dropna()

    print(f"      -> [XGBoost Input] X_5min (Models 1 & 2): {X_5min.shape[0]:,} rows x {X_5min.shape[1]} features")
    print(f"      -> [XGBoost Input] X_1min (Model 3):       {X_1min.shape[0]:,} rows x {X_1min.shape[1]} features")

    if len(X_5min) == 0 or len(X_1min) == 0:
        print("      -> [SKIP] No valid feature rows remaining after NaN cleanup.")
        return pd.DataFrame()

    # -----------------------------------------------------------------------
    # STEP 4: Run 3-Model Predictions
    # -----------------------------------------------------------------------
    print("    [Step 4/6: Model Predictions] Executing Model Inference...")

    # Model 1: Directional Classification (XGBClassifier)
    print("      -> [Model 1: DirectionalModel] Calling predict() for next 5-min close direction...")
    pred_dir, prob_dir = model_1.predict(X_5min)
    up_cnt = int(np.sum(pred_dir == 1))
    down_cnt = int(np.sum(pred_dir == 0))
    avg_conf = float(np.mean(prob_dir))
    print(f"         Direction Predictions: UP = {up_cnt} ({up_cnt/len(pred_dir)*100:.1f}%) | "
          f"DOWN = {down_cnt} ({down_cnt/len(pred_dir)*100:.1f}%) | Avg Confidence = {avg_conf:.4f}")

    # Model 2: Price Return Regression (XGBRegressor)
    print("      -> [Model 2: PriceModel] Calling predict() for first 1-min return of next 5m bar...")
    pred_price = model_2.predict(X_5min)
    print(f"         Price Move (1m Return): Mean = {np.mean(pred_price):.6f}, "
          f"Std = {np.std(pred_price):.6f}, Min = {np.min(pred_price):.6f}, Max = {np.max(pred_price):.6f}")

    # Model 3: Exhaustion Drawdown Regression (XGBRegressor)
    print("      -> [Model 3: ExhaustionModel] Calling predict() for forward 10-min max drawdown...")
    pred_drawdown = model_3.predict(X_1min)
    print(f"         Drawdown Predictions: Mean = {np.mean(pred_drawdown):.6f}, "
          f"Std = {np.std(pred_drawdown):.6f}, Min = {np.min(pred_drawdown):.6f}, Max = {np.max(pred_drawdown):.6f}")

    # -----------------------------------------------------------------------
    # STEP 5: Quantitative Signal Combiner & Risk Sizing
    # -----------------------------------------------------------------------
    print("    [Step 5/6: Signal Combiner & Risk Management] Evaluating Rules & Portfolio Constraints...")
    atr_series = calculate_atr(df_5min, period=14)
    exposure_mgr = ExposureManager(max_correlated_positions=2, correlation_threshold=0.7)

    min_len = min(len(pred_dir), len(pred_price), len(pred_drawdown))
    signals = []

    for i in range(min_len):
        direction = int(pred_dir[i])
        confidence = float(prob_dir[i])
        price_move = float(pred_price[i])
        drawdown = float(pred_drawdown[i])

        # Reference price from 5m dataframe
        if "mid_price" in df_5min.columns:
            current_price = float(df_5min["mid_price"].iloc[i])
        else:
            current_price = float(df_5min["close"].iloc[i])

        current_date = df_5min.index[i] if isinstance(df_5min.index, pd.DatetimeIndex) else i

        signal_type = "HOLD"
        entry_price = current_price

        # Combination Logic (Specification Section 5):
        # confidence is P(UP). For DOWN direction (0), P(DOWN) is (1.0 - confidence).
        p_up = confidence
        p_down = 1.0 - confidence

        if direction == 1 and drawdown < EXHAUSTION_THRESHOLD:
            # Mean Reversion Buy: Buy the dip at expected forward drawdown level
            entry_price = current_price * (1.0 - abs(drawdown))
            signal_type = "MEAN_REVERSION_BUY"
        elif direction == 1 and p_up >= conf_threshold and price_move > 0:
            # Momentum Buy: Strong upward probability confirmed by Model 2 positive drift
            signal_type = "MOMENTUM_BUY"
        elif direction == 0 and p_down >= conf_threshold and price_move < 0:
            # Momentum Sell: Strong downward probability confirmed by Model 2 negative drift
            signal_type = "MOMENTUM_SELL"

        # Risk Management & ATR Position Sizing
        current_atr = (
            float(atr_series.iloc[i])
            if i < len(atr_series) and not np.isnan(atr_series.iloc[i])
            else current_price * 0.005
        )

        shares = 0.0
        position_value = 0.0

        if signal_type != "HOLD":
            if exposure_mgr.can_open_position(symbol):
                shares = calculate_position_size(
                    capital=portfolio_capital,
                    risk_pct=0.02,
                    atr=current_atr,
                    price=entry_price,
                    max_position_pct=0.10,
                )
                position_value = round(shares * entry_price, 2)
            else:
                signal_type = "BLOCKED_BY_EXPOSURE_CAP"

        signals.append({
            "timestamp": str(current_date),
            "symbol": symbol,
            "direction": 1 if "BUY" in signal_type else (-1 if "SELL" in signal_type else 0),
            "confidence": round(confidence, 4),
            "predicted_price_move": round(price_move, 6),
            "predicted_drawdown": round(drawdown, 6),
            "entry_price": round(entry_price, 2),
            "current_price": round(current_price, 2),
            "atr_14": round(current_atr, 4),
            "shares": int(shares),
            "position_value": position_value,
            "signal_type": signal_type,
        })

    signals_df = pd.DataFrame(signals)

    # -----------------------------------------------------------------------
    # STEP 6: Execution Report
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"    [Step 6/6: Summary] Completed in {elapsed:.2f}s | Analyzed Bars: {len(signals_df):,}")
    if len(signals_df) > 0:
        counts = dict(Counter(signals_df["signal_type"]))
        print(f"      -> Actionable Breakdown: {counts}")
        actionable = signals_df[signals_df["signal_type"].isin(
            ["MEAN_REVERSION_BUY", "MOMENTUM_BUY", "MOMENTUM_SELL"]
        )]
        if len(actionable) > 0:
            sample = actionable.iloc[0]
            print(f"      -> [First Trigger] At {sample['timestamp']} | Signal: {sample['signal_type']} | "
                  f"Entry: INR {sample['entry_price']} | Shares: {sample['shares']} | Value: INR {sample['position_value']:,}")

    return signals_df


# ============================================================================
# PIPELINE PROCESSOR: AUX DATA 1 (NIFTY 50 Constituents)
# ============================================================================
def process_aux_data_1(
    model_1: DirectionalModel,
    model_2: PriceModel,
    model_3: ExhaustionModel,
    max_days: int | None = None,
    max_stocks_per_day: int | None = None,
) -> pd.DataFrame:
    """
    Discovers, standardizes, and processes NIFTY50 constituent stocks from aux data 1.
    """
    print("\n" + "=" * 80)
    print("EXECUTING PIPELINE: AUX DATA 1 (NIFTY 50 Constituents)")
    print(f"Source Directory: {AUX_DATA_1_DIR}")
    print("=" * 80)

    if not os.path.exists(AUX_DATA_1_DIR):
        print(f"ERROR: Aux Data 1 path not found: {AUX_DATA_1_DIR}")
        return pd.DataFrame()

    day_folders = sorted([
        d for d in os.listdir(AUX_DATA_1_DIR)
        if os.path.isdir(os.path.join(AUX_DATA_1_DIR, d))
    ])
    print(f"Discovered {len(day_folders)} trading day folders in Aux Data 1.")

    if max_days:
        day_folders = day_folders[:max_days]
        print(f"Limiting execution to first {max_days} days for testing.")

    all_signals: list[pd.DataFrame] = []

    for day_idx, day_name in enumerate(day_folders):
        print(f"\n>>> [Day {day_idx + 1}/{len(day_folders)}] Session: {day_name}")
        
        # Look for nested directory if present
        nested_path = os.path.join(AUX_DATA_1_DIR, day_name, day_name)
        active_day_dir = nested_path if os.path.exists(nested_path) else os.path.join(AUX_DATA_1_DIR, day_name)
        
        csv_files = sorted(glob.glob(os.path.join(active_day_dir, "*.csv")))
        print(f"    Found {len(csv_files)} constituent CSVs in {os.path.basename(active_day_dir)}/")

        if max_stocks_per_day:
            csv_files = csv_files[:max_stocks_per_day]

        for s_idx, csv_path in enumerate(csv_files):
            filename = os.path.basename(csv_path)
            symbol = filename.split("_")[0]

            try:
                raw_csv = pd.read_csv(csv_path)
                std_df = standardize_aux1_csv(raw_csv, symbol)

                if std_df.empty:
                    continue

                t_start = std_df["date"].min() if "date" in std_df.columns else "N/A"
                t_end = std_df["date"].max() if "date" in std_df.columns else "N/A"
                print("\n  +--------------------------------------------------------------------------------")
                print(f"  | [EXACT FILE CHOSEN] : {csv_path}")
                print(f"  | [DATA TIMELINE]     : Start: {t_start}  -->  End: {t_end} ({len(std_df):,} raw bars)")
                print("  +--------------------------------------------------------------------------------")

                signals = run_inference_verbose(
                    df_raw=std_df,
                    symbol=symbol,
                    model_1=model_1,
                    model_2=model_2,
                    model_3=model_3,
                    day_folder=active_day_dir,
                    is_stock=True,
                )

                if not signals.empty:
                    signals["source_day"] = day_name
                    signals["source_file"] = filename
                    all_signals.append(signals)

            except Exception as e:  # noqa: BLE001
                print(f"    ERROR processing {filename}: {e}")

            gc.collect()

    if all_signals:
        merged = pd.concat(all_signals, ignore_index=True)
        print(f"\n[AUX DATA 1 FINISHED] Generated {len(merged):,} signal evaluations across {len(all_signals)} stocks.")
        return merged
    return pd.DataFrame()


# ============================================================================
# PIPELINE PROCESSOR: AUX DATA 2 (Indices & Benchmarks)
# ============================================================================
def process_aux_data_2(
    model_1: DirectionalModel,
    model_2: PriceModel,
    model_3: ExhaustionModel,
    max_files: int | None = None,
) -> pd.DataFrame:
    """
    Discovers, standardizes, and processes broad and sectoral indices from aux data 2.
    Only processes *_minute.csv files for high-frequency resolution.
    """
    print("\n" + "=" * 80)
    print("EXECUTING PIPELINE: AUX DATA 2 (Index-Level Minute Data)")
    print(f"Source Directory: {AUX_DATA_2_DIR}")
    print("=" * 80)

    if not os.path.exists(AUX_DATA_2_DIR):
        print(f"ERROR: Aux Data 2 path not found: {AUX_DATA_2_DIR}")
        return pd.DataFrame()

    minute_files = sorted(glob.glob(os.path.join(AUX_DATA_2_DIR, "*_minute.csv")))
    print(f"Discovered {len(minute_files)} 1-minute index CSV files.")

    if max_files:
        minute_files = minute_files[:max_files]
        print(f"Limiting execution to first {max_files} index files for testing.")

    all_signals: list[pd.DataFrame] = []

    for f_idx, csv_path in enumerate(minute_files):
        filename = os.path.basename(csv_path)
        index_symbol = filename.replace("_minute.csv", "").replace(" ", "_")

        print(f"\n>>> [Index {f_idx + 1}/{len(minute_files)}] Asset: {index_symbol} ({filename})")

        try:
            # Read sample or full
            raw_csv = pd.read_csv(csv_path)
            std_df = standardize_aux2_csv(raw_csv, index_symbol)

            if std_df.empty:
                continue

            # If the index has over 50k rows, take the most recent 5,000 bars for prompt testing
            if len(std_df) > 5000:
                print(f"    Index has {len(std_df):,} bars. Evaluating latest 5,000 bars for inference.")
                std_df = std_df.tail(5000).reset_index(drop=True)

            t_start = std_df["date"].min() if "date" in std_df.columns else "N/A"
            t_end = std_df["date"].max() if "date" in std_df.columns else "N/A"
            print("  +--------------------------------------------------------------------------------")
            print(f"  | [EXACT FILE CHOSEN] : {csv_path}")
            print(f"  | [DATA TIMELINE]     : Start: {t_start}  -->  End: {t_end} ({len(std_df):,} raw bars)")
            print("  +--------------------------------------------------------------------------------")

            signals = run_inference_verbose(
                df_raw=std_df,
                symbol=index_symbol,
                model_1=model_1,
                model_2=model_2,
                model_3=model_3,
                day_folder=None,
                is_stock=False,
            )

            if not signals.empty:
                signals["source_file"] = filename
                all_signals.append(signals)

        except Exception as e:  # noqa: BLE001
            print(f"    ERROR processing {filename}: {e}")

        gc.collect()

    if all_signals:
        merged = pd.concat(all_signals, ignore_index=True)
        print(f"\n[AUX DATA 2 FINISHED] Generated {len(merged):,} signal evaluations across {len(all_signals)} index files.")
        return merged
    return pd.DataFrame()


# ============================================================================
# SINGLE-STOCK ACCURACY & METRICS EVALUATOR
# ============================================================================
def evaluate_single_stock_metrics(
    symbol: str,
    model_1: DirectionalModel,
    model_2: PriceModel,
    model_3: ExhaustionModel,
    day_name: str | None = None,
    all_days: bool = False,
    conf_threshold: float = 0.52,
) -> pd.DataFrame:
    """
    Evaluates inference predictions for a single stock against ground-truth future data.
    
    Computes rigorous quantitative metrics:
      - Model 1: Directional Accuracy (%), High-Confidence Accuracy, Precision, Recall, F1, Confusion Matrix
      - Model 2: Return Prediction MAE, RMSE, Directional Sign Match (%), Pearson Correlation
      - Model 3: Drawdown Prediction MAE, RMSE, Flush Detection Correlation
      - Strategy: Simulated trade P&L, Win Rate (%), Profit Factor, Cumulative Return
    """
    print("\n" + "=" * 80)
    print(f"  SINGLE-STOCK ACCURACY & PERFORMANCE AUDIT: {symbol}")
    print(f"  Source: {AUX_DATA_1_DIR}")
    print("=" * 80)

    # 1. Discover matching CSV files for this symbol
    all_day_dirs = sorted([
        d for d in os.listdir(AUX_DATA_1_DIR)
        if os.path.isdir(os.path.join(AUX_DATA_1_DIR, d))
    ])

    matched_files = []
    for d in all_day_dirs:
        nested = os.path.join(AUX_DATA_1_DIR, d, d)
        active_dir = nested if os.path.exists(nested) else os.path.join(AUX_DATA_1_DIR, d)
        csv_path = os.path.join(active_dir, f"{symbol}_{d.replace('nifty50_', '')}.csv")
        if not os.path.exists(csv_path):
            # Fallback pattern match
            matches = glob.glob(os.path.join(active_dir, f"{symbol}_*.csv"))
            if matches:
                csv_path = matches[0]
        if os.path.exists(csv_path):
            matched_files.append((d, active_dir, csv_path))

    if not matched_files:
        print(f"  ERROR: No data files found for {symbol} in {AUX_DATA_1_DIR}!")
        return pd.DataFrame()

    print(f"  Found {len(matched_files)} daily sessions for {symbol} in Aux Data 1.")

    if not all_days:
        if day_name and day_name.lower() != "all":
            matched_files = [m for m in matched_files if day_name.lower() in m[0].lower()]
            if not matched_files:
                print(f"  ERROR: Specified day '{day_name}' not found for {symbol}!")
                return pd.DataFrame()
        else:
            # Default to the first available day
            matched_files = matched_files[:1]
            print(f"  Evaluating single session: {matched_files[0][0]} (use --all-days to evaluate all {len(all_day_dirs)} days)")
    else:
        print(f"  Evaluating across ALL {len(matched_files)} available trading sessions.")

    # Data collection for aggregate metrics
    all_eval_rows = []
    session_info: list[dict[str, object]] = []

    for day_label, active_dir, csv_path in matched_files:
        raw_csv = pd.read_csv(csv_path)
        std_df = standardize_aux1_csv(raw_csv, symbol)

        if len(std_df) < 20:
            print(f"      -> SKIP: Insufficient rows in CSV ({csv_path})")
            continue

        t_start = std_df["date"].min() if "date" in std_df.columns else "N/A"
        t_end = std_df["date"].max() if "date" in std_df.columns else "N/A"
        session_info.append({
            "session": day_label,
            "file": csv_path,
            "filename": os.path.basename(csv_path),
            "start": str(t_start),
            "end": str(t_end),
            "bars": len(std_df),
        })

        print("\n  +--------------------------------------------------------------------------------")
        print(f"  | [EXACT FILE CHOSEN] : {csv_path}")
        print(f"  | [SESSION IDENTIFIER]: {day_label} (Symbol: {symbol})")
        print(f"  | [DATA TIMELINE]     : Start: {t_start}  -->  End: {t_end}")
        print(f"  | [RAW BARS LOADED]   : {len(std_df):,} 1-minute OHLCV bars")
        print("  +--------------------------------------------------------------------------------")

        # Feature engineering
        print(f"      -> Engineering 1-min & 5-min features for {symbol}...")
        df_1min = build_features_1min(std_df)
        df_5min = build_features_5min(std_df)

        if len(df_5min) < 3 or len(df_1min) < 3:
            print("      -> SKIP: Insufficient bars after feature engineering")
            continue

        # Inject contemporaneous peer features
        df_5min = inject_contemporaneous_peers(df_5min, symbol, day_folder=active_dir)

        # Assemble clean feature matrices
        X_5min = pd.DataFrame(df_5min[EXPECTED_FEATURES_5MIN]).replace([np.inf, -np.inf], np.nan).dropna()
        X_1min = pd.DataFrame(df_1min[EXPECTED_FEATURES_1MIN]).replace([np.inf, -np.inf], np.nan).dropna()

        # Run 3-model predictions
        pred_dir, prob_dir = model_1.predict(X_5min)
        pred_price = model_2.predict(X_5min)
        pred_drawdown = model_3.predict(X_1min)

        # -------------------------------------------------------------------
        # COMPUTE GROUND TRUTH FOR THIS SESSION
        # -------------------------------------------------------------------
        # Ground Truth 1 (Direction): Did the next 5-min Typical Price go UP?
        ref_5m = compute_reference_price(df_5min) if all(c in df_5min.columns for c in ['high', 'low', 'close']) else df_5min['close']
        next_ref_5m = ref_5m.shift(-1).reindex(X_5min.index)
        curr_ref_5m = ref_5m.reindex(X_5min.index)
        y_true_dir = (next_ref_5m > curr_ref_5m).astype(float)

        # Ground Truth 2 (Price Move): Actual return of next 5-min Typical Price
        y_true_ret = (next_ref_5m - curr_ref_5m) / curr_ref_5m

        # Ground Truth 3 (Drawdown): Actual forward 10-bar minimum drawdown on 1-min data
        reversed_low = df_1min["low"].iloc[::-1]
        rolling_min_low = reversed_low.rolling(10, min_periods=10).min().iloc[::-1].shift(-1).reindex(X_1min.index)
        ref_1m = compute_reference_price(df_1min) if all(c in df_1min.columns for c in ['high', 'low', 'close']) else df_1min['close']
        curr_ref_1m = ref_1m.reindex(X_1min.index)
        y_true_drawdown = ((rolling_min_low - curr_ref_1m) / curr_ref_1m).clip(lower=-0.15, upper=0.0)

        # Align lengths
        min_len = min(len(X_5min), len(pred_dir), len(pred_price))
        for i in range(min_len):
            idx_time = X_5min.index[i]
            # Match 1-min drawdown by timestamp if possible, else use index
            m3_val = pred_drawdown[min(i, len(pred_drawdown) - 1)]
            m3_actual = y_true_drawdown.iloc[min(i, len(y_true_drawdown) - 1)] if len(y_true_drawdown) > i else np.nan

            p_dir = int(pred_dir[i])
            conf = float(prob_dir[i])
            p_price = float(pred_price[i])
            act_dir = y_true_dir.iloc[i]
            act_ret = y_true_ret.iloc[i]

            # Signal classification
            # conf is P(UP). For DOWN direction, P(DOWN) is (1.0 - conf).
            p_up = conf
            p_down = 1.0 - conf

            sig_type = "HOLD"
            if p_dir == 1 and m3_val < EXHAUSTION_THRESHOLD:
                sig_type = "MEAN_REVERSION_BUY"
            elif p_dir == 1 and p_up >= conf_threshold and p_price > 0:
                sig_type = "MOMENTUM_BUY"
            elif p_dir == 0 and p_down >= conf_threshold and p_price < 0:
                sig_type = "MOMENTUM_SELL"

            # Strategy return:
            strat_ret = 0.0
            if not np.isnan(act_ret):
                if "BUY" in sig_type:
                    strat_ret = act_ret
                elif "SELL" in sig_type:
                    strat_ret = -act_ret

            all_eval_rows.append({
                "timestamp": str(idx_time),
                "session": day_label,
                "symbol": symbol,
                "pred_direction": p_dir,
                "confidence": conf,
                "actual_direction": act_dir,
                "pred_price_return": p_price,
                "actual_price_return": act_ret,
                "pred_drawdown": m3_val,
                "actual_drawdown": m3_actual,
                "signal_type": sig_type,
                "strategy_return": strat_ret,
            })

    if not all_eval_rows:
        print("  ERROR: No evaluated rows produced.")
        return pd.DataFrame()

    eval_df = pd.DataFrame(all_eval_rows)

    # -----------------------------------------------------------------------
    # QUANTITATIVE METRICS COMPUTATION
    # -----------------------------------------------------------------------
    # Filter valid rows (last bar has NaN future)
    valid_m1 = eval_df.dropna(subset=["actual_direction"])
    y_true_m1 = valid_m1["actual_direction"].to_numpy(dtype=int)
    y_pred_m1 = np.asarray(valid_m1["pred_direction"], dtype=int)
    y_conf_m1 = valid_m1["confidence"].to_numpy(dtype=float)

    total_bars = len(y_true_m1)
    correct_m1 = int(np.sum(y_pred_m1 == y_true_m1))
    acc_m1 = (correct_m1 / total_bars) * 100.0 if total_bars > 0 else 0.0

    # High-confidence metrics (>60% or <40%)
    high_conf_mask = (y_conf_m1 >= 0.60) | (y_conf_m1 <= 0.40)
    high_conf_total = int(np.sum(high_conf_mask))
    if high_conf_total > 0:
        high_conf_correct = int(np.sum(y_pred_m1[high_conf_mask] == y_true_m1[high_conf_mask]))
        high_conf_acc = (high_conf_correct / high_conf_total) * 100.0
    else:
        high_conf_acc = 0.0

    # Confusion matrix for Model 1
    tp = int(np.sum((y_pred_m1 == 1) & (y_true_m1 == 1)))
    fp = int(np.sum((y_pred_m1 == 1) & (y_true_m1 == 0)))
    tn = int(np.sum((y_pred_m1 == 0) & (y_true_m1 == 0)))
    fn = int(np.sum((y_pred_m1 == 0) & (y_true_m1 == 1)))

    precision_up = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall_up = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    f1_up = (2 * precision_up * recall_up / (precision_up + recall_up)) if (precision_up + recall_up) > 0 else 0.0
    baseline_majority = max(float(np.mean(y_true_m1)), 1.0 - float(np.mean(y_true_m1))) * 100.0

    # Model 2 Metrics (Price Return Regression)
    valid_m2 = eval_df.dropna(subset=["actual_price_return"])
    act_ret_m2 = valid_m2["actual_price_return"].to_numpy(dtype=float)
    pred_ret_m2 = valid_m2["pred_price_return"].to_numpy(dtype=float)

    mae_m2 = float(np.mean(np.abs(pred_ret_m2 - act_ret_m2)))
    rmse_m2 = float(np.sqrt(np.mean((pred_ret_m2 - act_ret_m2) ** 2)))
    sign_match_m2 = float(np.mean(np.sign(pred_ret_m2) == np.sign(act_ret_m2)) * 100.0)
    corr_m2 = float(np.corrcoef(pred_ret_m2, act_ret_m2)[0, 1]) if len(pred_ret_m2) > 2 else 0.0

    # Model 3 Metrics (Drawdown Regression)
    valid_m3 = eval_df.dropna(subset=["actual_drawdown"])
    act_dd_m3 = valid_m3["actual_drawdown"].to_numpy(dtype=float)
    pred_dd_m3 = valid_m3["pred_drawdown"].to_numpy(dtype=float)

    mae_m3 = float(np.mean(np.abs(pred_dd_m3 - act_dd_m3)))
    rmse_m3 = float(np.sqrt(np.mean((pred_dd_m3 - act_dd_m3) ** 2)))
    corr_m3 = float(np.corrcoef(pred_dd_m3, act_dd_m3)[0, 1]) if len(pred_dd_m3) > 2 else 0.0

    # Flush capture: actual drawdown < -0.3%
    actual_flushes = int(np.sum(act_dd_m3 < EXHAUSTION_THRESHOLD))

    # Strategy Backtest Metrics
    trade_signals = eval_df[eval_df["signal_type"].isin(["MOMENTUM_BUY", "MEAN_REVERSION_BUY", "MOMENTUM_SELL"])].copy()
    num_trades = len(trade_signals)
    winning_trades = int(np.sum(trade_signals["strategy_return"] > 0)) if num_trades > 0 else 0
    losing_trades = int(np.sum(trade_signals["strategy_return"] < 0)) if num_trades > 0 else 0
    win_rate = (winning_trades / num_trades * 100.0) if num_trades > 0 else 0.0

    gains = trade_signals.loc[trade_signals["strategy_return"] > 0, "strategy_return"].sum()
    losses = abs(trade_signals.loc[trade_signals["strategy_return"] < 0, "strategy_return"].sum())
    profit_factor = (gains / losses) if losses > 0 else (999.0 if gains > 0 else 0.0)
    cum_strategy_ret = (trade_signals["strategy_return"].sum() * 100.0) if num_trades > 0 else 0.0
    cum_benchmark_ret = (valid_m2["actual_price_return"].sum() * 100.0) if len(valid_m2) > 0 else 0.0

    # -----------------------------------------------------------------------
    # PRINT FORMATTED METRICS DASHBOARD
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"          INFERENCE ACCURACY & PERFORMANCE DASHBOARD: {symbol}")
    print("=" * 80)
    if len(session_info) == 1:
        print(f"  Exact File Chosen       : {session_info[0]['file']}")
        print(f"  Data Timeline (Start)   : {session_info[0]['start']}")
        print(f"  Data Timeline (End)     : {session_info[0]['end']}")
        print(f"  Raw 1-Min Bars Loaded   : {session_info[0]['bars']:,}")
    elif len(session_info) > 1:
        print(f"  Files Evaluated ({len(session_info)} files across {len(matched_files)} session(s)):")
        for s_item in session_info:
            print(f"    * [{s_item['session']}] {s_item['file']}")
            print(f"      Timeline: {s_item['start']}  -->  {s_item['end']} ({s_item['bars']:,} bars)")
        valid_starts = [str(s["start"]) for s in session_info if s["start"] != "N/A"]
        valid_ends = [str(s["end"]) for s in session_info if s["end"] != "N/A"]
        if valid_starts and valid_ends:
            print(f"  Overall Data Timeline   : Start: {min(valid_starts)}  -->  End: {max(valid_ends)}")
    print(f"  Sessions Analyzed       : {len(matched_files)} day(s)")
    print(f"  Total 5-min Bars        : {total_bars:,}")
    print(f"  Market Baseline (UP %)  : {baseline_majority:.2f}%")

    print("\n  [MODEL 1: 5-MIN DIRECTIONAL CLASSIFIER (XGBClassifier)]")
    print("  " + "-" * 60)
    print(f"  Overall Directional Accuracy : {acc_m1:.2f}%  ({correct_m1}/{total_bars} correct)")
    print(f"  High-Conviction Accuracy    : {high_conf_acc:.2f}%  ({high_conf_total} bars with prob > 60% or < 40%)")
    print(f"  UP Precision                : {precision_up:.2f}%")
    print(f"  UP Recall                   : {recall_up:.2f}%")
    print(f"  UP F1-Score                 : {f1_up:.2f}%")
    print(f"  Confusion Matrix Breakdown  : TP={tp} | FP={fp} | TN={tn} | FN={fn}")

    print("\n  [MODEL 2: 1-MIN RETURN REGRESSION (XGBRegressor)]")
    print("  " + "-" * 60)
    print(f"  Mean Absolute Error (MAE)   : {mae_m2:.6f}  ({mae_m2*100:.4f}%)")
    print(f"  Root Mean Sq Error (RMSE)   : {rmse_m2:.6f}  ({rmse_m2*100:.4f}%)")
    print(f"  Directional Sign Match Rate : {sign_match_m2:.2f}%")
    print(f"  Pearson Correlation (r)     : {corr_m2:.4f}")

    print("\n  [MODEL 3: 10-MIN EXHAUSTION DRAWDOWN REGRESSION (XGBRegressor)]")
    print("  " + "-" * 60)
    print(f"  Mean Absolute Error (MAE)   : {mae_m3:.6f}  ({mae_m3*100:.4f}%)")
    print(f"  Root Mean Sq Error (RMSE)   : {rmse_m3:.6f}  ({rmse_m3*100:.4f}%)")
    print(f"  Drawdown Correlation (r)    : {corr_m3:.4f}")
    print(f"  Actual Severe Flushes (<-0.3%): {actual_flushes} detected in market")

    print("\n  [3-MODEL QUANTITATIVE TRADING STRATEGY PERFORMANCE]")
    print("  " + "-" * 60)
    print(f"  Actionable Signals Generated: {num_trades}")
    print(f"  Signal Breakdown            : {dict(Counter(trade_signals['signal_type']))}")
    print(f"  Win Rate                    : {win_rate:.2f}%  ({winning_trades} wins, {losing_trades} losses)")
    print(f"  Profit Factor               : {profit_factor:.2f}")
    print(f"  Cumulative Strategy Return  : {cum_strategy_ret:+.2f}%")
    print(f"  Buy-and-Hold Return (Ref)   : {cum_benchmark_ret:+.2f}%")
    print("=" * 80)

    # Save detailed evaluation rows
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_eval_csv = os.path.join(RESULTS_DIR, f"{symbol}_accuracy_evaluation.csv")
    eval_df.to_csv(out_eval_csv, index=False)
    print(f"\n  [SAVED] Detailed per-bar evaluation saved to:\n  -> {out_eval_csv}\n")

    return eval_df


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model XGBoost Inference & Accuracy Evaluation Pipeline for Aux Data 1 & 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="RELIANCE",
        help="Stock ticker from aux data 1 to evaluate (default: RELIANCE)",
    )
    parser.add_argument(
        "--day",
        type=str,
        default="nifty50_01Jun",
        help="Specific trading session to evaluate (default: nifty50_01Jun). Use 'all' for all days.",
    )
    parser.add_argument(
        "--all-days",
        action="store_true",
        help="Evaluate the chosen stock across ALL 23 days in aux data 1",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.52,
        help="Confidence hurdle for directional trade entry (default: 0.52 / 52%%)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run multi-stock and multi-index batch runner across full datasets instead of single stock",
    )
    args = parser.parse_args()

    print("*" * 80)
    print("  MULTI-MODEL XGBOOST INFERENCE & ACCURACY ENGINE")
    print("*" * 80)
    print(f"  Artifacts Location : {ARTIFACTS_DIR}")
    print(f"  Aux Data 1 Source  : {AUX_DATA_1_DIR}")
    print(f"  Aux Data 2 Source  : {AUX_DATA_2_DIR}")
    print(f"  Output Directory   : {RESULTS_DIR}")
    print(f"  Target Stock       : {args.symbol}")
    print(f"  Target Session     : {'ALL 23 Days' if args.all_days else args.day}")
    print(f"  Confidence Hurdle  : {args.confidence * 100:.1f}%")

    # 1. Kill Switch Check
    print("\n[Regulatory Risk] Verifying Kill Switch State...")
    if check_kill_switch():
        print("  [HALT] KILL SWITCH ACTIVATED! All inference aborted.")
        sys.exit(1)
    print("  [STATUS] Kill Switch is CLEAR. Proceeding.")

    # 2. Load Models
    print("\n[Model Loader] Loading Trained XGBoost Models from disk...")
    for p, name in [
        (MODEL_1_PATH, "Model 1 (Direction)"),
        (MODEL_2_PATH, "Model 2 (Price)"),
        (MODEL_3_PATH, "Model 3 (Exhaustion)"),
    ]:
        if not os.path.exists(p):
            print(f"  ERROR: {name} weights missing at {p}!")
            sys.exit(1)

    model_1 = DirectionalModel(MODEL_1_PATH)
    model_2 = PriceModel(MODEL_2_PATH)
    model_3 = ExhaustionModel(MODEL_3_PATH)
    print("  [SUCCESS] DirectionalModel, PriceModel, and ExhaustionModel loaded successfully.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 3. Execution Mode Selection
    if args.batch:
        # Full Batch Mode across multiple stocks and indices
        print("\n[MODE] Running Multi-Stock and Multi-Index Batch Evaluation...")
        signals_aux1 = process_aux_data_1(
            model_1=model_1,
            model_2=model_2,
            model_3=model_3,
            max_days=3,
            max_stocks_per_day=4,
        )
        if not signals_aux1.empty:
            out_aux1 = os.path.join(RESULTS_DIR, "aux_data_1_signals.csv")
            signals_aux1.to_csv(out_aux1, index=False)
            print(f"  [SAVED] Aux Data 1 signals written to: {out_aux1}")

        signals_aux2 = process_aux_data_2(
            model_1=model_1,
            model_2=model_2,
            model_3=model_3,
            max_files=3,
        )
        if not signals_aux2.empty:
            out_aux2 = os.path.join(RESULTS_DIR, "aux_data_2_signals.csv")
            signals_aux2.to_csv(out_aux2, index=False)
            print(f"  [SAVED] Aux Data 2 signals written to: {out_aux2}")

    else:
        # Single Stock Accuracy Evaluation Mode (DEFAULT)
        # Evaluates chosen stock (e.g. RELIANCE) and prints full accuracy & metrics dashboard
        evaluate_single_stock_metrics(
            symbol=args.symbol.upper(),
            model_1=model_1,
            model_2=model_2,
            model_3=model_3,
            day_name=args.day,
            all_days=args.all_days,
            conf_threshold=args.confidence,
        )


if __name__ == "__main__":
    main()

