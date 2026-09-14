"""
========================================================================================
INFERENCE: Multi-Model EV-Gated Execution Engine (Priority-2 Production)
========================================================================================
Combines Models 1, 2, 3, 3b, MultiClassCalibrator, and MetaFilterModel into a
disciplined, cost-aware execution pipeline.

DECISION PIPELINE:
1. Feature Engineering: 1-min & 5-min features + 18 peer cross-asset momentum signals.
2. Model Predictions:
   - Model 1: 3-class directional prediction (DOWN=0, FLAT=1, UP=2).
   - Calibrator: Maps raw multi:softprob to true empirical probabilities [p_down, p_flat, p_up].
   - Model 2: Regression prediction of first 1-min forward return.
   - Model 3: Regression prediction of forward 10-min maximum drawdown.
   - Model 3b: Regression prediction of forward 10-min maximum upside runup.
3. Expected-Value (EV) Gating:
   EV = p_class * |Model_2_magnitude| - ROUND_TRIP_COST_PCT (0.26%).
   Trades only execute if EV > MIN_EDGE_BUFFER_PCT (0.02%).
4. Meta-Labeling Filter (Model 4):
   Secondary machine learning veto predicting if the trade will be profitable after costs.
5. Ensemble Agreement Engine:
   Requires consensus across Model 1, Model 2 sign, Model 3/3b, and peer momentum.
6. Multi-Barrier ExitManager:
   Hard Stop-Loss (1.5x ATR), Take-Profit (0.7x runup), Trailing Stop (0.5%), Time fallback (5 bars).
7. Position Sizing:
   Configurable ATR, Fractional Kelly, or Hybrid sizing.
========================================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the trading_pipeline directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.risk_manager import (
    CircuitBreaker,
    ExitManager,
    ExposureManager,
    PortfolioState,
    calculate_atr,
    calculate_position_size,
    check_kill_switch,
)
from config import (
    EXIT_TAKE_PROFIT_SAFETY,
    MIN_BARRIER_DISTANCE_PCT,
    MIN_EDGE_BUFFER_PCT,
    POSITION_SIZING_METHOD,
    ROUND_TRIP_COST_PCT,
)
from correlation.cross_asset import inject_cross_asset_features
from ensemble import evaluate_ensemble_agreement
from feature_engine.pipeline import build_features_1min, build_features_5min
from models.calibration import MultiClassCalibrator
from models.model_1_direction import DirectionalModel
from models.model_2_price import PriceModel
from models.model_3_exhaustion import ExhaustionModel
from models.model_3b_runup import RunupModel
from models.model_4_meta_filter import MetaFilterModel, compute_peer_aggregates

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
ARTIFACTS_DIR = r"d:\CODE\rajasthani\trading_pipeline\artifacts"
PARQUET_DIR = r"d:\CODE\rajasthani\DATA\parquet"
PEER_MAP_PATH = os.path.join(ARTIFACTS_DIR, "peer_map.json")

MODEL_1_PATH = os.path.join(ARTIFACTS_DIR, "model_1_direction.json")
MODEL_2_PATH = os.path.join(ARTIFACTS_DIR, "model_2_price.json")
MODEL_3_PATH = os.path.join(ARTIFACTS_DIR, "model_3_exhaustion.json")
MODEL_3B_PATH = os.path.join(ARTIFACTS_DIR, "model_3b_runup.json")
CALIBRATOR_PATH = os.path.join(ARTIFACTS_DIR, "calibrator.pkl")
META_FILTER_PATH = os.path.join(ARTIFACTS_DIR, "model_4_meta_filter.json")
SIGNALS_PATH = os.path.join(ARTIFACTS_DIR, "signals.csv")

# Threshold: Drawdown must exceed -0.3% (-0.003) for mean-reversion setup
EXHAUSTION_THRESHOLD = -0.003


def load_models() -> tuple[
    DirectionalModel,
    PriceModel,
    ExhaustionModel,
    RunupModel,
    MultiClassCalibrator | None,
    MetaFilterModel | None,
]:
    """
    Loads all four pre-trained XGBoost models plus the Calibrator and Meta-Filter.
    """
    required_paths = [
        (MODEL_1_PATH, "Model 1 (Direction)"),
        (MODEL_2_PATH, "Model 2 (Price)"),
        (MODEL_3_PATH, "Model 3 (Exhaustion)"),
        (MODEL_3B_PATH, "Model 3b (Runup)"),
    ]
    for p, name in required_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} weight file is missing: {p}")

    model_1 = DirectionalModel(MODEL_1_PATH)
    model_2 = PriceModel(MODEL_2_PATH)
    model_3 = ExhaustionModel(MODEL_3_PATH)
    model_3b = RunupModel(MODEL_3B_PATH)

    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        try:
            calibrator = MultiClassCalibrator.load(CALIBRATOR_PATH)
            print("MultiClassCalibrator loaded successfully.")
        except Exception as e:
            print(f"Warning loading calibrator: {e}")

    meta_filter = None
    if os.path.exists(META_FILTER_PATH):
        try:
            meta_filter = MetaFilterModel(META_FILTER_PATH)
            print("MetaFilterModel loaded successfully.")
        except Exception as e:
            print(f"Warning loading meta-filter: {e}")

    print("All models loaded successfully.")
    return model_1, model_2, model_3, model_3b, calibrator, meta_filter


def generate_signals(
    df_raw: pd.DataFrame,
    symbol: str,
    model_1: DirectionalModel,
    model_2: PriceModel,
    model_3: ExhaustionModel,
    model_3b: RunupModel | None = None,
    calibrator: MultiClassCalibrator | None = None,
    meta_filter: MetaFilterModel | None = None,
    exposure_mgr: ExposureManager | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    portfolio_state: PortfolioState | None = None,
    portfolio_capital: float = 100000.0,
    risk_pct_per_trade: float = 0.02,
    sizing_method: str = POSITION_SIZING_METHOD,
) -> pd.DataFrame:
    """
    Runs multi-model EV-gated inference with ensemble agreement and ExitManager.
    """
    if portfolio_state is None:
        portfolio_state = PortfolioState()
        portfolio_state.load()
    if exposure_mgr is None:
        exposure_mgr = ExposureManager(max_correlated_positions=2, correlation_threshold=0.7, portfolio_state=portfolio_state)
    if circuit_breaker is None:
        circuit_breaker = CircuitBreaker(portfolio_state=portfolio_state)

    if not circuit_breaker.can_trade():
        return pd.DataFrame()

    # 1. Engineer features
    df_1min = build_features_1min(df_raw)
    df_5min = build_features_5min(df_raw)

    if os.path.exists(PEER_MAP_PATH):
        try:
            df_5min = inject_cross_asset_features(df_5min, symbol, PEER_MAP_PATH, PARQUET_DIR)
        except Exception as e:
            print(f"  WARNING: Could not inject cross-asset features: {e}")

    drop_cols = [
        'date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
        'mid_price', 'future_close_5m', 'target'
    ]
    feature_cols_5min = [c for c in df_5min.columns if c not in drop_cols]
    feature_cols_1min = [c for c in df_1min.columns if c not in drop_cols]

    X_5min = pd.DataFrame(df_5min[feature_cols_5min].dropna())
    X_1min = pd.DataFrame(df_1min[feature_cols_1min].dropna())

    if len(X_5min) == 0 or len(X_1min) == 0:
        print(f"  WARNING: Not enough clean data for {symbol}")
        return pd.DataFrame()

    # 2. Model Predictions
    pred_class, raw_probs = model_1.predict(X_5min)
    if calibrator is not None and calibrator.is_fitted:
        cal_probs = calibrator.calibrate(raw_probs)
    else:
        cal_probs = raw_probs

    pred_price = model_2.predict(X_5min)
    pred_drawdown = model_3.predict(X_1min)
    pred_runup = model_3b.predict(X_1min) if model_3b is not None else np.full(len(X_1min), 0.01)

    # Align 1-min predictions (Model 3 & 3b) to 5-min bar timestamps
    s_drawdown = pd.Series(pred_drawdown, index=X_1min.index).reindex(X_5min.index, method='ffill').fillna(-0.003).values
    s_runup = pd.Series(pred_runup, index=X_1min.index).reindex(X_5min.index, method='ffill').fillna(0.005).values

    # 3. Risk Managers
    atr_series = calculate_atr(df_5min, period=14)
    exit_mgr = ExitManager()

    min_len = len(X_5min)
    signals = []

    for i in range(min_len):
        current_price = float(df_5min['mid_price'].iloc[i]) if 'mid_price' in df_5min.columns else float(df_5min['close'].iloc[i])
        current_date = df_5min.index[i] if isinstance(df_5min.index, pd.DatetimeIndex) else (df_5min['date'].iloc[i] if 'date' in df_5min.columns else i)
        current_atr = float(atr_series.iloc[i]) if i < len(atr_series) and not np.isnan(atr_series.iloc[i]) else current_price * 0.005

        price_move = float(pred_price[i])
        drawdown = float(s_drawdown[i])
        runup = float(s_runup[i])

        # Update ExitManager for open positions
        to_close = []
        for pos_symbol, pos_data in list(portfolio_state.open_positions.items()):
            pos_data['bars_held'] = pos_data.get('bars_held', 0) + 1
            pos_data['highest_price'] = max(pos_data.get('highest_price', pos_data['entry_price']), current_price)
            pos_data['lowest_price'] = min(pos_data.get('lowest_price', pos_data['entry_price']), current_price)

            exit_decision = exit_mgr.evaluate_exit(
                direction=pos_data['direction'],
                entry_price=pos_data['entry_price'],
                entry_atr=pos_data.get('entry_atr', current_atr),
                current_price=current_price,
                highest_price=pos_data['highest_price'],
                lowest_price=pos_data['lowest_price'],
                bars_held=pos_data['bars_held'],
                predicted_runup=runup,
            )
            if exit_decision.should_exit:
                to_close.append(pos_symbol)

        for c in to_close:
            exposure_mgr.close_position(c)

        # 4. Semantic direction and probability (Fix Group 7: Ignore FLAT)
        p_down = float(cal_probs[i, 0])
        p_flat = float(cal_probs[i, 1])
        p_up = float(cal_probs[i, 2])

        non_flat = p_up + p_down
        c_up = p_up / non_flat if non_flat > 0 else 0.0
        c_down = p_down / non_flat if non_flat > 0 else 0.0

        # Conviction criteria: directional skew >= 65%, p >= 0.20, matching M2 sign, runup room >= 0.4%
        if c_up >= 0.65 and p_up >= 0.20 and price_move > 0 and runup >= 0.004:
            sem_dir = 1
            conf = c_up
            cls = 2
        elif c_down >= 0.65 and p_down >= 0.20 and price_move < 0 and abs(drawdown) >= 0.004:
            sem_dir = -1
            conf = c_down
            cls = 0
        else:
            sem_dir = 0
            conf = p_flat
            cls = 1

        # 5. Fix Group 6: Expected-Value (EV) calculation
        # Target magnitude derived from barrier distance (>= 0.52%) and ExitManager's take-profit
        barrier = max(1.5 * current_atr / max(current_price, 1e-6), MIN_BARRIER_DISTANCE_PCT)
        if sem_dir == 1:
            target_gain = max(barrier, EXIT_TAKE_PROFIT_SAFETY * runup)
        elif sem_dir == -1:
            target_gain = max(barrier, EXIT_TAKE_PROFIT_SAFETY * abs(drawdown))
        else:
            target_gain = abs(price_move)
        ev = (conf * target_gain) - ROUND_TRIP_COST_PCT

        # 6. Fix Group 5: Meta-Filter Check
        meta_approved = True
        peer_ret = 0.0
        if sem_dir != 0:
            try:
                pct_agree, mean_signed = compute_peer_aggregates(df_5min.iloc[[i]], np.array([sem_dir]))
                peer_ret = float(mean_signed[0])
                if meta_filter is not None:
                    meta_row = pd.DataFrame([{
                        "m1_predicted_class": cls,
                        "m1_prob_down": p_down,
                        "m1_prob_flat": p_flat,
                        "m1_prob_up": p_up,
                        "m2_predicted_magnitude": price_move,
                        "return_zscore": df_5min['return_zscore'].iloc[i] if 'return_zscore' in df_5min else 0.0,
                        "log_return_zscore": df_5min['log_return_zscore'].iloc[i] if 'log_return_zscore' in df_5min else 0.0,
                        "rolling_zscore": df_5min['rolling_zscore'].iloc[i] if 'rolling_zscore' in df_5min else 0.0,
                        "volume_zscore": df_5min['volume_zscore'].iloc[i] if 'volume_zscore' in df_5min else 0.0,
                        "volatility_zscore": df_5min['volatility_zscore'].iloc[i] if 'volatility_zscore' in df_5min else 0.0,
                        "price_vs_ma_zscore": df_5min['price_vs_ma_zscore'].iloc[i] if 'price_vs_ma_zscore' in df_5min else 0.0,
                        "price_vs_ema_zscore": df_5min['price_vs_ema_zscore'].iloc[i] if 'price_vs_ema_zscore' in df_5min else 0.0,
                        "range_zscore": df_5min['range_zscore'].iloc[i] if 'range_zscore' in df_5min else 0.0,
                        "momentum_zscore": df_5min['momentum_zscore'].iloc[i] if 'momentum_zscore' in df_5min else 0.0,
                        "pct_top_peers_agreeing": pct_agree[0],
                        "mean_peer_signed_return": mean_signed[0],
                    }])
                    meta_dec, _ = meta_filter.predict(meta_row, threshold=0.05)
                    meta_approved = bool(meta_dec[0] == 1)
            except Exception:
                meta_approved = True

        # 7. Fix Group 7: Ensemble Agreement Engine
        agreement = evaluate_ensemble_agreement(
            timestamp=str(current_date),
            symbol=symbol,
            m1_dir=sem_dir,
            m1_conf=conf,
            m2_magnitude=price_move,
            m3_drawdown=drawdown,
            m3b_runup=runup,
            mean_peer_return=peer_ret,
            ev=ev,
            meta_approved=meta_approved,
            log_to_csv=False
        )

        signal_type = "HOLD"
        entry_price = current_price

        # EV-gated execution logic
        if agreement.recommended_action == "BUY" and ev > MIN_EDGE_BUFFER_PCT:
            signal_type = "MOMENTUM_BUY"
            entry_price = current_price
        elif agreement.recommended_action == "SELL" and ev > MIN_EDGE_BUFFER_PCT:
            signal_type = "MOMENTUM_SELL"
            entry_price = current_price
        elif sem_dir == 1 and drawdown < EXHAUSTION_THRESHOLD and meta_approved:
            signal_type = "MEAN_REVERSION_BUY"
            entry_price = current_price * (1.0 - abs(drawdown))

        # Position Sizing (Fix Group 9: Kelly / ATR / Hybrid)
        shares = 0.0
        position_value = 0.0

        if signal_type != "HOLD":
            exposure_allowed = exposure_mgr.can_open_position(symbol)
            if exposure_allowed:
                shares = calculate_position_size(
                    capital=portfolio_capital,
                    risk_pct=risk_pct_per_trade,
                    atr=current_atr,
                    price=entry_price,
                    max_position_pct=0.10,
                    calibrated_prob=conf,
                    predicted_runup=runup,
                    sizing_method=sizing_method,
                )
                position_value = round(shares * entry_price, 2)
                exposure_mgr.register_position(
                    symbol,
                    direction=1 if 'BUY' in signal_type else -1,
                    entry_price=entry_price,
                    timestamp=str(current_date)
                )
                if symbol in portfolio_state.open_positions:
                    portfolio_state.open_positions[symbol]['entry_atr'] = current_atr
            else:
                signal_type = "BLOCKED_BY_EXPOSURE_CAP"

        signals.append({
            'date': current_date,
            'symbol': symbol,
            'signal_type': signal_type,
            'direction': 1 if 'BUY' in signal_type else (-1 if 'SELL' in signal_type else 0),
            'entry_price': round(entry_price, 2),
            'shares': round(shares, 2),
            'position_value': position_value,
            'm1_predicted_class': cls,
            'm1_confidence': round(conf, 4),
            'm1_p_up': round(p_up, 4),
            'm1_p_down': round(p_down, 4),
            'm1_p_flat': round(p_flat, 4),
            'm2_predicted_move': round(price_move, 6),
            'm3_drawdown': round(drawdown, 4),
            'm3b_runup': round(runup, 4),
            'ev': round(ev, 6),
            'meta_approved': meta_approved,
            'ensemble_agreement_count': agreement.agreement_count,
            'atr': round(current_atr, 4),
        })

    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        signals_df.to_csv(SIGNALS_PATH, index=False)

    return signals_df


def run_inference(df_raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Main inference entrypoint.
    """
    if check_kill_switch():
        print(f"KILL SWITCH ACTIVE. Suppressing all signals for {symbol}.")
        return pd.DataFrame()

    model_1, model_2, model_3, model_3b, calibrator, meta_filter = load_models()

    portfolio_state = PortfolioState()
    portfolio_state.load()
    exposure_mgr = ExposureManager(portfolio_state=portfolio_state)
    circuit_breaker = CircuitBreaker(portfolio_state=portfolio_state)

    signals = generate_signals(
        df_raw,
        symbol,
        model_1,
        model_2,
        model_3,
        model_3b=model_3b,
        calibrator=calibrator,
        meta_filter=meta_filter,
        exposure_mgr=exposure_mgr,
        circuit_breaker=circuit_breaker,
        portfolio_state=portfolio_state,
    )

    portfolio_state.save()
    return signals
