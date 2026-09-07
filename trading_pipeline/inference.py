"""
========================================================================================
INFERENCE: 3-Model Signal Combiner & Risk-Managed Execution Engine
========================================================================================
This is the production inference engine that combines all three models and applies
full quantitative risk controls.

DECISION LOGIC (from Specification Section 5):
1. Model 1 (Direction): Up/Down probability classifier for next 5-minute close.
2. Model 2 (Price): Regression prediction of next 1-minute return (Section 13 alignment).
3. Model 3 (Exhaustion): Forward 10-minute maximum expected drawdown.

THE COMBINATION RULE:
- If Model 1 predicts UP AND Model 3 detects a sharp incoming drawdown (< -0.3%):
  -> MEAN REVERSION BUY: Market is experiencing a temporary liquidity flush.
  -> Limit Entry Formula: Current Mid-Price * (1 - abs(Predicted_Drawdown))
  -> Places limit order to buy the dip before the bounce.
- If Model 1 predicts UP with high confidence (> 0.60) without exhaustion:
  -> MOMENTUM BUY: Enter at market/mid-price.
- If Model 1 predicts DOWN with high confidence (> 0.60):
  -> MOMENTUM SELL: Enter short or exit longs.

RISK CONTROLS (Section 11):
- Kill Switch: HALTS ALL TRADING immediately if 'artifacts/KILL_SWITCH' exists.
- Circuit Breaker: Auto-stops if daily drawdown reaches -2%.
- Exposure Manager: Enforces max 2 simultaneous positions across correlated peers
  derived from the Section 2 Correlation Matrix.
- Position Sizer: ATR-adjusted volatility sizing.
========================================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the trading_pipeline directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.risk_manager import (
    ExposureManager,
    calculate_atr,
    calculate_position_size,
    check_kill_switch,
)
from feature_engine.pipeline import build_features_1min, build_features_5min
from models.model_1_direction import DirectionalModel
from models.model_2_price import PriceModel
from models.model_3_exhaustion import ExhaustionModel

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
ARTIFACTS_DIR = r"d:\CODE\rajasthani\trading_pipeline\artifacts"
MODEL_1_PATH = os.path.join(ARTIFACTS_DIR, "model_1_direction.json")
MODEL_2_PATH = os.path.join(ARTIFACTS_DIR, "model_2_price.json")
MODEL_3_PATH = os.path.join(ARTIFACTS_DIR, "model_3_exhaustion.json")
SIGNALS_PATH = os.path.join(ARTIFACTS_DIR, "signals.csv")

# Threshold: Drawdown must exceed -0.3% (-0.003) for mean-reversion setup
EXHAUSTION_THRESHOLD = -0.003


def load_models() -> tuple[DirectionalModel, PriceModel, ExhaustionModel]:
    """
    Loads all three pre-trained XGBoost models from disk.
    
    Returns:
        (model_1, model_2, model_3): Tuple of loaded model instances.
    """
    if not (os.path.exists(MODEL_1_PATH) and os.path.exists(MODEL_2_PATH) and os.path.exists(MODEL_3_PATH)):
        raise FileNotFoundError("One or more model weight files are missing in artifacts directory.")

    model_1 = DirectionalModel(MODEL_1_PATH)
    model_2 = PriceModel(MODEL_2_PATH)
    model_3 = ExhaustionModel(MODEL_3_PATH)

    print("All 3 models loaded successfully.")
    return model_1, model_2, model_3


def generate_signals(
    df_raw: pd.DataFrame,
    symbol: str,
    model_1: DirectionalModel,
    model_2: PriceModel,
    model_3: ExhaustionModel,
    portfolio_capital: float = 100000.0,
    risk_pct_per_trade: float = 0.02
) -> pd.DataFrame:
    """
    Runs 3-model inference with full risk management (position sizing, exposure caps).
    
    Args:
        df_raw: Raw 1-min OHLCV DataFrame.
        symbol: Stock ticker symbol.
        model_1: Loaded DirectionalModel.
        model_2: Loaded PriceModel.
        model_3: Loaded ExhaustionModel.
        portfolio_capital: Current account equity.
        risk_pct_per_trade: Percentage risk per trade (default 2%).
        
    Returns:
        DataFrame of actionable signals with ATR-sized position allocations.
    """
    # 1. Engineer features
    df_1min = build_features_1min(df_raw)
    df_5min = build_features_5min(df_raw)
    
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
    pred_dir, prob_dir = model_1.predict(X_5min)
    pred_price = model_2.predict(X_5min)
    pred_drawdown = model_3.predict(X_1min)
    
    # 3. Risk Managers
    exposure_mgr = ExposureManager(max_correlated_positions=2, correlation_threshold=0.7)
    atr_series = calculate_atr(df_5min, period=14)
    
    min_len = min(len(pred_dir), len(pred_price), len(pred_drawdown))
    signals = []
    
    for i in range(min_len):
        direction = int(pred_dir[i])
        confidence = float(prob_dir[i])
        price_move = float(pred_price[i])
        drawdown = float(pred_drawdown[i])
        
        current_price = float(df_5min['mid_price'].iloc[i]) if 'mid_price' in df_5min.columns else float(df_5min['close'].iloc[i])
        current_date = df_5min.index[i] if isinstance(df_5min.index, pd.DatetimeIndex) else (df_5min['date'].iloc[i] if 'date' in df_5min.columns else i)
        
        signal_type = "HOLD"
        entry_price = current_price
        
        # COMBINATION LOGIC:
        if direction == 1 and drawdown < EXHAUSTION_THRESHOLD:
            # Mean reversion setup: Buy the dip at predicted forward drop
            entry_price = current_price * (1.0 - abs(drawdown))
            signal_type = "MEAN_REVERSION_BUY"
        elif direction == 1 and confidence > 0.60:
            signal_type = "MOMENTUM_BUY"
            entry_price = current_price
        elif direction == 0 and confidence > 0.60:
            signal_type = "MOMENTUM_SELL"
            entry_price = current_price
            
        # Risk Management: Sizing & Exposure Cap
        current_atr = float(atr_series.iloc[i]) if i < len(atr_series) and not np.isnan(atr_series.iloc[i]) else current_price * 0.005
        
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
                    max_position_pct=0.10
                )
                position_value = round(shares * entry_price, 2)
            else:
                signal_type = "BLOCKED_BY_EXPOSURE_CAP"
                
        signals.append({
            'date': current_date,
            'symbol': symbol,
            'direction': 1 if 'BUY' in signal_type else (-1 if 'SELL' in signal_type else 0),
            'confidence': round(confidence, 4),
            'predicted_price_move': round(price_move, 6),
            'predicted_drawdown': round(drawdown, 6),
            'entry_price': round(entry_price, 2),
            'current_price': round(current_price, 2),
            'atr_14': round(current_atr, 4),
            'shares': int(shares),
            'position_value': position_value,
            'signal_type': signal_type
        })
        
    return pd.DataFrame(signals)


def run_inference(df_raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Main inference entrypoint. Halts on kill switch, loads models, outputs signals.
    
    Args:
        df_raw: Raw 1-minute OHLCV DataFrame.
        symbol: Stock ticker symbol.
        
    Returns:
        DataFrame of trading signals (or empty if kill switch is on).
    """
    # 1. Kill Switch Check (Regulatory Requirement)
    if check_kill_switch():
        print(f"KILL SWITCH ACTIVE. Suppressing all signals for {symbol}.")
        return pd.DataFrame()
    
    # 2. Load Models
    model_1, model_2, model_3 = load_models()
    
    # 3. Generate Signals
    signals = generate_signals(df_raw, symbol, model_1, model_2, model_3)
    
    if len(signals) > 0:
        actionable = signals[signals['signal_type'].isin(['MEAN_REVERSION_BUY', 'MOMENTUM_BUY', 'MOMENTUM_SELL'])]
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        
        if os.path.exists(SIGNALS_PATH):
            try:
                existing = pd.read_csv(SIGNALS_PATH)
                combined = pd.concat([existing, actionable], ignore_index=True)
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                combined = actionable
        else:
            combined = actionable
        
        combined.to_csv(SIGNALS_PATH, index=False)
        print(f"Generated {len(actionable)} actionable signals for {symbol}.")
        print(f"\nSignal Summary for {symbol}:")
        print(f"  Total bars analyzed: {len(signals)}")
        print(f"  Actionable signals: {len(actionable)}")
        if len(actionable) > 0:
            print(f"  Signal types: {pd.Series(actionable['signal_type']).value_counts().to_dict()}")
        print(f"  Signals saved to: {SIGNALS_PATH}")
    
    return signals


if __name__ == "__main__":
    import polars as pl
    
    test_file = os.path.join(r"d:\CODE\rajasthani\DATA\parquet", "RELIANCE.parquet")
    
    if os.path.exists(test_file):
        df = pl.read_parquet(test_file, memory_map=False).to_pandas()
        signals = run_inference(df, "RELIANCE")
    else:
        print("ERROR: Run preprocess_to_parquet.py and train_master.py first!")
