"""
========================================================================================
FEATURE ENGINE: Pipeline Feature Orchestrator
========================================================================================
Builds feature-enriched DataFrames for 1-minute (Model 3) and 5-minute (Models 1 & 2) data.
Integrates Z-scores, technical indicators (RSI, VWAP distance), and session grouping.
All calculations use shift(1) to strictly eliminate data leakage.
========================================================================================
"""

import numpy as np
import pandas as pd

from .aggregator import Aggregator
from .indicators import TechnicalIndicators
from .reference_price import compute_reference_price
from .zscore import ZScoreCalculator


def build_features_1min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds features for the 1-minute dataset (used for Model 3).
    Builds features for the 1-minute dataset (used primarily for Model 3 Exhaustion).
    
    Steps:
    1. Computes mid_price.
    2. Applies all z-scores (return, log_return, volume, volatility, price_vs_ma).
    3. Adds exhaustion features (RSI, VWAP distance, Volume Z).
    4. Drops NaN rows.
    1. Computes mid_price = (open + close) / 2.
    2. Computes core 1-min Z-scores: return, log-return, volume, volatility, price-vs-MA,
       dollar volume, and range.
    3. Adds exhaustion features: 20-min Volume Z, VWAP Distance, and 14-min RSI.
    4. Drops initial NaN rows resulting from rolling lookback windows.
    
    Args:
        df (pd.DataFrame): Raw 1-min OHLCV DataFrame.
        df: Raw 1-min OHLCV DataFrame with 'date' or DatetimeIndex.
        
    Returns:
        pd.DataFrame: Feature-enriched DataFrame.
        Feature-enriched 1-minute DataFrame.
    """
    # Create copies to avoid SettingWithCopyWarning
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    
    # 1. Mid-Price: Typical Price = (H+L+C)/3 — single source of truth
    # NOTE: No real bid/ask data exists. See feature_engine/reference_price.py.
    df['mid_price'] = compute_reference_price(df)
    
    # Initialize calculators
    zscore_calc = ZScoreCalculator()
    indicators = TechnicalIndicators()
    
    # 2. Apply all z-scores
    # Return Z-Score: Captures short-term momentum and mean reversion
    # 2. Compute 1-minute features
    # Return & Momentum Z-scores
    df['return_zscore'] = zscore_calc.return_zscore(df['mid_price'], window=20)
    
    # Log Return Z-Score: Normalizes extreme price movements, useful for volatility modeling
    df['log_return_zscore'] = zscore_calc.log_return_zscore(df['mid_price'], window=20)
    df['rolling_zscore'] = zscore_calc.rolling_zscore(df['mid_price'], window=20)
    
    # Volume Z-Score: Highlights abnormal trading activity or liquidity bursts
    # Volume & Dollar Volume Z-scores
    df['volume_zscore'] = zscore_calc.volume_zscore(df['volume'], window=20)
    df['dollar_volume_zscore'] = zscore_calc.dollar_volume_zscore(df['mid_price'], df['volume'], window=20)
    
    # Volatility Z-Score: Identifies periods of market stress or unusual calm
    # Volatility & Range Z-scores (volatility computed strictly from returns)
    returns = df['mid_price'].pct_change()
    df['volatility_zscore'] = zscore_calc.volatility_zscore(returns, window=20)
    df['range_zscore'] = zscore_calc.range_zscore(df['high'], df['low'], window=20)
    
    # Price vs MA Z-Score: Indicates overbought or oversold conditions (mean reversion)
    # Price vs. Mean & Trend Z-scores
    df['price_vs_ma_zscore'] = zscore_calc.price_vs_ma_zscore(df['mid_price'], window=20)
    df['price_vs_ema_zscore'] = zscore_calc.price_vs_ema_zscore(df['mid_price'], span=20)
    
    # 3. Add exhaustion features
    # Exhaustion features combine momentum (RSI), positioning (VWAP distance), and activity (Volume Z)
    # to identify potential trend reversals.
    # 3. Exhaustion features (Volume Z, VWAP Distance, RSI 14)
    exhaustion_df = indicators.exhaustion_features(df, zscore_calc)
    
    # Join exhaustion features
    df = df.join(exhaustion_df)
    
    # 4. Drop NaN rows resulting from rolling windows and shifts
    df = df.dropna()
    
    # Clean inf and NaNs from rolling window startup
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def build_features_5min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds features for the 5-minute dataset (used for Models 1 & 2).
    
    Steps:
    1. Resamples 1-min raw data to 5-min intervals.
    2. Applies all z-scores on the 5-min data.
    3. Drops NaN rows.
    1. Resamples 1-min raw OHLCV to 5-min intervals with true OHLC aggregation.
    2. Computes 5-min mid_price = (open + close) / 2.
    3. Applies 5-min Z-scores: return, log-return, volume, volatility, price-vs-MA,
       momentum, and range.
    4. Drops initial NaN rows.
    
    Args:
        df (pd.DataFrame): Raw 1-min OHLCV DataFrame.
        df: Raw 1-min OHLCV DataFrame.
        
    Returns:
        pd.DataFrame: Feature-enriched 5-minute DataFrame.
        Feature-enriched 5-minute DataFrame.
    """
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    # Initialize aggregator and z-score calculator
        
    aggregator = Aggregator()
    zscore_calc = ZScoreCalculator()
    
    # 1. Resample to 5-min
    # Aggregation smooths out market microstructure noise
    # 1. Resample to 5-min intervals (first open, max high, min low, last close, sum volume)
    df_5m = aggregator.resample_to_5min(df)
    
    # 2. Apply all z-scores
    # Return Z-Score: Captures 5-min momentum
    # 2. Compute 5-minute features
    df_5m['return_zscore'] = zscore_calc.return_zscore(df_5m['mid_price'], window=20)
    
    # Log Return Z-Score: Captures continuous compounding returns over 5-min
    df_5m['log_return_zscore'] = zscore_calc.log_return_zscore(df_5m['mid_price'], window=20)
    
    # Volume Z-Score: Captures aggregated trading volume anomalies
    df_5m['rolling_zscore'] = zscore_calc.rolling_zscore(df_5m['mid_price'], window=20)
    df_5m['volume_zscore'] = zscore_calc.volume_zscore(df_5m['volume'], window=20)
    
    # Volatility Z-Score: Captures 5-min volatility shifts
    returns_5m = df_5m['mid_price'].pct_change()
    df_5m['volatility_zscore'] = zscore_calc.volatility_zscore(returns_5m, window=20)
    
    # Price vs MA Z-Score: Captures broader trend deviations
    df_5m['price_vs_ma_zscore'] = zscore_calc.price_vs_ma_zscore(df_5m['mid_price'], window=20)
    df_5m['price_vs_ema_zscore'] = zscore_calc.price_vs_ema_zscore(df_5m['mid_price'], span=20)
    df_5m['range_zscore'] = zscore_calc.range_zscore(df_5m['high'], df_5m['low'], window=20)
    df_5m['momentum_zscore'] = zscore_calc.momentum_zscore(df_5m['mid_price'], lag=3, window=20)
    
    # 3. Drop NaN rows
    df_5m = df_5m.dropna()
    
    # Clean inf and NaNs
    df_5m = df_5m.replace([np.inf, -np.inf], np.nan).dropna()
    return df_5m
