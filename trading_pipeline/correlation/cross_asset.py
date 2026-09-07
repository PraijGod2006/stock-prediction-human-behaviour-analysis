"""
========================================================================================
CORRELATION: Cross-Asset Peer Momentum Feature Injector
========================================================================================
Injects correlated peer momentum features into the target symbol's dataframe.

WHY THIS EXISTS:
Cross-asset lead-lag effects are one of the strongest statistical edges in quantitative
trading. Sector peers (e.g., HDFCBANK & ICICIBANK) share macro drivers, but one often
leads the other by 1 to 5 minutes.

CRITICAL DESIGN (Normalized Peer Roles):
To allow XGBoost incremental learning (xgb_model=) across all 100 companies, feature
column names MUST remain identical across companies. Therefore, we name features by
their rank-ordered peer relationship:
- peer_pos_1_return_lag1 (highest positively correlated peer)
- peer_pos_2_return_lag1 (2nd highest positively correlated peer)
- peer_pos_3_return_lag1 (3rd highest positively correlated peer)
- peer_neg_1_return_lag1 (highest inversely correlated peer)
- peer_neg_2_return_lag1 (2nd highest inversely correlated peer)
- peer_neg_3_return_lag1 (3rd highest inversely correlated peer)

LEAKAGE PREVENTION:
All peer returns are lagged by shift(1), so feature at T uses ONLY data up to T-1.
========================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import polars as pl
from typing import Union


def inject_cross_asset_features(
    df: Union[pd.DataFrame, pl.DataFrame],
    symbol: str,
    peer_map_path: str = r"d:\CODE\rajasthani\trading_pipeline\artifacts\peer_map.json",
    parquet_dir: str = r"d:\CODE\rajasthani\DATA\parquet"
) -> pd.DataFrame:
    """
    Injects rank-normalized cross-asset peer momentum features.
    
    Args:
        df: Target symbol DataFrame (Pandas or Polars) with DatetimeIndex or 'date'.
        symbol: Target ticker symbol.
        peer_map_path: Path to peer_map.json.
        parquet_dir: Directory containing parquet files.
        
    Returns:
        Pandas DataFrame enriched with peer_pos_1..3 and peer_neg_1..3 features.
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

    # Default placeholder peer columns with zeros to guarantee consistent schema
    default_peer_cols = [
        "peer_pos_1_return_lag1", "peer_pos_2_return_lag1", "peer_pos_3_return_lag1",
        "peer_neg_1_return_lag1", "peer_neg_2_return_lag1", "peer_neg_3_return_lag1"
    ]

    if not os.path.exists(peer_map_path):
        for col in default_peer_cols:
            df[col] = 0.0
        return df

    with open(peer_map_path, "r") as f:
        peer_map = json.load(f)

    if symbol not in peer_map:
        for col in default_peer_cols:
            df[col] = 0.0
        return df

    peers_pos = peer_map[symbol].get("top_positive", [])[:3]
    peers_neg = peer_map[symbol].get("top_negative", [])[:3]

    # Process positive peers
    for rank, peer in enumerate(peers_pos):
        col_name = f"peer_pos_{rank+1}_return_lag1"
        peer_file = os.path.join(parquet_dir, f"{peer}.parquet")
        if not os.path.exists(peer_file):
            df[col_name] = 0.0
            continue
        try:
            peer_raw = pl.read_parquet(peer_file, columns=["date", "open", "close"], memory_map=False).to_pandas()
            if 'date' in peer_raw.columns:
                peer_raw['date'] = pd.to_datetime(peer_raw['date'])
                peer_raw = peer_raw.set_index('date')
            peer_mid = (peer_raw['open'] + peer_raw['close']) / 2.0
            # 5-min lagged return, shifted by 1 to prevent lookahead bias
            peer_ret = (peer_mid / peer_mid.shift(5) - 1.0).shift(1).rename(col_name)
            df = df.join(peer_ret, how='left')
            del peer_raw, peer_mid, peer_ret
        except Exception:
            df[col_name] = 0.0

    # Process negative peers
    for rank, peer in enumerate(peers_neg):
        col_name = f"peer_neg_{rank+1}_return_lag1"
        peer_file = os.path.join(parquet_dir, f"{peer}.parquet")
        if not os.path.exists(peer_file):
            df[col_name] = 0.0
            continue
        try:
            peer_raw = pl.read_parquet(peer_file, columns=["date", "open", "close"], memory_map=False).to_pandas()
            if 'date' in peer_raw.columns:
                peer_raw['date'] = pd.to_datetime(peer_raw['date'])
                peer_raw = peer_raw.set_index('date')
            peer_mid = (peer_raw['open'] + peer_raw['close']) / 2.0
            peer_ret = (peer_mid / peer_mid.shift(5) - 1.0).shift(1).rename(col_name)
            df = df.join(peer_ret, how='left')
            del peer_raw, peer_mid, peer_ret
        except Exception:
            df[col_name] = 0.0

    # Ensure all 6 standard peer columns exist and nulls are filled with 0.0
    for col in default_peer_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    return df
