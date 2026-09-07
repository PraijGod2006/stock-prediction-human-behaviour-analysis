"""
cross_asset.py
========================================================================================
CORRELATION: Cross-Asset Peer Momentum Feature Injector
========================================================================================
Injects correlated peer momentum features into the target symbol's dataframe.

WHY THIS EXISTS:
Cross-asset lead-lag effects are one of the strongest statistical edges in quantitative
trading. Sector peers (e.g., HDFCBANK & ICICIBANK) share macro drivers, but one often
leads the other by 1 to 5 minutes. By injecting lagged peer returns as features,
XGBoost can learn these lead-lag dynamics.

BUG FIX HISTORY:
- Original version crashed with "Joining multiple DataFrames only supported
  for joining on index" because it used Polars joins on a Pandas DataFrame.
  Fixed to accept both Pandas and Polars, and use iterative pd.DataFrame.join
  on the datetime index.

CRITICAL DESIGN (Normalized Peer Roles):
To allow XGBoost incremental learning (xgb_model=) across all 100 companies, feature
column names MUST remain identical across companies. Therefore, we name features by
their rank-ordered peer relationship, NOT by the peer's actual ticker symbol:
- peer_pos_1_return_lag1 (highest positively correlated peer)
- peer_pos_2_return_lag1 (2nd highest positively correlated peer)
- peer_pos_3_return_lag1 (3rd highest positively correlated peer)
- peer_neg_1_return_lag1 (highest inversely correlated peer)
- peer_neg_2_return_lag1 (2nd highest inversely correlated peer)
- peer_neg_3_return_lag1 (3rd highest inversely correlated peer)

LEAKAGE PREVENTION:
Peer return at time T = (peer_mid_{T-1} - peer_mid_{T-6}) / peer_mid_{T-6}
i.e. a 5-bar return that is additionally shift(1)'d, so the feature at T uses
ONLY peer data up to T-1.
========================================================================================
"""

import os
import json
import pandas as pd
import polars as pl
from typing import Union, Optional

# Canonical schema: every symbol gets exactly these 6 columns, in this order,
# regardless of how many peers were actually available for it.
DEFAULT_PEER_COLS = [
    "peer_pos_1_return_lag1", "peer_pos_2_return_lag1", "peer_pos_3_return_lag1",
    "peer_neg_1_return_lag1", "peer_neg_2_return_lag1", "peer_neg_3_return_lag1",
]


def _load_peer_return(peer: str, parquet_dir: str) -> Optional[pd.Series]:
    """
    Loads a single peer's parquet file and computes its shift(1)-lagged
    5-bar mid-price return, indexed by datetime.

    Returns None if the peer's file doesn't exist or anything goes wrong
    while loading/parsing it, so a single bad peer can't kill the pipeline.
    """
    peer_file = os.path.join(parquet_dir, f"{peer}.parquet")
    if not os.path.exists(peer_file):
        return None

    try:
        peer_raw = pl.read_parquet(
            peer_file,
            columns=["date", "open", "close"],
            memory_map=False,
        ).to_pandas()

        if 'date' in peer_raw.columns:
            peer_raw['date'] = pd.to_datetime(peer_raw['date'])
            peer_raw = peer_raw.set_index('date')

        peer_mid = (peer_raw['open'] + peer_raw['close']) / 2.0

        # 5-bar return, additionally shifted by 1 to prevent lookahead leakage:
        # feature at T = return realized from T-6 to T-1.
        peer_return = (peer_mid / peer_mid.shift(5) - 1.0).shift(1)
        return peer_return
    except Exception as e:
        print(f"    WARNING: Could not load peer {peer}: {e}")
        return None


def inject_cross_asset_features(
    df: Union[pd.DataFrame, pl.DataFrame],
    symbol: str,
    peer_map_path: str = r"d:\CODE\rajasthani\trading_pipeline\artifacts\peer_map.json",
    parquet_dir: str = r"d:\CODE\rajasthani\DATA\parquet",
) -> pd.DataFrame:
    """
    Injects rank-normalized cross-asset peer momentum features based on the
    pre-computed correlation matrix (peer_map.json).

    For each of the top-3 positively and top-3 inversely correlated peers,
    computes a shift(1)-lagged 5-bar mid-price return and merges it onto the
    target symbol's DataFrame by timestamp, under a rank-normalized column
    name (see CRITICAL DESIGN above) so the schema is identical for every
    symbol in the universe.

    Args:
        df: Target symbol's feature DataFrame (Pandas or Polars).
            Must have a DatetimeIndex, or a 'date' column that can be set as one.
        symbol: Target ticker symbol (e.g., "RELIANCE").
        peer_map_path: Path to the pre-computed peer_map.json.
        parquet_dir: Directory containing per-company Parquet files.

    Returns:
        Pandas DataFrame enriched with the 6 peer_pos_*/peer_neg_* columns.
    """
    # --- Convert to Pandas if Polars ---
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # --- Ensure datetime index ---
    if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

    # --- No peer map at all: fill placeholder zero columns and bail out ---
    if not os.path.exists(peer_map_path):
        for col in DEFAULT_PEER_COLS:
            df[col] = 0.0
        return df

    with open(peer_map_path, "r") as f:
        peer_map = json.load(f)

    if symbol not in peer_map:
        for col in DEFAULT_PEER_COLS:
            df[col] = 0.0
        return df

    peers_pos = peer_map[symbol].get("top_positive", [])[:3]
    peers_neg = peer_map[symbol].get("top_negative", [])[:3]

    # --- Positive peers: peer_pos_1..3_return_lag1 ---
    for rank, peer in enumerate(peers_pos):
        col_name = f"peer_pos_{rank + 1}_return_lag1"
        peer_return = _load_peer_return(peer, parquet_dir)
        if peer_return is None:
            df[col_name] = 0.0
        else:
            df = df.join(peer_return.rename(col_name), how='left')

    # --- Negative (inversely correlated) peers: peer_neg_1..3_return_lag1 ---
    for rank, peer in enumerate(peers_neg):
        col_name = f"peer_neg_{rank + 1}_return_lag1"
        peer_return = _load_peer_return(peer, parquet_dir)
        if peer_return is None:
            df[col_name] = 0.0
        else:
            df = df.join(peer_return.rename(col_name), how='left')

    # --- Guarantee all 6 standard columns exist, fill NaNs with 0.0 ---
    # (Missing timestamps or fewer than 3 peers in a direction => "no signal".)
    for col in DEFAULT_PEER_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    return df