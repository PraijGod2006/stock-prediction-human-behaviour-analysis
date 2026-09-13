"""
========================================================================================
REFERENCE PRICE: Single Source of Truth for the Entire Pipeline
========================================================================================
Every module that needs "the price" — target generation, feature engineering,
backtest fill execution, correlation matrix computation — must call this function.

WHY TYPICAL PRICE?
- Formula: TP = (High + Low + Close) / 3
- Standard in technical analysis (CCI, Money Flow Index, etc.)
- Captures intra-bar extension: a bar that spikes +2% and reverses to flat
  produces a DIFFERENT Typical Price than a genuinely flat bar, unlike (O+C)/2
  which treats both identically.
- More information-dense than (O+C)/2 for the same OHLCV input.

WHY NOT (Bid + Ask) / 2?
- The raw data (NIFTY50 1-minute CSVs) contains only OHLCV columns.
- No Level-1 order book / bid-ask quote data is available.
- This is a known limitation, documented here rather than silently masked.

ENFORCEMENT:
This is the ONLY function that should compute a reference price anywhere in
the codebase. Search for any inline (open+close)/2, (high+low+close)/3, or
raw 'close' used as a reference price and replace with a call to this module.
========================================================================================
"""

import pandas as pd


def compute_reference_price(
    df: pd.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pd.Series:
    """
    Computes the Typical Price = (High + Low + Close) / 3.

    This is the single source of truth for "the price" used throughout the
    entire trading pipeline — features, targets, backtest fills, and
    correlation matrix computation.

    Args:
        df: DataFrame containing high, low, close columns.
        high_col: Name of the high price column.
        low_col: Name of the low price column.
        close_col: Name of the close price column.

    Returns:
        pd.Series of Typical Price values, same index as input.
    """
    return (df[high_col] + df[low_col] + df[close_col]) / 3.0

