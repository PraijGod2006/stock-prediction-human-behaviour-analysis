import pandas as pd

from feature_engine.reference_price import compute_reference_price


class Aggregator:
    """
    Handles aggregation and resampling of financial time series data.
    """

    def resample_to_5min(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples 1-minute OHLCV DataFrame to 5-minute intervals.
        Preserves the datetime index properly.
        Computes mid_price = (open + close) / 2.
        
        Aggregation rules:
        - Open: First
        - High: Max
        - Low: Min
        - Close: Last
        - Volume: Sum
        
        Args:
            df (pd.DataFrame): 1-minute OHLCV DataFrame with a DatetimeIndex.
            
        Returns:
            pd.DataFrame: Resampled 5-minute DataFrame.
        """
        # Define aggregation dictionary
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample to 5-minute intervals using label='left' and closed='left'
        resampled_df = df.resample('5min', label='left', closed='left').agg(agg_dict)
        
        # Drop rows where all elements are NaN (periods with no trades)
        resampled_df = resampled_df.dropna(how='all')
        
        # Compute mid_price using the single source of truth: Typical Price = (H+L+C)/3
        # NOTE: No real bid/ask data exists in the raw OHLCV feed. See reference_price.py.
        resampled_df['mid_price'] = compute_reference_price(resampled_df)
        
        return resampled_df
