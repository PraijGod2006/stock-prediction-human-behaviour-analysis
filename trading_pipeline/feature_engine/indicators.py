from typing import Any

import pandas as pd


class TechnicalIndicators:
    """
    Calculates technical indicators with lookahead bias prevention by applying shift(1).
    All methods reset their statistics at session boundaries (> 4 hour gaps).
    """

    def _get_session_groups(self, index: pd.Index) -> pd.Series:
        """
        Detects session boundaries by finding gaps > 4 hours in the index.
        Returns a group ID series used for groupby operations.
        
        Args:
            index (pd.DatetimeIndex): The datetime index of the dataframe.
            
        Returns:
            pd.Series: Integer series representing session group IDs.
        """
        # Calculate time difference between consecutive rows
        tdelta = index.to_series().diff()
        # Create a boolean mask where difference > 4 hours
        new_session = tdelta > pd.Timedelta(hours=4)
        # Cumulative sum to create unique group IDs for each session
        return new_session.cumsum()

    def rsi(self, price_series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculates Relative Strength Index (RSI) using Wilder's smoothing.
        Formula: RSI = 100 - (100 / (1 + RS))
                 RS = Average Gain / Average Loss
        
        Args:
            price_series (pd.Series): Price series, expected to have a DatetimeIndex.
            period (int): Lookback period. Default is 14.
            
        Returns:
            pd.Series: RSI values.
        """
        groups = self._get_session_groups(price_series.index)
        
        def _calc_rsi(group_series: pd.Series) -> pd.Series:
            # Shift the series by 1 to prevent lookahead bias
            # Compute difference of shifted prices
            delta = group_series.shift(1).diff()
            
            # Separate gains and losses
            gain = delta.clip(lower=0)
            loss = -1 * delta.clip(upper=0)
            
            # Calculate Exponential Moving Average (Wilder's smoothing)
            avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
            avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
            
        return price_series.groupby(groups, group_keys=False).apply(_calc_rsi)

    def vwap(self, df: pd.DataFrame, price_col: str = 'mid_price', volume_col: str = 'volume') -> pd.Series:
        """
        Calculates Session-resetting Volume Weighted Average Price (VWAP).
        Formula: VWAP = cumsum(price*volume) / cumsum(volume)
        
        Args:
            df (pd.DataFrame): DataFrame containing price and volume columns.
            price_col (str): Column name for price. Default is 'mid_price'.
            volume_col (str): Column name for volume. Default is 'volume'.
            
        Returns:
            pd.Series: VWAP values.
        """
        groups = self._get_session_groups(df.index)
        
        # Shift price and volume by 1 to prevent lookahead bias
        shifted_price = df[price_col].shift(1)
        shifted_volume = df[volume_col].shift(1)
        
        # Vectorized cumulative product and cumulative volume per session group
        cum_vol_price = (shifted_price * shifted_volume).groupby(groups).cumsum()
        cum_vol = shifted_volume.groupby(groups).cumsum()
        
        # Compute VWAP. If volume is zero (e.g. index data), fall back to shifted_price
        vwap_series = cum_vol_price / cum_vol
        vwap_series = vwap_series.fillna(shifted_price)
        return pd.Series(vwap_series, index=df.index, name='vwap')

    def distance_from_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates distance of current mid-price from VWAP.
        Formula: (Current Mid-Price - VWAP) / VWAP
        
        Args:
            df (pd.DataFrame): DataFrame containing mid_price and volume columns.
            
        Returns:
            pd.Series: Distance from VWAP.
        """
        # Calculate VWAP
        current_vwap = self.vwap(df, price_col='mid_price', volume_col='volume')
        
        # Calculate distance
        # We use current mid_price vs shifted VWAP calculation
        distance = (df['mid_price'] - current_vwap) / current_vwap
        return pd.Series(distance.fillna(0.0), index=df.index, name='distance_from_vwap')

    def exhaustion_features(self, df: pd.DataFrame, zscore_calc: Any) -> pd.DataFrame:
        """
        Calculates exhaustion features for the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame containing mid_price and volume.
            zscore_calc: Instance of ZScoreCalculator.
            
        Returns:
            pd.DataFrame: DataFrame containing exhaustion features.
        """
        # 20-min Volume Z-Score
        vol_z = zscore_calc.volume_zscore(df['volume'], window=20)
        
        # Distance from VWAP
        dist_vwap = self.distance_from_vwap(df)
        
        # 14-min RSI
        rsi_14 = self.rsi(pd.Series(df['mid_price']), period=14)
        
        # Combine into DataFrame
        result = pd.DataFrame({
            'volume_zscore_20': vol_z,
            'distance_from_vwap': dist_vwap,
            'rsi_14': rsi_14
        }, index=df.index)
        
        return result
