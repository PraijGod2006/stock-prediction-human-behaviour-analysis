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

    def relative_volume_time_of_day(self, df: pd.DataFrame, historical_window_days: int = 30) -> pd.Series:
        """
        Calculates time-of-day relative volume.
        Compares current volume to the historical average volume for the same minute-of-day
        across the previous `historical_window_days` trading days.
        
        Args:
            df (pd.DataFrame): DataFrame containing 'volume' and a DatetimeIndex.
            historical_window_days (int): Lookback window in days (sessions). Default is 30.
            
        Returns:
            pd.Series: Relative volume for the time of day.
        """
        sessions = self._get_session_groups(df.index)
        mod = df.index.hour * 60 + df.index.minute
        
        temp = pd.DataFrame({'vol': df['volume'], 'mod': mod, 'session': sessions})
        
        def _calc_baseline(x):
            return x.shift(1).rolling(window=historical_window_days, min_periods=1).mean()
            
        hist_avg = temp.groupby('mod')['vol'].transform(_calc_baseline)
        
        rel_vol = temp['vol'] / hist_avg
        rel_vol = rel_vol.where(hist_avg.notna(), 1.0)
        rel_vol = rel_vol.where(hist_avg != 0, 1.0)
        
        return pd.Series(rel_vol, index=df.index, name='relative_volume_tod')

    def bollinger_bands(self, series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Calculates Bollinger Bands features with session-resetting.
        
        Args:
            series (pd.Series): Price series.
            window (int): Rolling window size. Default is 20.
            num_std (float): Number of standard deviations for bands. Default is 2.0.
            
        Returns:
            pd.DataFrame: DataFrame containing 'bollinger_pctb', 'bollinger_dist_upper', 
                          and 'bollinger_bandwidth'.
        """
        groups = self._get_session_groups(series.index)
        
        def _calc_bb(group_series: pd.Series) -> pd.DataFrame:
            middle = group_series.rolling(window=window).mean().shift(1)
            rolling_std_val = group_series.rolling(window=window).std().shift(1)
            
            upper = middle + num_std * rolling_std_val
            lower = middle - num_std * rolling_std_val
            
            pctb = (group_series - lower) / (upper - lower)
            dist_upper = (group_series - upper) / upper
            bandwidth = (upper - lower) / middle
            
            zero_width = (upper == lower)
            pctb = pctb.mask(zero_width, 0.5)
            dist_upper = dist_upper.mask(zero_width, 0.0)
            bandwidth = bandwidth.mask(zero_width, 0.0)
            
            return pd.DataFrame({
                'bollinger_pctb': pctb,
                'bollinger_dist_upper': dist_upper,
                'bollinger_bandwidth': bandwidth
            }, index=group_series.index)
            
        return series.groupby(groups, group_keys=False).apply(_calc_bb)
