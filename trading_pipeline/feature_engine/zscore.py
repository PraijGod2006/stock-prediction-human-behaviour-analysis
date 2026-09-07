"""
========================================================================================
FEATURE ENGINE: Z-Score Calculator Module (14 Distinct Formulas)
========================================================================================
This module calculates 14 mathematically distinct Z-score metrics for financial
time-series data. 

MATHEMATICAL FOUNDATION:
A Z-score standardizes a value relative to its historical distribution:
    Z = (X_t - mu) / sigma

WHY Z-SCORES FOR TRADING?
1. Normalization: Removes price scale dependencies (MRF at Rs 100,000 vs IDEA at Rs 8).
2. Regime Invariance: Normalizes for fluctuating volatility regimes.
3. Mean-Reversion Signals: Extreme |Z| > 2.0 indicates statistical overextension.

STRICT LEAKAGE PREVENTION:
To ensure 100% zero lookahead bias:
- Every rolling calculation applies .shift(1) BEFORE computing the Z-score.
  This guarantees that the feature at time t uses ONLY information up to t-1.
- All metrics reset at session boundaries (> 4 hour gap), preventing overnight
  gap distortion from polluting intraday rolling windows.
========================================================================================
"""

import numpy as np
import pandas as pd
import numpy as np
from typing import Optional


class ZScoreCalculator:
    """
    Calculates various Z-scores with lookahead bias prevention by applying shift(1).
    All methods reset their rolling statistics at session boundaries (> 4 hour gaps).
    Computes 14 distinct Z-score feature transformations.
    All rolling statistics use .shift(1) to strictly eliminate lookahead bias.
    All windows reset at trading session boundaries (gaps > 4 hours).
    """
    

    def _get_session_groups(self, index: pd.DatetimeIndex) -> pd.Series:
        """
        Detects session boundaries by finding gaps > 4 hours in the index.
        Returns a group ID series used for groupby operations.
        Identifies session boundaries by detecting gaps > 4 hours in the DatetimeIndex.
        Returns unique session integer IDs for groupby operations.
        
        Args:
            index (pd.DatetimeIndex): The datetime index of the dataframe.
            index: DatetimeIndex of the series/dataframe.
            
        Returns:
            pd.Series: Integer series representing session group IDs.
            pd.Series: Integer series indicating session membership.
        """
        # Calculate time difference between consecutive rows
        tdelta = index.to_series().diff()
        # Create a boolean mask where difference > 4 hours
        new_session = tdelta > pd.Timedelta(hours=4)
        # Cumulative sum to create unique group IDs for each session
        return new_session.cumsum()

    # -------------------------------------------------------------------------
    # Formula 1: Basic Z-Score (Sample Standard Deviation, ddof=1)
    # -------------------------------------------------------------------------
    def basic_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculates basic Z-score with rolling logic.
        Formula: Z = (X - mu) / sigma
        
        Args:
            series (pd.Series): Input data series, expected to have a DatetimeIndex.
            window (int): Window size. Default is 20.
            
        Returns:
            pd.Series: Z-score series.
        Formula 1: Basic Z-Score using SAMPLE standard deviation s (ddof=1).
            Z_t = (X_t - mu_{t-1}) / s_{t-1}
        Divides by N - 1, producing an unbiased estimator for sample variance.
        """
        # Group by sessions
        groups = self._get_session_groups(series.index)
        
        # Calculate rolling mean and std within each group, and shift(1) to prevent lookahead
        # Shift is applied after rolling to ensure t's value is not in t's mean/std
        rolling_mean = series.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        rolling_std = series.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        
        # Calculate Z-score
        zscore = (series - rolling_mean) / rolling_std
        return zscore
        mean = series.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        std = series.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std(ddof=1).shift(1))
        return (series - mean) / std.replace(0, np.nan)

    # -------------------------------------------------------------------------
    # Formula 2: Rolling Z-Score (Population Standard Deviation, ddof=0)
    # -------------------------------------------------------------------------
    def rolling_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculates rolling Z-score using POPULATION standard deviation (ddof=0).
        
        This is distinct from basic_zscore which uses SAMPLE std (ddof=1, Pandas default).
        Population std divides by N instead of N-1, producing slightly smaller variance
        estimates. This matters for short windows where N-1 vs N makes a meaningful difference.
        
        Formula: Z_t = (X_t - mu_(t,N)) / sigma_pop_(t,N)
        where sigma_pop uses ddof=0 (divides by N, not N-1).
        
        AUDIT FIX: Was previously an exact copy of basic_zscore (return self.basic_zscore).
        Now uses ddof=0 for population standard deviation.
        
        Args:
            series (pd.Series): Input data series, expected to have a DatetimeIndex.
            window (int): Lookback window size. Default is 20.
            
        Returns:
            pd.Series: Rolling Z-score series with population sigma.
        Formula 2: Rolling Z-Score using POPULATION standard deviation sigma (ddof=0).
            Z_t = (X_t - mu_{t-1}) / sigma_{t-1}
        Divides by N (not N-1). Mathematically distinct from basic_zscore.
        Particularly relevant in high-frequency trading where N bars represent
        the entire population of states in that fixed window.
        """
        groups = self._get_session_groups(series.index)
        
        # Population std (ddof=0) vs sample std (ddof=1) in basic_zscore
        rolling_mean = series.groupby(groups).transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        rolling_std = series.groupby(groups).transform(
            lambda x: x.rolling(window=window, min_periods=2).std(ddof=0).shift(1)
        )
        
        zscore = (series - rolling_mean) / rolling_std
        return zscore
        mean = series.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        std = series.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std(ddof=0).shift(1))
        return (series - mean) / std.replace(0, np.nan)

    # -------------------------------------------------------------------------
    # Formula 3: Simple Return Z-Score
    # -------------------------------------------------------------------------
    def return_zscore(self, price_series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculates Z-score of simple returns.
        Formula: Simple return R_t = (P_t - P_(t-1))/P_(t-1)
                 Z_(R,t) = (R_t - mu_R) / sigma_R
        
        Args:
            price_series (pd.Series): Price series, expected to have a DatetimeIndex.
            window (int): Lookback window size. Default is 20.
            
        Returns:
            pd.Series: Z-score of returns.
        Formula 3: Z-score of simple returns.
            R_t = (P_t - P_{t-1}) / P_{t-1}
            Z_{R,t} = (R_t - mu_{R, t-1}) / sigma_{R, t-1}
        Captures short-term linear return momentum and exhaustion.
        """
        groups = self._get_session_groups(price_series.index)
        
        # Calculate simple returns within sessions
        returns = price_series.groupby(groups).transform(lambda x: x.pct_change())
        
        # We need to shift(1) rolling mean and std of returns
        rolling_mean = returns.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        rolling_std = returns.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        
        zscore = (returns - rolling_mean) / rolling_std
        return zscore
        mean = returns.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        std = returns.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        return (returns - mean) / std.replace(0, np.nan)

    # -------------------------------------------------------------------------
    # Formula 4: Log Return Z-Score
    # -------------------------------------------------------------------------
    def log_return_zscore(self, price_series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculates Z-score of log returns.
        Formula: Log return r_t = ln(P_t/P_(t-1))
                 Z_(r,t) = (r_t - mu_r) / sigma_r
        
        Args:
            price_series (pd.Series): Price series, expected to have a DatetimeIndex.
            window (int): Lookback window size. Default is 20.
            
        Returns:
            pd.Series: Z-score of log returns.
        Formula 4: Z-score of continuously compounded logarithmic returns.
            r_t = ln(P_t / P_{t-1})
            Z_{r,t} = (r_t - mu_{r, t-1}) / sigma_{r, t-1}
        Log returns are additive across time and symmetric for gains/losses.
        """
        groups = self._get_session_groups(price_series.index)
        
        # Calculate log returns within sessions
        log_returns = price_series.groupby(groups).transform(lambda x: np.log(x / x.shift(1)))
        
        # Shift(1) rolling mean and std
        rolling_mean = log_returns.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        rolling_std = log_returns.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        
        zscore = (log_returns - rolling_mean) / rolling_std
        return zscore
        log_ret = price_series.groupby(groups).transform(lambda x: np.log(x / x.shift(1)))
        mean = log_ret.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        std = log_ret.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        return (log_ret - mean) / std.replace(0, np.nan)

    # -------------------------------------------------------------------------
    # Formula 5: Volume Z-Score
    # -------------------------------------------------------------------------
    def volume_zscore(self, volume_series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculates Z-score of volume.
        Formula: Z_(V,t) = (V_t - mu_V) / sigma_V
        
        Args:
            volume_series (pd.Series): Volume series, expected to have a DatetimeIndex.
            window (int): Lookback window size. Default is 20.
            
        Returns:
            pd.Series: Z-score of volume.
        Formula 5: Z-score of trading volume.
            Z_{V,t} = (V_t - mu_{V, t-1}) / sigma_{V, t-1}
        Detects unusual volume bursts indicating institutional participation or stop cascades.
        """
        return self.basic_zscore(volume_series, window)

    # -------------------------------------------------------------------------
    # Formula 6: Volatility Z-Score (computed strictly from returns, not price)
    # -------------------------------------------------------------------------
    def volatility_zscore(self, return_series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculates Z-score of rolling volatility (standard deviation of returns).
        Formula: sigma_t = Std of returns over window, then Z_(sigma,t)
        
        Args:
            return_series (pd.Series): Return series, expected to have a DatetimeIndex.
            window (int): Lookback window size. Default is 20.
            
        Returns:
            pd.Series: Z-score of volatility.
        Formula 6: Z-score of rolling return volatility.
            sigma_t = Std(R_{t-window:t-1})
            Z_{sigma,t} = (sigma_t - mu_{sigma, t-1}) / sigma_{sigma, t-1}
        Computed strictly from returns (not raw price). Identifies regime shifts
        between low-volatility consolidation and high-volatility expansion.
        """
        groups = self._get_session_groups(return_series.index)
        
        # Calculate rolling volatility within sessions. 
        # Shift(1) the rolling calculation so it does not see t
        volatility = return_series.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        
        # Now we want the Z-score of the volatility itself.
        # This requires rolling mean and std of the volatility, shifted.
        rolling_mean = volatility.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        rolling_std = volatility.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        
        zscore = (volatility - rolling_mean) / rolling_std
        return zscore
        # Shift(1) rolling std of returns to prevent leakage
        vol = return_series.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        mean_vol = vol.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        std_vol = vol.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        return (vol - mean_vol) / std_vol.replace(0, np.nan)

    # -------------------------------------------------------------------------
    # Formula 7: Price vs. Moving Average Z-Score
    # -------------------------------------------------------------------------
    def price_vs_ma_zscore(self, price_series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculates Z-score of price relative to its moving average.
        Formula: MA = rolling mean, Z = (P - MA) / rolling_std
        
        Args:
            price_series (pd.Series): Price series, expected to have a DatetimeIndex.
            window (int): Lookback window size. Default is 20.
            
        Returns:
            pd.Series: Z-score of price vs MA.
        Formula 7: Z-score of current price relative to its historical moving average.
            Z_{P, MA} = (P_t - MA_{t-1}) / sigma_{P, t-1}
        Classic mean-reversion indicator measuring distance from fair value in units of std dev.
        """
        groups = self._get_session_groups(price_series.index)
        
        # Calculate rolling mean and std, shift(1) to avoid leakage
        ma = price_series.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        rolling_std = price_series.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        
        zscore = (price_series - ma) / rolling_std
        return zscore
        std = price_series.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        return (price_series - ma) / std.replace(0, np.nan)

    # -------------------------------------------------------------------------
    # Formula 8: Normalized Pair Spread Z-Score
    # -------------------------------------------------------------------------
    def spread_zscore(self, price_a: pd.Series, price_b: pd.Series, beta: float = 1.0, window: int = 20) -> pd.Series:
        """
        Calculates Z-score of a spread between two assets.
        Formula: S = P_A - beta*P_B, Z = (S - mu_S) / sigma_S
        
        Args:
            price_a (pd.Series): Price series for asset A.
            price_b (pd.Series): Price series for asset B.
            beta (float): Hedge ratio or multiplier. Default is 1.0.
            window (int): Lookback window size. Default is 20.
            
        Returns:
            pd.Series: Z-score of the spread.
        Formula 8: Z-score of the statistical arbitrage cointegration spread.
            Spread_t = P_A,t - beta * P_B,t
            Z_{S,t} = (Spread_t - mu_{S, t-1}) / sigma_{S, t-1}
        Essential for pairs trading and cross-asset relative value strategies.
        """
        # Calculate spread
        spread = price_a - beta * price_b
        return self.basic_zscore(spread, window)

    # -------------------------------------------------------------------------
    # Formula 9: Dollar Volume (Turnover) Z-Score
    # -------------------------------------------------------------------------
    def dollar_volume_zscore(self, price_series: pd.Series, volume_series: pd.Series, window: int = 20) -> pd.Series:
        """
        Formula 9: Z-score of Dollar Volume (Price * Volume).
            Turnover_t = P_t * V_t
            Z_{DV,t} = (Turnover_t - mu_{DV, t-1}) / sigma_{DV, t-1}
        Filters out low-priced volume spikes and identifies true institutional capital flow.
        """
        dollar_vol = price_series * volume_series
        return self.basic_zscore(dollar_vol, window)

    # -------------------------------------------------------------------------
    # Formula 10: High-Low Normalized Range (Bar Spread) Z-Score
    # -------------------------------------------------------------------------
    def range_zscore(self, high_series: pd.Series, low_series: pd.Series, window: int = 20) -> pd.Series:
        """
        Formula 10: Z-score of the percentage intrabar range.
            Range_t = (High_t - Low_t) / Low_t
            Z_{Range,t} = (Range_t - mu_{Range, t-1}) / sigma_{Range, t-1}
        Measures intrabar volatility expansion, highlighting liquidity squeezes.
        """
        range_pct = (high_series - low_series) / low_series.replace(0, np.nan)
        return self.basic_zscore(range_pct, window)

    # -------------------------------------------------------------------------
    # Formula 11: Price vs. Exponential Moving Average (EMA) Z-Score
    # -------------------------------------------------------------------------
    def price_vs_ema_zscore(self, price_series: pd.Series, span: int = 20) -> pd.Series:
        """
        Formula 11: Z-score of price relative to its Exponential Moving Average.
            EMA_t = ExponentialWeightedMean(P_{0:t-1})
            Z_{P, EMA} = (P_t - EMA_{t-1}) / rolling_std_{t-1}
        Weights recent price action more heavily than simple SMA, reacting faster to trend changes.
        """
        groups = self._get_session_groups(price_series.index)
        ema = price_series.groupby(groups).transform(lambda x: x.ewm(span=span, adjust=False).mean().shift(1))
        std = price_series.groupby(groups).transform(lambda x: x.rolling(window=span, min_periods=2).std().shift(1))
        return (price_series - ema) / std.replace(0, np.nan)

    # -------------------------------------------------------------------------
    # Formula 12: Return Skewness Z-Score
    # -------------------------------------------------------------------------
    def skewness_zscore(self, return_series: pd.Series, window: int = 20) -> pd.Series:
        """
        Formula 12: Rolling return asymmetry / skewness Z-score.
            Skew_t = E[((R - mu) / sigma)^3]
            Z_{Skew,t} = (Skew_t - mu_{Skew, t-1}) / sigma_{Skew, t-1}
        Detects tail-risk imbalances (e.g. fat left tails preceding flash crashes).
        """
        groups = self._get_session_groups(return_series.index)
        skew = return_series.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=5).skew().shift(1))
        mean_skew = skew.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).mean().shift(1))
        std_skew = skew.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        return (skew - mean_skew) / std_skew.replace(0, np.nan)

    # -------------------------------------------------------------------------
    # Formula 13: Return Kurtosis Z-Score
    # -------------------------------------------------------------------------
    def kurtosis_zscore(self, return_series: pd.Series, window: int = 20) -> pd.Series:
        """
        Formula 13: Rolling return peakedness / kurtosis Z-score.
            Kurt_t = E[((R - mu) / sigma)^4] - 3
            Z_{Kurt,t} = (Kurt_t - mu_{Kurt, t-1}) / sigma_{Kurt, t-1}
        Identifies fat-tail market regimes prone to extreme price leaps.
        """
        groups = self._get_session_groups(return_series.index)
        kurt = return_series.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=5).kurt().shift(1))
        mean_kurt = kurt.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).mean().shift(1))
        std_kurt = kurt.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        return (kurt - mean_kurt) / std_kurt.replace(0, np.nan)

    # -------------------------------------------------------------------------
    # Formula 14: Return Momentum / Acceleration Z-Score
    # -------------------------------------------------------------------------
    def momentum_zscore(self, price_series: pd.Series, lag: int = 5, window: int = 20) -> pd.Series:
        """
        Formula 14: Z-score of multi-period price momentum (ROC).
            Mom_t = (P_t - P_{t-lag}) / P_{t-lag}
            Z_{Mom,t} = (Mom_t - mu_{Mom, t-1}) / sigma_{Mom, t-1}
        Measures medium-term directional velocity relative to recent historical volatility.
        """
        groups = self._get_session_groups(price_series.index)
        mom = price_series.groupby(groups).transform(lambda x: x.pct_change(periods=lag))
        mean_mom = mom.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        std_mom = mom.groupby(groups).transform(lambda x: x.rolling(window=window, min_periods=2).std().shift(1))
        return (mom - mean_mom) / std_mom.replace(0, np.nan)
