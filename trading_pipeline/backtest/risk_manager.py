"""
========================================================================================
RISK MANAGEMENT: Position Sizing, Exposure Caps, Circuit Breakers, Kill Switch
========================================================================================
This module prevents the trading system from blowing up your account.

Even if the models are brilliant, risk management is what separates profitable
traders from bankrupt ones. Warren Buffett's Rule #1: "Never lose money."

COMPONENTS:
1. Position Sizing (ATR-Adjusted): Instead of betting a fixed dollar amount,
   we adjust position size based on how volatile the stock is. Calm stocks get
   bigger positions, crazy stocks get smaller ones.

2. Correlation-Aware Exposure Cap: If HDFC Bank, ICICI Bank, and Kotak Bank
   are all in the banking sector and highly correlated, going LONG on all three
   simultaneously is essentially a 3x bet on banking. We cap this.

3. Daily Max-Loss Circuit Breaker: If the day's P&L drops below -2%, we stop
   all trading for the rest of the day. This prevents catastrophic spiral losses.

4. Manual Kill Switch: A file-based emergency stop. If the file
   'artifacts/KILL_SWITCH' exists, ALL signal generation halts immediately.
   This is a regulatory requirement for automated trading systems.
========================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional


# -------------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------------
ARTIFACTS_DIR = r"d:\CODE\rajasthani\trading_pipeline\artifacts"
KILL_SWITCH_FILE = os.path.join(ARTIFACTS_DIR, "KILL_SWITCH")
PEER_MAP_FILE = os.path.join(ARTIFACTS_DIR, "peer_map.json")


def check_kill_switch() -> bool:
    """
    Checks if the manual kill switch is engaged.
    
    If the file 'artifacts/KILL_SWITCH' exists, this returns True and
    ALL trading signals must be suppressed immediately.
    
    This is a regulatory requirement for automated trading systems. A human
    operator must be able to halt the system at any time by simply creating
    this file (e.g., `echo STOP > artifacts/KILL_SWITCH`).
    
    Returns:
        True if kill switch is engaged (STOP trading), False otherwise.
    """
    if os.path.exists(KILL_SWITCH_FILE):
        print("KILL SWITCH ENGAGED. All trading halted.")
        return True
    return False


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates Average True Range (ATR), a measure of volatility.
    
    ATR measures the average size of price bars (including gaps). It tells us
    how much a stock typically moves in a single bar. High ATR = volatile stock.
    
    Formula:
        True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = Rolling Mean of True Range over `period` bars
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        period: Lookback period for the rolling average (default 14).
    
    Returns:
        pd.Series of ATR values.
    """
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)
    
    # True Range is the maximum of three calculations:
    # 1. Current bar's range (high - low)
    # 2. Gap up from previous close (high - prev_close)
    # 3. Gap down from previous close (low - prev_close)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_position_size(
    capital: float,
    risk_pct: float,
    atr: float,
    price: float,
    max_position_pct: float = 0.10
) -> float:
    """
    Calculates position size using ATR-adjusted fractional sizing.
    
    Instead of always betting 2% of capital, we adjust based on volatility:
    - Low volatility stock (small ATR) → Larger position (more shares)
    - High volatility stock (large ATR) → Smaller position (fewer shares)
    
    Formula:
        Risk Amount = Capital * Risk%
        Shares = Risk Amount / (ATR * 2)  (2x ATR as stop-loss distance)
        Position Value = Shares * Price
        Cap at max_position_pct of capital
    
    Args:
        capital: Current portfolio value.
        risk_pct: Fraction of capital to risk (e.g., 0.02 = 2%).
        atr: Current ATR value for the stock.
        price: Current stock price.
        max_position_pct: Maximum fraction of capital for a single position.
    
    Returns:
        Number of shares to trade (float, round down for actual orders).
    """
    if atr <= 0 or price <= 0:
        return 0.0
    
    # Amount we're willing to lose on this trade
    risk_amount = capital * risk_pct
    
    # Stop-loss distance = 2x ATR (a common professional rule of thumb)
    stop_distance = atr * 2.0
    
    # Number of shares where hitting the stop-loss = losing exactly risk_amount
    shares = risk_amount / stop_distance
    
    # Cap the total position value at max_position_pct of capital
    max_shares = (capital * max_position_pct) / price
    shares = min(shares, max_shares)
    
    return max(0.0, shares)


class ExposureManager:
    """
    Prevents over-concentration in correlated positions.
    
    If HDFC Bank, ICICI Bank, and Kotak Bank all signal LONG simultaneously,
    this manager ensures we don't take all three positions — because if banking
    crashes, all three would lose together, amplifying the damage.
    
    It uses the correlation matrix peer_map to detect when multiple open
    positions belong to the same cluster of highly correlated stocks.
    """
    
    def __init__(self, max_correlated_positions: int = 2, correlation_threshold: float = 0.7):
        """
        Args:
            max_correlated_positions: Max number of positions allowed in the same
                                      correlation cluster.
            correlation_threshold: Minimum correlation to be considered "same cluster".
        """
        self.max_correlated_positions = max_correlated_positions
        self.correlation_threshold = correlation_threshold
        self.peer_map: dict = {}
        self.open_positions: set = set()
        
        # Load peer map if available
        if os.path.exists(PEER_MAP_FILE):
            with open(PEER_MAP_FILE, 'r') as f:
                self.peer_map = json.load(f)
    
    def can_open_position(self, symbol: str) -> bool:
        """
        Checks if we're allowed to open a new position in the given symbol
        without exceeding the correlation-based exposure cap.
        
        Returns:
            True if position is allowed, False if blocked by exposure limits.
        """
        if symbol in self.open_positions:
            return False  # Already have a position in this stock
        
        # Find this symbol's highly correlated peers
        peers = self.peer_map.get(symbol, {}).get('top_positive', [])
        
        # Count how many correlated peers we already have open positions in
        correlated_open = sum(1 for peer in peers if peer in self.open_positions)
        
        if correlated_open >= self.max_correlated_positions:
            print(f"  EXPOSURE CAP: Blocked {symbol} — {correlated_open} correlated positions already open.")
            return False
        
        return True
    
    def register_position(self, symbol: str):
        """Records that a position has been opened in the given symbol."""
        self.open_positions.add(symbol)
    
    def close_position(self, symbol: str):
        """Records that a position has been closed."""
        self.open_positions.discard(symbol)


class CircuitBreaker:
    """
    Daily max-loss circuit breaker. If the day's P&L drops below a threshold,
    ALL trading is halted for the remainder of the day.
    
    This prevents catastrophic spiral losses where the model keeps trading
    into a crashing market, each trade losing more than the last.
    """
    
    def __init__(self, max_daily_loss_pct: float = -0.02):
        """
        Args:
            max_daily_loss_pct: Maximum daily loss as fraction of capital (e.g., -0.02 = -2%).
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.daily_pnl = 0.0
        self.starting_capital = 0.0
        self.is_tripped = False
        self.current_date = None
    
    def reset_day(self, capital: float, date):
        """Resets the circuit breaker at the start of a new trading day."""
        self.daily_pnl = 0.0
        self.starting_capital = capital
        self.is_tripped = False
        self.current_date = date
    
    def record_trade(self, pnl: float) -> bool:
        """
        Records a trade's P&L and checks if the circuit breaker should trip.
        
        Args:
            pnl: Profit/loss of the trade in currency units.
        
        Returns:
            True if trading can continue, False if circuit breaker has tripped.
        """
        self.daily_pnl += pnl
        
        if self.starting_capital > 0:
            daily_return = self.daily_pnl / self.starting_capital
            
            if daily_return <= self.max_daily_loss_pct:
                self.is_tripped = True
                print(f"  CIRCUIT BREAKER TRIPPED: Daily loss {daily_return*100:.2f}% "
                      f"exceeded threshold {self.max_daily_loss_pct*100:.1f}%")
                return False
        
        return True
    
    def can_trade(self) -> bool:
        """Returns True if trading is allowed (circuit breaker not tripped)."""
        return not self.is_tripped
