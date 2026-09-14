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


from dataclasses import dataclass
from config import (
    EXIT_MAX_HOLD_BARS,
    EXIT_STOP_LOSS_ATR_MULT,
    EXIT_TAKE_PROFIT_SAFETY,
    EXIT_TRAILING_PCT,
    KELLY_FRACTION,
    POSITION_SIZING_METHOD,
)


def calculate_kelly_position_size(
    capital: float,
    calibrated_prob: float,
    predicted_runup: float,
    atr: float,
    price: float,
    realized_trades: list[float] | None = None,
    fraction: float = KELLY_FRACTION,
    max_position_pct: float = 0.10,
) -> float:
    """
    Calculates position size using fractional Kelly Criterion.

    Formula (ExitManager aligned):
        b = (0.7 * predicted_runup) / (1.5 * ATR_pct)
        If >= 50 realized trades exist: b = avg_win / avg_loss from history.
        f* = fraction * (p - (1 - p) / b)
    """
    if price <= 0 or atr <= 0 or calibrated_prob <= 0:
        return 0.0

    p = float(calibrated_prob)
    atr_pct = atr / price

    if realized_trades and len(realized_trades) >= 50:
        wins = [t for t in realized_trades if t > 0]
        losses = [abs(t) for t in realized_trades if t < 0]
        if wins and losses:
            b = float(np.mean(wins) / np.mean(losses))
        else:
            b = (EXIT_TAKE_PROFIT_SAFETY * max(predicted_runup, 0.002)) / (EXIT_STOP_LOSS_ATR_MULT * max(atr_pct, 0.001))
    else:
        b = (EXIT_TAKE_PROFIT_SAFETY * max(predicted_runup, 0.002)) / (EXIT_STOP_LOSS_ATR_MULT * max(atr_pct, 0.001))

    if b <= 0:
        return 0.0

    f_star = fraction * (p - (1.0 - p) / b)

    if f_star <= 0:
        return 0.0

    f_capped = min(f_star, max_position_pct)
    position_value = capital * f_capped
    shares = position_value / price

    return max(0.0, float(shares))


def calculate_position_size(
    capital: float,
    risk_pct: float,
    atr: float,
    price: float,
    max_position_pct: float = 0.10,
    calibrated_prob: float | None = None,
    predicted_runup: float | None = None,
    sizing_method: str = POSITION_SIZING_METHOD,
    realized_trades: list[float] | None = None,
) -> float:
    """
    Calculates position size using configurable sizing methods:
    - 'atr'   : Volatility-adjusted fractional risk sizing
    - 'kelly' : Fractional Kelly Criterion using ExitManager-derived win/loss ratio
    - 'hybrid': Conservative minimum of ATR sizing and Kelly sizing
    """
    if atr <= 0 or price <= 0:
        return 0.0

    # 1. ATR sizing
    risk_amount = capital * risk_pct
    stop_distance = atr * EXIT_STOP_LOSS_ATR_MULT
    atr_shares = risk_amount / stop_distance
    max_shares = (capital * max_position_pct) / price
    atr_shares = min(atr_shares, max_shares)

    if sizing_method == "atr" or calibrated_prob is None or predicted_runup is None:
        return max(0.0, float(atr_shares))

    # 2. Kelly sizing
    kelly_shares = calculate_kelly_position_size(
        capital=capital,
        calibrated_prob=calibrated_prob,
        predicted_runup=predicted_runup,
        atr=atr,
        price=price,
        realized_trades=realized_trades,
        fraction=KELLY_FRACTION,
        max_position_pct=max_position_pct,
    )

    if sizing_method == "kelly":
        return max(0.0, float(kelly_shares))
    elif sizing_method == "hybrid":
        return max(0.0, float(min(atr_shares, kelly_shares)))
    else:
        return max(0.0, float(atr_shares))


@dataclass
class ExitDecision:
    """Dataclass returned by ExitManager."""
    should_exit: bool
    exit_reason: str  # 'STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP', 'TIME_EXIT', or 'HOLD'
    exit_price: float


class ExitManager:
    """
    Multi-barrier exit manager for open positions (Fix Group 8).
    Evaluates:
      1. Hard Stop-Loss (1.5x ATR from entry)
      2. Take-Profit (Model 3b predicted runup * 0.7)
      3. Trailing Stop (ratchets 0.5% once excursion is favorable)
      4. Time-based fallback exit (5 bars)
    """

    def __init__(
        self,
        stop_loss_atr_mult: float = EXIT_STOP_LOSS_ATR_MULT,
        take_profit_safety: float = EXIT_TAKE_PROFIT_SAFETY,
        trailing_pct: float = EXIT_TRAILING_PCT,
        max_hold_bars: int = EXIT_MAX_HOLD_BARS,
    ) -> None:
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_safety = take_profit_safety
        self.trailing_pct = trailing_pct
        self.max_hold_bars = max_hold_bars

    def evaluate_exit(
        self,
        direction: int,            # +1 (Long) or -1 (Short)
        entry_price: float,
        entry_atr: float,
        current_price: float,
        highest_price: float,
        lowest_price: float,
        bars_held: int,
        predicted_runup: float = 0.01,
    ) -> ExitDecision:
        """
        Evaluates whether an open position should be closed on the current bar.
        """
        if direction == 1:  # LONG POSITION
            # 1. Hard Stop-Loss
            stop_level = entry_price - (self.stop_loss_atr_mult * entry_atr)
            if current_price <= stop_level:
                return ExitDecision(True, "STOP_LOSS", current_price)

            # 2. Take-Profit from Model 3b Runup
            tp_target = entry_price * (1.0 + self.take_profit_safety * max(predicted_runup, 0.003))
            if current_price >= tp_target:
                return ExitDecision(True, "TAKE_PROFIT", current_price)

            # 3. Trailing Stop: ratchets once position moves favorable by trailing_pct
            if highest_price >= entry_price * (1.0 + self.trailing_pct):
                trail_stop = highest_price * (1.0 - self.trailing_pct)
                if current_price <= trail_stop:
                    return ExitDecision(True, "TRAILING_STOP", current_price)

            # 4. Time-based fallback
            if bars_held >= self.max_hold_bars:
                return ExitDecision(True, "TIME_EXIT", current_price)

        elif direction == -1:  # SHORT POSITION
            # 1. Hard Stop-Loss
            stop_level = entry_price + (self.stop_loss_atr_mult * entry_atr)
            if current_price >= stop_level:
                return ExitDecision(True, "STOP_LOSS", current_price)

            # 2. Take-Profit
            tp_target = entry_price * (1.0 - self.take_profit_safety * max(predicted_runup, 0.003))
            if current_price <= tp_target:
                return ExitDecision(True, "TAKE_PROFIT", current_price)

            # 3. Trailing Stop
            if lowest_price <= entry_price * (1.0 - self.trailing_pct):
                trail_stop = lowest_price * (1.0 + self.trailing_pct)
                if current_price >= trail_stop:
                    return ExitDecision(True, "TRAILING_STOP", current_price)

            # 4. Time-based fallback
            if bars_held >= self.max_hold_bars:
                return ExitDecision(True, "TIME_EXIT", current_price)

        return ExitDecision(False, "HOLD", current_price)


class PortfolioState:
    """
    Unified state persistence for ExposureManager and CircuitBreaker.
    Both read from and write to a single artifacts/portfolio_state.json file
    to prevent state drift between related concerns.
    """
    STATE_FILE = os.path.join(ARTIFACTS_DIR, "portfolio_state.json")
    
    def __init__(self):
        self.open_positions = {}  # {symbol: {direction, entry_price, timestamp, bars_held}}
        self.daily_pnl = 0.0
        self.daily_date = None
        self.starting_capital = 0.0
    
    def save(self):
        state = {
            "open_positions": self.open_positions,
            "daily_pnl": self.daily_pnl,
            "daily_date": str(self.daily_date) if self.daily_date else None,
            "starting_capital": self.starting_capital
        }
        os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
        with open(self.STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self):
        if not os.path.exists(self.STATE_FILE):
            return
        try:
            with open(self.STATE_FILE, 'r') as f:
                state = json.load(f)
            self.open_positions = state.get("open_positions", {})
            self.daily_pnl = state.get("daily_pnl", 0.0)
            self.daily_date = state.get("daily_date")
            self.starting_capital = state.get("starting_capital", 0.0)
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupted state file — start fresh


class ExposureManager:
    """
    Prevents over-concentration in correlated positions.
    
    If HDFC Bank, ICICI Bank, and Kotak Bank all signal LONG simultaneously,
    this manager ensures we don't take all three positions — because if banking
    crashes, all three would lose together, amplifying the damage.
    
    It uses the correlation matrix peer_map to detect when multiple open
    positions belong to the same cluster of highly correlated stocks.
    """
    
    def __init__(self, max_correlated_positions: int = 2, correlation_threshold: float = 0.7, portfolio_state: Optional['PortfolioState'] = None):
        """
        Args:
            max_correlated_positions: Max number of positions allowed in the same
                                      correlation cluster.
            correlation_threshold: Minimum correlation to be considered "same cluster".
        """
        self.max_correlated_positions = max_correlated_positions
        self.correlation_threshold = correlation_threshold
        self.peer_map: dict = {}
        
        self.portfolio_state = portfolio_state
        if self.portfolio_state is not None:
            self.open_positions = self.portfolio_state.open_positions
        else:
            self.open_positions = {}
        
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
    
    def register_position(self, symbol: str, direction: int, entry_price: float, timestamp: str):
        """Records that a position has been opened in the given symbol."""
        self.open_positions[symbol] = {
            "direction": direction,
            "entry_price": entry_price,
            "timestamp": timestamp,
            "bars_held": 0
        }
    
    def close_position(self, symbol: str):
        """Records that a position has been closed."""
        if symbol in self.open_positions:
            del self.open_positions[symbol]


class CircuitBreaker:
    """
    Daily max-loss circuit breaker. If the day's P&L drops below a threshold,
    ALL trading is halted for the remainder of the day.
    
    This prevents catastrophic spiral losses where the model keeps trading
    into a crashing market, each trade losing more than the last.
    """
    
    def __init__(self, max_daily_loss_pct: float = -0.02, portfolio_state: Optional['PortfolioState'] = None):
        """
        Args:
            max_daily_loss_pct: Maximum daily loss as fraction of capital (e.g., -0.02 = -2%).
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.portfolio_state = portfolio_state
        
        if self.portfolio_state is not None:
            self.daily_pnl = self.portfolio_state.daily_pnl
            self.starting_capital = self.portfolio_state.starting_capital
            self.current_date = self.portfolio_state.daily_date
        else:
            self.daily_pnl = 0.0
            self.starting_capital = 0.0
            self.current_date = None
            
        self.is_tripped = False
        if self.starting_capital > 0 and (self.daily_pnl / self.starting_capital) <= self.max_daily_loss_pct:
            self.is_tripped = True
    
    def reset_day(self, capital: float, date):
        """Resets the circuit breaker at the start of a new trading day."""
        self.daily_pnl = 0.0
        self.starting_capital = capital
        self.is_tripped = False
        self.current_date = date
        if self.portfolio_state is not None:
            self.portfolio_state.daily_pnl = 0.0
            self.portfolio_state.starting_capital = capital
            self.portfolio_state.daily_date = date
    
    def record_trade(self, pnl: float) -> bool:
        """
        Records a trade's P&L and checks if the circuit breaker should trip.
        
        Args:
            pnl: Profit/loss of the trade in currency units.
        
        Returns:
            True if trading can continue, False if circuit breaker has tripped.
        """
        self.daily_pnl += pnl
        if self.portfolio_state is not None:
            self.portfolio_state.daily_pnl = self.daily_pnl
            
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
