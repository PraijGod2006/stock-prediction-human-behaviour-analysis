"""
========================================================================================
BACKTEST ENGINE: Realistic Trading Simulation with Transaction Costs
========================================================================================
WHY BACKTEST?
A model can have 99% accuracy but still LOSE money if it trades at the wrong times,
pays too much in fees, or gets killed by slippage. Backtesting simulates how the
model would have performed in a real trading environment with real-world friction.

WHAT MAKES THIS REALISTIC?
1. Brokerage fees: 0.03% per trade (typical Indian discount broker like Zerodha).
2. STT (Securities Transaction Tax): 0.1% on the sell side (Indian regulation).
3. Bid-Ask Slippage: The difference between the price you see and the price you get.
4. Limit Order Fill Probability: If the model places a limit order at a specific
   price, we DON'T assume it always gets filled. We check if the market actually
   traded at that price during the next bar.

METRICS REPORTED:
- Equity Curve: Your portfolio value over time.
- Max Drawdown: The worst peak-to-trough decline (how much you'd lose in the worst case).
- Sharpe Ratio: Risk-adjusted return (> 1.0 is good, > 2.0 is excellent).
- Sortino Ratio: Like Sharpe but only penalizes downside volatility.
- Win Rate: Percentage of profitable trades.
- Profit Factor: Gross profits / Gross losses (> 1.0 means profitable overall).
========================================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime


class TransactionCostModel:
    """
    Models the real-world costs of executing trades in the Indian stock market.
    
    Every time you buy or sell, you lose money to:
    1. Brokerage: The fee your broker charges (e.g., Zerodha charges 0.03% or Rs 20, whichever is lower).
    2. STT: Securities Transaction Tax levied by the Indian government (0.1% on sell side for intraday).
    3. Slippage: The market moves between when you decide to trade and when the order actually fills.
       This is modeled as a fixed percentage of the trade value.
    """
    
    def __init__(
        self,
        brokerage_pct: float = 0.0003,     # 0.03% per trade
        stt_pct: float = 0.001,             # 0.1% on sell side
        slippage_pct: float = 0.0005        # 0.05% estimated slippage
    ):
        self.brokerage_pct = brokerage_pct
        self.stt_pct = stt_pct
        self.slippage_pct = slippage_pct
    
    def compute_cost(self, trade_value: float, is_sell: bool = False) -> float:
        """
        Computes the total transaction cost for a single trade.
        
        Args:
            trade_value: The absolute value of the trade in currency units.
            is_sell: True if this is a sell order (STT applies only on sells in intraday).
        
        Returns:
            Total cost in currency units.
        """
        cost = trade_value * self.brokerage_pct   # Brokerage
        cost += trade_value * self.slippage_pct    # Slippage
        if is_sell:
            cost += trade_value * self.stt_pct     # STT (sell side only)
        return cost


def estimate_limit_fill_probability(
    entry_price: float,
    bar_low: float,
    bar_high: float
) -> float:
    """
    Estimates whether a limit order would have been filled during a given bar.
    
    A limit BUY order fills only if the market price drops to or below the limit price.
    We check if the bar's low was at or below the entry price.
    
    If filled, we estimate a partial fill probability based on how deep into the 
    bar's range the limit price sits (deeper = more likely to fill fully).
    
    Args:
        entry_price: The limit order price.
        bar_low: The lowest price during the bar.
        bar_high: The highest price during the bar.
    
    Returns:
        Fill probability between 0.0 (no fill) and 1.0 (guaranteed fill).
    """
    if bar_high == bar_low:
        # Zero-range bar: fill only if price matches exactly
        return 1.0 if bar_low <= entry_price else 0.0
    
    if entry_price < bar_low:
        # Price never dropped to our limit — no fill
        return 0.0
    
    if entry_price >= bar_high:
        # Price was below our limit the entire bar — guaranteed fill
        return 1.0
    
    # Partial probability: how deep into the range did price go?
    # Deeper = higher chance the order book had volume there
    fill_depth = (entry_price - bar_low) / (bar_high - bar_low)
    return min(1.0, fill_depth * 0.8)  # Cap at 80% for conservative estimate


class BacktestEngine:
    """
    Simulates trading based on the 3-model ensemble signals.
    
    Takes the signals DataFrame (from inference.py) and simulates execution
    with realistic transaction costs and fill probabilities.
    
    Usage:
        engine = BacktestEngine(initial_capital=100000)
        results = engine.run(signals_df, prices_df)
        engine.print_report(results)
    """
    
    def __init__(self, initial_capital: float = 100000.0, position_size_pct: float = 0.02):
        """
        Args:
            initial_capital: Starting portfolio value in currency units.
            position_size_pct: Fraction of capital to risk per trade (2% default).
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.cost_model = TransactionCostModel()
    
    def run(self, signals: pd.DataFrame, prices: pd.DataFrame) -> dict:
        """
        Runs the backtest simulation.
        
        Args:
            signals: DataFrame with columns [date, direction, entry_price, predicted_price].
                     direction: 1 = buy, -1 = sell/short.
            prices: DataFrame with columns [date, open, high, low, close, volume].
        
        Returns:
            dict with equity_curve, trades, and summary metrics.
        """
        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        position = 0   # Current shares held
        entry_price = 0
        
        # Normalize date column types before merging
        prices = prices.copy()
        signals = signals.copy()
        if 'date' in prices.columns:
            prices['date'] = pd.to_datetime(prices['date'])
        if 'date' in signals.columns:
            signals['date'] = pd.to_datetime(signals['date'])
            
        # Merge signals with prices on date
        merged = prices.merge(signals, on='date', how='left', suffixes=('', '_signal'))
        
        for i in range(len(merged)):
            row = merged.iloc[i]
            
            # Check for signal
            if pd.notna(row.get('direction')):
                direction = int(row['direction'])
                target_entry = row.get('entry_price', row['close'])
                
                # Estimate fill probability
                fill_prob = estimate_limit_fill_probability(
                    target_entry, row['low'], row['high']
                )
                
                # Random fill check (simulate market uncertainty)
                if np.random.random() < fill_prob and position == 0:
                    # Calculate position size
                    trade_value = capital * self.position_size_pct
                    shares = trade_value / target_entry
                    
                    # Deduct transaction costs
                    cost = self.cost_model.compute_cost(trade_value, is_sell=False)
                    capital -= cost
                    
                    position = shares * direction
                    entry_price = target_entry
                    
                    trades.append({
                        'entry_date': row['date'],
                        'entry_price': target_entry,
                        'direction': direction,
                        'shares': abs(shares),
                        'entry_cost': cost
                    })
            
            # Exit logic: close position after 5 bars (simple time-based exit)
            if position != 0 and len(trades) > 0:
                trade = trades[-1]
                if 'exit_date' not in trade:
                    bars_held = i - merged.index.get_loc(
                        merged[merged['date'] == trade['entry_date']].index[0]
                    ) if trade['entry_date'] in merged['date'].values else 0
                    
                    if bars_held >= 5:
                        exit_price = row['close']
                        pnl = (exit_price - entry_price) * position
                        exit_cost = self.cost_model.compute_cost(
                            abs(position) * exit_price, is_sell=True
                        )
                        capital += pnl - exit_cost
                        
                        trade['exit_date'] = row['date']
                        trade['exit_price'] = exit_price
                        trade['pnl'] = pnl - exit_cost - trade['entry_cost']
                        trade['exit_cost'] = exit_cost
                        
                        position = 0
            
            equity_curve.append(capital)
        
        # Calculate summary metrics
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[np.isfinite(returns)]
        
        completed_trades = [t for t in trades if 'pnl' in t]
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] <= 0]
        
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1e-8
        
        # Max Drawdown: worst peak-to-trough decline
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Sharpe Ratio: mean(returns) / std(returns) * sqrt(252 * 375)
        # 375 = trading minutes per day, 252 = trading days per year
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252 * 375) if len(returns) > 0 else 0
        
        # Sortino Ratio: like Sharpe but only uses downside deviation
        downside = returns[returns < 0]
        sortino = (np.mean(returns) / (np.std(downside) + 1e-8)) * np.sqrt(252 * 375) if len(downside) > 0 else 0
        
        summary = {
            'initial_capital': self.initial_capital,
            'final_capital': round(capital, 2),
            'total_return_pct': round((capital - self.initial_capital) / self.initial_capital * 100, 2),
            'total_trades': len(completed_trades),
            'win_rate': round(len(winning_trades) / max(len(completed_trades), 1) * 100, 2),
            'profit_factor': round(gross_profit / gross_loss, 2),
            'max_drawdown_pct': round(max_drawdown * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
        }
        
        return {
            'summary': summary,
            'equity_curve': equity_curve,
            'trades': completed_trades
        }
    
    def print_report(self, results: dict):
        """Prints a clean summary of the backtest results."""
        s = results['summary']
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Initial Capital:   {s['initial_capital']:>12,.2f}")
        print(f"Final Capital:     {s['final_capital']:>12,.2f}")
        print(f"Total Return:      {s['total_return_pct']:>11.2f}%")
        print(f"Total Trades:      {s['total_trades']:>12d}")
        print(f"Win Rate:          {s['win_rate']:>11.2f}%")
        print(f"Profit Factor:     {s['profit_factor']:>12.2f}")
        print(f"Max Drawdown:      {s['max_drawdown_pct']:>11.2f}%")
        print(f"Sharpe Ratio:      {s['sharpe_ratio']:>12.2f}")
        print(f"Sortino Ratio:     {s['sortino_ratio']:>12.2f}")
        print("=" * 60)
