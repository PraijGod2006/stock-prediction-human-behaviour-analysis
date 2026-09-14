"""
========================================================================================
TRADING PIPELINE: Central Configuration Constants
========================================================================================
Single source of truth for all cost, barrier, sizing, and threshold parameters used
across the pipeline. Import from here instead of hardcoding constants in individual
modules.

Derived from real TransactionCostModel in backtest/engine.py:
  Brokerage: 0.03% × 2 sides = 0.06%
  Slippage:  0.05% × 2 sides = 0.10%
  STT:       0.10% × 1 side   = 0.10%
  Total round-trip             = 0.26%
========================================================================================
"""

# ---------------------------------------------------------------------------
# Transaction Costs
# ---------------------------------------------------------------------------
ROUND_TRIP_COST_PCT = 0.0026  # 0.26% total round-trip cost

# ---------------------------------------------------------------------------
# Triple-Barrier Labeling (Fix Group 1)
# ---------------------------------------------------------------------------
TRIPLE_BARRIER_K_UPPER = 1.5      # Upper barrier = TP * (1 + max(k * ATR_pct, MIN_BARRIER_DISTANCE_PCT))
TRIPLE_BARRIER_K_LOWER = 1.5      # Lower barrier = TP * (1 - max(k * ATR_pct, MIN_BARRIER_DISTANCE_PCT))
TRIPLE_BARRIER_MAX_HOLD_1MIN = 25 # 25 one-minute bars = 5 five-minute bars (matches trade holding horizon)

# Minimum barrier distance must be >= 2x round-trip cost = 0.52%
MIN_BARRIER_DISTANCE_PCT = 2.0 * ROUND_TRIP_COST_PCT  # 0.0052 = 0.52%

# ---------------------------------------------------------------------------
# EV Gating (Fix Group 6)
# ---------------------------------------------------------------------------
MIN_EDGE_BUFFER_PCT = 0.0002  # 0.02% above breakeven for EV gate

# ---------------------------------------------------------------------------
# Position Sizing (Fix Group 9)
# ---------------------------------------------------------------------------
KELLY_FRACTION = 0.25            # Quarter-Kelly default
POSITION_SIZING_METHOD = "atr"   # Options: "atr", "kelly", "hybrid"

# ---------------------------------------------------------------------------
# ExitManager (Fix Group 8)
# ---------------------------------------------------------------------------
EXIT_STOP_LOSS_ATR_MULT = 1.5    # Hard stop-loss at 1.5x ATR from entry
EXIT_TAKE_PROFIT_SAFETY = 0.7    # Scale predicted runup by 0.7 for TP target
EXIT_TRAILING_PCT = 0.005        # 0.5% trailing stop ratchet
EXIT_MAX_HOLD_BARS = 5           # Time-based fallback exit (bars)

# ---------------------------------------------------------------------------
# Kelly b formula (corrected per user review)
# Uses ExitManager's actual TP/SL numbers, not raw model predictions:
#   b = (EXIT_TAKE_PROFIT_SAFETY * predicted_runup) / (EXIT_STOP_LOSS_ATR_MULT * ATR_pct)
# ---------------------------------------------------------------------------
"""
Config module for the trading pipeline. Import constants from here.
"""

