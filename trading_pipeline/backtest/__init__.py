# Backtest module
from .engine import BacktestEngine, TransactionCostModel
from .risk_manager import (
    check_kill_switch,
    calculate_atr,
    calculate_position_size,
    ExposureManager,
    CircuitBreaker
)
