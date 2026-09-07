"""
========================================================================================
BACKTEST RUNNER -- Validates Trained Models Against Historical Data
========================================================================================
This script loads the trained models, generates signals on held-out data,
and runs the BacktestEngine to produce realistic performance metrics.

AUDIT FIX: The BacktestEngine existed but was never called. This script
connects the inference pipeline to the backtest engine.

Usage:
    .\\shitimon\\Scripts\\python.exe trading_pipeline\\run_backtest.py
========================================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import load_models, generate_signals
from backtest.engine import BacktestEngine
from feature_engine.pipeline import build_features_1min, build_features_5min

PARQUET_DIR = r"d:\CODE\rajasthani\DATA\parquet"
ARTIFACTS_DIR = r"d:\CODE\rajasthani\trading_pipeline\artifacts"


def run_backtest_on_symbol(symbol: str, model_1, model_2, model_3, engine: BacktestEngine):
    """
    Runs inference + backtest on a single symbol.
    Uses the last 20% of data as the test set (simulating out-of-sample).
    """
    parquet_path = os.path.join(PARQUET_DIR, f"{symbol}.parquet")
    if not os.path.exists(parquet_path):
        print(f"  {symbol}: Parquet file not found, skipping.")
        return None

    # Load full data
    df_raw = pl.read_parquet(parquet_path, memory_map=False).to_pandas()

    # Use last 20% as test set (out-of-sample)
    split = int(len(df_raw) * 0.8)
    df_test = df_raw.iloc[split:].copy()

    if len(df_test) < 500:
        print(f"  {symbol}: Not enough test data ({len(df_test)} rows), skipping.")
        return None

    # Generate signals
    try:
        signals = generate_signals(df_test, symbol, model_1, model_2, model_3)
    except Exception as e:
        print(f"  {symbol}: Signal generation failed: {e}")
        return None

    if len(signals) == 0 or signals['signal_type'].eq('HOLD').all():
        print(f"  {symbol}: No actionable signals generated.")
        return None

    actionable = signals[signals['signal_type'] != 'HOLD']
    print(f"  {symbol}: {len(actionable)} actionable signals out of {len(signals)} total")

    # Prepare prices DataFrame for backtest
    if 'date' not in df_test.columns and df_test.index.name == 'date':
        df_test = df_test.reset_index()

    prices = df_test[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Run backtest
    try:
        results = engine.run(actionable, prices)
        return results
    except Exception as e:
        print(f"  {symbol}: Backtest failed: {e}")
        return None


def main():
    """
    Runs backtest across a sample of stocks and prints aggregate results.
    """
    print("=" * 70)
    print("BACKTEST: Loading trained models...")
    print("=" * 70)

    try:
        model_1, model_2, model_3 = load_models()
    except Exception as e:
        print(f"ERROR loading models: {e}")
        print("Run train_master.py first!")
        return

    engine = BacktestEngine(initial_capital=100000.0, position_size_pct=0.02)

    # Test on a diverse sample of stocks
    test_symbols = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "SBIN", "WIPRO", "TATAMOTORS", "LT", "BAJFINANCE"
    ]

    all_results = []
    for symbol in test_symbols:
        print(f"\n--- Backtesting {symbol} ---")
        result = run_backtest_on_symbol(symbol, model_1, model_2, model_3, engine)
        if result is not None:
            result['summary']['symbol'] = symbol
            all_results.append(result)
            engine.print_report(result)

    if not all_results:
        print("\nNo backtest results produced. Check model outputs.")
        return

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE BACKTEST RESULTS")
    print("=" * 70)

    total_trades = sum(r['summary']['total_trades'] for r in all_results)
    avg_return = np.mean([r['summary']['total_return_pct'] for r in all_results])
    avg_sharpe = np.mean([r['summary']['sharpe_ratio'] for r in all_results])
    avg_sortino = np.mean([r['summary']['sortino_ratio'] for r in all_results])
    avg_win_rate = np.mean([r['summary']['win_rate'] for r in all_results])
    avg_drawdown = np.mean([r['summary']['max_drawdown_pct'] for r in all_results])
    avg_pf = np.mean([r['summary']['profit_factor'] for r in all_results])

    print(f"Stocks tested:       {len(all_results)}")
    print(f"Total trades:        {total_trades}")
    print(f"Avg Return:          {avg_return:.2f}%")
    print(f"Avg Win Rate:        {avg_win_rate:.2f}%")
    print(f"Avg Profit Factor:   {avg_pf:.2f}")
    print(f"Avg Max Drawdown:    {avg_drawdown:.2f}%")
    print(f"Avg Sharpe Ratio:    {avg_sharpe:.2f}")
    print(f"Avg Sortino Ratio:   {avg_sortino:.2f}")
    print("=" * 70)

    # Save results to JSON
    results_path = os.path.join(ARTIFACTS_DIR, "backtest_results.json")
    summaries = [r['summary'] for r in all_results]
    with open(results_path, 'w') as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
