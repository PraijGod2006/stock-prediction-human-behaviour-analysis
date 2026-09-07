# Multi-Model XGBoost Trading Pipeline

A production-grade quantitative trading system built on top of XGBoost, trained incrementally across 100 NIFTY50 companies on 1-minute historical data (~4.6 GB).

## Architecture

```
trading_pipeline/
├── preprocess_to_parquet.py    # Step 1: CSV → Parquet + validation
├── train_master.py             # Step 2: Incremental training across all companies
├── inference.py                # Step 3: Generate trading signals
│
├── feature_engine/             # Technical indicator & z-score calculations
│   ├── zscore.py               # 10+ z-score variants (return, volume, volatility, etc.)
│   ├── indicators.py           # RSI, VWAP, exhaustion features
│   ├── aggregator.py           # 1-min → 5-min resampling
│   └── pipeline.py             # Feature orchestrator
│
├── correlation/                # Cross-asset signal injection
│   ├── matrix_builder.py       # 100×100 correlation matrix (3 types)
│   └── cross_asset.py          # Peer momentum feature injection
│
├── models/                     # 3-model XGBoost ensemble
│   ├── model_1_direction.py    # Up/Down classifier (5-min)
│   ├── model_2_price.py        # Exact price predictor (regression)
│   └── model_3_exhaustion.py   # Mean-reversion detector (1-min)
│
├── validation/                 # Purged walk-forward CV + Optuna
│   └── purged_cv.py
│
├── monitoring/                 # Drift detection + metrics logging
│   ├── drift_detector.py       # PSI-based feature drift
│   └── metrics_logger.py       # JSON training log
│
├── backtest/                   # Realistic trading simulation
│   ├── engine.py               # Transaction costs, fill probability
│   └── risk_manager.py         # ATR sizing, exposure caps, circuit breaker
│
└── artifacts/                  # Auto-generated at runtime
    ├── model_*.json            # Saved model weights
    ├── correlation_matrix.parquet
    ├── peer_map.json
    ├── replay_buffer.parquet
    ├── training_log.json
    ├── drift_log.json
    └── signals.csv
```

## How To Run

### Step 0: Install Dependencies
```bash
.\shitimon\Scripts\pip.exe install polars pandera optuna pyarrow
```

### Step 1: Preprocess Data (CSV → Parquet)
```bash
.\shitimon\Scripts\python.exe trading_pipeline\preprocess_to_parquet.py
```
This validates and converts all 100 NIFTY50 CSVs into compressed Parquet files.

### Step 2: Build Correlation Matrix (Optional but Recommended)
```bash
.\shitimon\Scripts\python.exe trading_pipeline\correlation\matrix_builder.py
```
Computes the 100×100 stock correlation matrix and extracts peer signals.

### Step 3: Train All 3 Models
```bash
.\shitimon\Scripts\python.exe trading_pipeline\train_master.py
```
Iterates through all 100 companies, training incrementally with replay buffer.

### Step 4: Run Inference
```bash
.\shitimon\Scripts\python.exe trading_pipeline\inference.py
```
Generates trading signals by combining all 3 model outputs.

## The 3 Models

| Model | Type | Input | Target | Metric |
|-------|------|-------|--------|--------|
| Model 1 | XGBClassifier | 5-min features + cross-asset | Next 5-min direction (Up/Down) | >55% accuracy |
| Model 2 | XGBRegressor | 5-min features | Price move of first 1-min bar in next window | RMSE |
| Model 3 | XGBRegressor | 1-min exhaustion features | Max drawdown in next 10 minutes | RMSE |

## Key Design Decisions

- **No Bid/Ask Data**: NIFTY CSVs only have OHLCV. Mid-price approximated as `(open + close) / 2`.
- **Leakage Prevention**: All rolling features use `shift(1)`. VWAP/RSI reset at session boundaries.
- **Purged Walk-Forward CV**: No random splits. Time-ordered folds with purge + embargo gaps.
- **Replay Buffer**: 5% sample from each company retained to prevent catastrophic forgetting.
- **Realistic Targets**: >55% directional accuracy with positive expectancy after costs (not 95%).
- **GPU Acceleration**: All XGBoost models use `device='cuda'` for RTX 3050.

## Risk Controls

- **Kill Switch**: Create `artifacts/KILL_SWITCH` file to halt all signals immediately.
- **Circuit Breaker**: Auto-stops trading if daily loss exceeds -2%.
- **Exposure Cap**: Max 2 positions in correlated stocks simultaneously.
- **ATR Position Sizing**: Volatile stocks get smaller positions automatically.
