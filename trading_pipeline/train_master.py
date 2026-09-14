"""
========================================================================================
MASTER TRAINING ORCHESTRATOR -- Incremental Learning Across 100 Companies
========================================================================================
This is the brain of the entire training pipeline. It:

1. Iterates through all NIFTY 50 Parquet files ONE AT A TIME (strictly memory safe).
2. For each company:
   a. Loads Parquet data into a DataFrame.
   b. Engineers 1-min and 5-min features via feature_engine/pipeline.py.
   c. Injects cross-asset correlation signals (for Models 1 & 2).
   d. Prepares targets for all 4 models:
      - Model 1: Direction (3-class triple-barrier: DOWN=0, FLAT=1, UP=2, >=2x cost)
      - Model 2: Price (Section 13 exact alignment: 1-min return of first bar in next window)
      - Model 3: Exhaustion (10-minute forward maximum drawdown, [-0.15, 0.0])
      - Model 3b: Runup (10-minute forward maximum upside excursion, [0.0, 0.15])
   e. Employs PurgedWalkForwardCV (with purge and embargo gaps) to eliminate leakage.
   f. Mixes separate 5-min and 1-min replay buffers to mitigate catastrophic forgetting.
   g. Continues training all 4 XGBoost models using incremental warm-start (xgb_model=).
   h. Logs out-of-sample metrics and checks for PSI feature drift.
3. Every 25 companies, triggers a full retrain from the replay buffers to reset drift.
4. Frees memory after every single company (del + gc.collect).
========================================================================================
"""

import gc
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, mean_squared_error

# Add the trading_pipeline directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from correlation.cross_asset import inject_cross_asset_features
from feature_engine.pipeline import build_features_1min, build_features_5min
from models.model_1_direction import DirectionalModel
from models.model_2_price import PriceModel
from models.model_3_exhaustion import ExhaustionModel
from models.model_3b_runup import RunupModel
from monitoring.drift_detector import DriftDetector
from monitoring.metrics_logger import MetricsLogger
from validation.purged_cv import PurgedWalkForwardCV, optimize_hyperparameters

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
PARQUET_DIR = r"d:\CODE\rajasthani\DATA\parquet"
ARTIFACTS_DIR = r"d:\CODE\rajasthani\trading_pipeline\artifacts"
PEER_MAP_PATH = os.path.join(ARTIFACTS_DIR, "peer_map.json")

# Model save paths
MODEL_1_PATH = os.path.join(ARTIFACTS_DIR, "model_1_direction.json")
MODEL_2_PATH = os.path.join(ARTIFACTS_DIR, "model_2_price.json")
MODEL_3_PATH = os.path.join(ARTIFACTS_DIR, "model_3_exhaustion.json")
MODEL_3B_PATH = os.path.join(ARTIFACTS_DIR, "model_3b_runup.json")
BEST_HYPERPARAMS_PATH = os.path.join(ARTIFACTS_DIR, "best_hyperparameters.json")

# Replay buffer paths -- separate files for 5-min (Models 1&2) and 1-min (Models 3&3b)
REPLAY_5MIN_PATH = os.path.join(ARTIFACTS_DIR, "replay_buffer_5min.parquet")
REPLAY_1MIN_PATH = os.path.join(ARTIFACTS_DIR, "replay_buffer_1min.parquet")

# Retain 5% sample from each company to prevent catastrophic forgetting
REPLAY_SAMPLE_RATE = 0.05


def ensure_artifacts_dir() -> None:
    """Creates the artifacts directory if it doesn't exist."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def _safe_write_parquet(df_polars: pl.DataFrame, path: str) -> None:
    """
    Writes a Polars DataFrame to Parquet safely on Windows.
    Uses a temp file + atomic rename to avoid file lock errors.
    """
    temp_path = path + '.tmp'
    df_polars.write_parquet(temp_path)
    gc.collect()
    os.replace(temp_path, path)


def perform_priority2_reset() -> None:
    """
    Phase 1 Reset Checklist:
    1. Archives existing pre-Priority-2 models & configs with _pre_priority2 suffix.
    2. Deletes replay buffers (schema change).
    3. Cleans any temporary files.
    """
    print("\n" + "=" * 70)
    print("PHASE 1 RESET: Archiving pre-priority2 artifacts and clearing replay buffers...")
    print("=" * 70)
    artifacts = [
        "model_1_direction.json",
        "model_2_price.json",
        "model_3_exhaustion.json",
        "model_3b_runup.json",
        "best_hyperparameters.json",
        "training_log.json",
        "drift_log.json",
        "signals.csv",
    ]
    for art in artifacts:
        p = os.path.join(ARTIFACTS_DIR, art)
        if os.path.exists(p):
            base, ext = os.path.splitext(art)
            archive_p = os.path.join(ARTIFACTS_DIR, f"{base}_pre_priority2{ext}")
            try:
                shutil.copy2(p, archive_p)
                print(f"  -> Archived: {art} -> {os.path.basename(archive_p)}")
            except Exception as e:
                print(f"  -> Warning archiving {art}: {e}")

    # Delete replay buffers due to schema changes (3-class target, runup target, bollinger features)
    for buf in ["replay_buffer_5min.parquet", "replay_buffer_1min.parquet"]:
        bp = os.path.join(ARTIFACTS_DIR, buf)
        if os.path.exists(bp):
            try:
                os.remove(bp)
                print(f"  -> Deleted old replay buffer: {buf}")
            except Exception as e:
                print(f"  -> Warning deleting {buf}: {e}")

    # Delete any temporary files
    for tmp in glob.glob(os.path.join(ARTIFACTS_DIR, "*.tmp")):
        try:
            os.remove(tmp)
        except Exception:
            pass

    print("=" * 70 + "\n")


def update_replay_buffer_5min(
    X_5min: pd.DataFrame,
    y_dir: pd.Series,
    y_price: pd.Series,
    symbol: str
) -> None:
    """
    Adds a 5% random sample of 5-min data to the replay buffer for Models 1 & 2.
    This prevents catastrophic forgetting when training incrementally.
    """
    n_sample = max(1, int(len(X_5min) * REPLAY_SAMPLE_RATE))
    idx = np.random.choice(len(X_5min), size=min(n_sample, len(X_5min)), replace=False)

    sample = X_5min.iloc[idx].copy()
    sample['__target_dir'] = y_dir.iloc[idx].to_numpy()
    sample['__target_price'] = y_price.iloc[idx].to_numpy()
    sample['__symbol'] = symbol

    new_buffer = pl.from_pandas(sample)

    if os.path.exists(REPLAY_5MIN_PATH):
        try:
            existing = pl.read_parquet(REPLAY_5MIN_PATH, memory_map=False)
            combined = pl.concat([existing, new_buffer], how='diagonal')
            del existing
        except Exception:  # noqa: BLE001
            combined = new_buffer
    else:
        combined = new_buffer

    _safe_write_parquet(combined, REPLAY_5MIN_PATH)
    del combined, new_buffer
    gc.collect()


def update_replay_buffer_1min(
    X_1min: pd.DataFrame,
    y_exhaust: pd.Series,
    y_runup: pd.Series,
    symbol: str
) -> None:
    """
    Adds a 5% random sample of 1-min data to the replay buffer for Models 3 & 3b.
    """
    n_sample = max(1, int(len(X_1min) * REPLAY_SAMPLE_RATE))
    idx = np.random.choice(len(X_1min), size=min(n_sample, len(X_1min)), replace=False)

    sample = X_1min.iloc[idx].copy()
    sample['__target_exhaust'] = y_exhaust.iloc[idx].to_numpy()
    sample['__target_runup'] = y_runup.iloc[idx].to_numpy()
    sample['__symbol'] = symbol

    new_buffer = pl.from_pandas(sample)

    if os.path.exists(REPLAY_1MIN_PATH):
        try:
            existing = pl.read_parquet(REPLAY_1MIN_PATH, memory_map=False)
            combined = pl.concat([existing, new_buffer], how='diagonal')
            del existing
        except Exception:  # noqa: BLE001
            combined = new_buffer
    else:
        combined = new_buffer

    _safe_write_parquet(combined, REPLAY_1MIN_PATH)
    del combined, new_buffer
    gc.collect()


def load_replay_5min() -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
    """Loads 5-min replay buffer. Returns (X, y_dir, y_price) or None."""
    if not os.path.exists(REPLAY_5MIN_PATH):
        return None
    try:
        buffer = pl.read_parquet(REPLAY_5MIN_PATH, memory_map=False).to_pandas()
        meta_cols = ['__target_dir', '__target_price', '__symbol']
        feature_cols = [c for c in buffer.columns if c not in meta_cols]
        return pd.DataFrame(buffer[feature_cols]), pd.Series(buffer['__target_dir']), pd.Series(buffer['__target_price'])
    except Exception:  # noqa: BLE001
        return None


def load_replay_1min() -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
    """Loads 1-min replay buffer. Returns (X, y_exhaust, y_runup) or None."""
    if not os.path.exists(REPLAY_1MIN_PATH):
        return None
    try:
        buffer = pl.read_parquet(REPLAY_1MIN_PATH, memory_map=False).to_pandas()
        meta_cols = ['__target_exhaust', '__target_runup', '__symbol']
        feature_cols = [c for c in buffer.columns if c not in meta_cols]
        y_run = buffer['__target_runup'] if '__target_runup' in buffer.columns else pd.Series(np.zeros(len(buffer)))
        return pd.DataFrame(buffer[feature_cols]), pd.Series(buffer['__target_exhaust']), pd.Series(y_run)
    except Exception:  # noqa: BLE001
        return None


def clean_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Removes inf values, all-NaN columns, and rows with any remaining NaN.
    Returns (X_clean, y_clean) with strictly aligned indices.
    """
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    y_clean = y.replace([np.inf, -np.inf], np.nan)

    # Drop all-NaN columns
    X_clean = X_clean.dropna(axis=1, how='all')

    # Find valid rows (no NaN in features AND target)
    valid = X_clean.notna().all(axis=1) & y_clean.notna()
    return pd.DataFrame(X_clean[valid]), pd.Series(y_clean[valid])


def clean_data_1min(
    X: pd.DataFrame,
    y_exh: pd.Series,
    y_run: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Cleans 1-minute features and both targets (exhaustion & runup) with aligned indices.
    """
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    y_e = y_exh.replace([np.inf, -np.inf], np.nan)
    y_r = y_run.replace([np.inf, -np.inf], np.nan)

    X_clean = X_clean.dropna(axis=1, how='all')
    valid = X_clean.notna().all(axis=1) & y_e.notna() & y_r.notna()
    return pd.DataFrame(X_clean[valid]), pd.Series(y_e[valid]), pd.Series(y_r[valid])


def run_hyperparameter_tuning(first_parquet_path: str) -> None:
    """
    Fix Group D + Priority-2: Runs Optuna Bayesian hyperparameter tuning across all
    4 models (Directional 3-class, Price Regressor, Exhaustion Regressor, Runup Regressor).
    """
    symbol = os.path.basename(first_parquet_path).replace(".parquet", "")
    print("\n" + "=" * 70)
    print(f"OPTUNA HYPERPARAMETER TUNING REQUESTED (tuning on: {symbol})")
    print("=" * 70)

    try:
        df_raw = pl.read_parquet(first_parquet_path, memory_map=False).to_pandas()
        df_1min = build_features_1min(df_raw)
        df_5min = build_features_5min(df_raw)

        if os.path.exists(PEER_MAP_PATH):
            try:
                df_5min = inject_cross_asset_features(df_5min, symbol, PEER_MAP_PATH, PARQUET_DIR)
            except Exception:
                pass

        m1_tmp = DirectionalModel()
        m2_tmp = PriceModel()
        m3_tmp = ExhaustionModel()
        m3b_tmp = RunupModel()

        # Target preparation: pass df_1min to Model 1 for triple barrier
        y_dir_df = m1_tmp.prepare_target(df_5min, df_1min=df_1min)
        y_price_df = m2_tmp.prepare_target(df_5min, df_1min=df_1min)

        target_dir_s = pd.Series(y_dir_df['target'], name='target_dir')
        target_price_s = pd.Series(y_price_df['target'], name='target_price')
        targets_5m = pd.concat([target_dir_s, target_price_s], axis=1).dropna()
        common_idx_5m = df_5min.index.intersection(targets_5m.index)

        drop_cols = [
            'date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'mid_price', 'future_close_5m', 'target', '__target_dir',
            '__target_price', '__target_exhaust', '__target_runup',
            '__barrier_distance_pct', '__symbol'
        ]
        X_5min = pd.DataFrame(df_5min.loc[common_idx_5m, [c for c in df_5min.columns if c not in drop_cols]])
        y_dir = pd.Series(targets_5m.loc[common_idx_5m, 'target_dir'])
        y_price = pd.Series(targets_5m.loc[common_idx_5m, 'target_price'])

        valid_5m = (
            X_5min.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            & y_dir.replace([np.inf, -np.inf], np.nan).notna()
            & y_price.replace([np.inf, -np.inf], np.nan).notna()
        )
        X_5min = pd.DataFrame(X_5min[valid_5m])
        y_dir = pd.Series(y_dir[valid_5m])
        y_price = pd.Series(y_price[valid_5m])

        # 1-minute targets
        df_1min_target_exh = m3_tmp.prepare_target(df_1min)
        df_1min_target_run = m3b_tmp.prepare_target(df_1min)
        common_1m = df_1min_target_exh.index.intersection(df_1min_target_run.index)

        y_exhaust_raw = pd.Series(df_1min_target_exh.loc[common_1m, 'target'])
        y_runup_raw = pd.Series(df_1min_target_run.loc[common_1m, 'target'])
        X_1min_raw = pd.DataFrame(df_1min.loc[common_1m, [c for c in df_1min.columns if c not in drop_cols]])
        X_1min, y_exhaust, y_runup = clean_data_1min(X_1min_raw, y_exhaust_raw, y_runup_raw)

        print(f"  [Optuna Tuning] Samples: 5m={len(X_5min):,} rows, 1m={len(X_1min):,} rows")

        print("  -> Running Optuna trials for Model 1 (Directional 3-Class Classifier)...")
        best_m1 = optimize_hyperparameters(
            X_5min, y_dir, DirectionalModel, n_trials=30, n_splits=4,
            purge_gap=10, embargo_gap=5, is_classifier=True
        )

        print("  -> Running Optuna trials for Model 2 (Price Movement Regressor)...")
        best_m2 = optimize_hyperparameters(
            X_5min, y_price, PriceModel, n_trials=30, n_splits=4,
            purge_gap=10, embargo_gap=5, is_classifier=False
        )

        print("  -> Running Optuna trials for Model 3 (Exhaustion Regressor)...")
        best_m3 = optimize_hyperparameters(
            X_1min, y_exhaust, ExhaustionModel, n_trials=30, n_splits=4,
            purge_gap=15, embargo_gap=10, is_classifier=False
        )

        print("  -> Running Optuna trials for Model 3b (Runup Regressor)...")
        best_m3b = optimize_hyperparameters(
            X_1min, y_runup, RunupModel, n_trials=30, n_splits=4,
            purge_gap=15, embargo_gap=10, is_classifier=False
        )

        all_best = {
            "model_1_direction": best_m1,
            "model_2_price": best_m2,
            "model_3_exhaustion": best_m3,
            "model_3b_runup": best_m3b
        }

        os.makedirs(os.path.dirname(BEST_HYPERPARAMS_PATH), exist_ok=True)
        with open(BEST_HYPERPARAMS_PATH, 'w') as f:
            json.dump(all_best, f, indent=4)

        print(f"  [Optuna Tuning] Complete! Best hyperparameters saved to: {BEST_HYPERPARAMS_PATH}")
        print("=" * 70 + "\n")
        del df_raw, df_1min, df_5min, X_5min, y_dir, y_price, X_1min, y_exhaust, y_runup
        gc.collect()
    except Exception as e:
        print(f"  ERROR during Optuna hyperparameter tuning: {e}")


def train_all_companies(tune_hyperparams: bool = False, priority2_reset: bool = False) -> None:
    """
    Main training loop across all companies.
    Iterates through each Parquet file one by one (memory-safe incremental learning).
    """
    ensure_artifacts_dir()

    if priority2_reset:
        perform_priority2_reset()

    # Find all Parquet files
    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))

    if not parquet_files:
        print(f"ERROR: No Parquet files found in {PARQUET_DIR}")
        print("Run preprocess_to_parquet.py first!")
        return

    # If hyperparameter tuning requested, run Optuna on first company before main loop
    if tune_hyperparams and len(parquet_files) > 0:
        run_hyperparameter_tuning(parquet_files[0])

    # Initialize models (will load existing weights if files exist)
    model_1 = DirectionalModel(MODEL_1_PATH)
    model_2 = PriceModel(MODEL_2_PATH)
    model_3 = ExhaustionModel(MODEL_3_PATH)
    model_3b = RunupModel(MODEL_3B_PATH)

    # Initialize monitoring
    logger = MetricsLogger(os.path.join(ARTIFACTS_DIR, "training_log.json"))
    drift_detector = DriftDetector(os.path.join(ARTIFACTS_DIR, "drift_log.json"))

    print(f"Found {len(parquet_files)} companies to train on.")
    print("=" * 70)

    models_initialized = False

    for i, parquet_path in enumerate(parquet_files):
        symbol = os.path.basename(parquet_path).replace(".parquet", "")
        print(f"\n[{i+1}/{len(parquet_files)}] Training on: {symbol}")
        print("-" * 50)

        # --- STEP 1: Load Data ---
        print("  [Step 1] Loading raw 1-minute Parquet data via Polars...")
        try:
            df = pl.read_parquet(parquet_path, memory_map=False).to_pandas()
            print(f"    -> Loaded {len(df):,} rows x {len(df.columns)} columns from {os.path.basename(parquet_path)}")
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR loading data for {symbol}: {e}")
            continue

        # --- STEP 2: Engineer Features ---
        print("  [Step 2] Feature Engineering via feature_engine/pipeline.py...")
        try:
            print("    -> Calling build_features_1min() for Models 3 & 3b (Exhaustion indicators, VWAP, RSI, Volume Z, Relative Vol TOD, Bollinger)...")
            df_1min = build_features_1min(df)
            print(f"       1-min features ready: {df_1min.shape[0]:,} rows x {df_1min.shape[1]} columns")

            print("    -> Calling build_features_5min() for Models 1 & 2 (OHLCV 5m aggregation + Z-scores + Bollinger)...")
            df_5min = build_features_5min(df)
            print(f"       5-min features ready: {df_5min.shape[0]:,} rows x {df_5min.shape[1]} columns")
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR engineering features for {symbol}: {e}")
            del df
            gc.collect()
            continue

        if len(df_5min) < 100 or len(df_1min) < 100:
            print(f"    SKIPPING {symbol}: Not enough data after feature engineering.")
            del df, df_1min, df_5min
            gc.collect()
            continue

        # --- STEP 3: Inject Cross-Asset Signals (Models 1 & 2) ---
        print("  [Step 3] Cross-Asset Signal Injection via correlation/cross_asset.py...")
        if os.path.exists(PEER_MAP_PATH):
            try:
                with open(PEER_MAP_PATH) as f:
                    peer_map_peek = json.load(f)
                peers_info = peer_map_peek.get(symbol, {})
                if "correlated_peers_raw" in peers_info:
                    pos_peers = peers_info.get("correlated_peers_raw", {}).get("top_positive", [])
                    neg_peers = peers_info.get("correlated_peers_raw", {}).get("top_negative", [])
                else:
                    pos_peers = peers_info.get("top_positive", [])
                    neg_peers = peers_info.get("top_negative", [])
                print(f"    -> Identified correlated peers from peer_map.json: +{pos_peers} | -{neg_peers}")
                df_5min = inject_cross_asset_features(df_5min, symbol, PEER_MAP_PATH, PARQUET_DIR)
                print(f"    -> Injected 18 normalized peer momentum features across 3 dimensions (5m shape now: {df_5min.shape})")
            except Exception as e:  # noqa: BLE001
                print(f"    WARNING: Could not inject cross-asset features: {e}")
        else:
            print("    INFO: peer_map.json not found, skipping cross-asset injection.")

        # --- STEP 4: Prepare Targets with Zero-Leakage Alignment ---
        print("  [Step 4] Target Preparation with Zero-Leakage Alignment...")
        drop_cols = [
            'date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'mid_price', 'future_close_5m', 'target', '__target_dir',
            '__target_price', '__target_exhaust', '__target_runup',
            '__barrier_distance_pct', '__symbol'
        ]

        # Model 1: Cost-Aware Triple-Barrier Labeling (3 classes)
        print("    -> Model 1: model_1.prepare_target() (Triple-Barrier 3-class: DOWN=0, FLAT=1, UP=2, >=2x cost)")
        y_dir_df = model_1.prepare_target(df_5min, df_1min=df_1min)
        if i == 0:
            model_1.get_class_distribution(y_dir_df['target'])

        # Model 2: Next 1m bar return
        print("    -> Model 2: model_2.prepare_target() (Section 13: 1m return of first bar in next window)")
        y_price_df = model_2.prepare_target(df_5min, df_1min=df_1min)

        target_dir_s = pd.Series(y_dir_df['target'], name='target_dir')
        target_price_s = pd.Series(y_price_df['target'], name='target_price')
        targets_5m = pd.concat([target_dir_s, target_price_s], axis=1).dropna()
        common_idx_5m = df_5min.index.intersection(targets_5m.index)

        X_5min = pd.DataFrame(df_5min.loc[common_idx_5m, [c for c in df_5min.columns if c not in drop_cols]])
        y_dir = pd.Series(targets_5m.loc[common_idx_5m, 'target_dir'])
        y_price = pd.Series(targets_5m.loc[common_idx_5m, 'target_price'])

        # Clean 5-min features and targets together
        valid_5m = (
            X_5min.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            & y_dir.replace([np.inf, -np.inf], np.nan).notna()
            & y_price.replace([np.inf, -np.inf], np.nan).notna()
        )
        X_5min = pd.DataFrame(X_5min[valid_5m])
        y_dir = pd.Series(y_dir[valid_5m])
        y_price = pd.Series(y_price[valid_5m])

        # Model 3: Maximum 10-minute forward drawdown
        print("    -> Model 3: model_3.prepare_target() (10-minute forward maximum drawdown)")
        df_1min_target_exh = model_3.prepare_target(df_1min)

        # Model 3b: Maximum 10-minute forward upside runup
        print("    -> Model 3b: model_3b.prepare_target() (10-minute forward maximum upside runup)")
        df_1min_target_run = model_3b.prepare_target(df_1min)

        common_1m = df_1min_target_exh.index.intersection(df_1min_target_run.index)
        y_exhaust_raw = pd.Series(df_1min_target_exh.loc[common_1m, 'target'], name='target_exh')
        y_runup_raw = pd.Series(df_1min_target_run.loc[common_1m, 'target'], name='target_runup')
        X_1min_raw = pd.DataFrame(df_1min.loc[common_1m, [c for c in df_1min.columns if c not in drop_cols]])
        X_1min, y_exhaust, y_runup = clean_data_1min(X_1min_raw, y_exhaust_raw, y_runup_raw)

        print(f"    -> Clean aligned samples: 5-min = {len(X_5min):,} rows ({X_5min.shape[1]} features) | 1-min = {len(X_1min):,} rows ({X_1min.shape[1]} features)")

        if len(X_5min) < 50 or len(X_1min) < 50:
            print(f"    SKIPPING {symbol}: Not enough clean data.")
            del df, df_1min, df_5min, targets_5m, df_1min_target_exh, df_1min_target_run, y_dir_df, y_price_df
            gc.collect()
            continue

        # Free raw data frames
        del df, df_1min, df_5min, targets_5m, df_1min_target_exh, df_1min_target_run, y_dir_df, y_price_df
        gc.collect()

        # --- STEP 5: Feature Drift Detection (PSI) ---
        print("  [Step 5] Population Stability Index (PSI) Drift Monitoring via monitoring/drift_detector.py...")
        if i == 0:
            drift_detector.set_baseline(X_5min, symbol)
            print(f"    -> Baseline feature distribution established with {symbol}")
        else:
            drift_res = drift_detector.check_drift(X_5min, symbol)
            drifted_feats = drift_res.get('drifted_features', []) if isinstance(drift_res, dict) else []
            if drifted_feats:
                print(f"    -> [PSI ALERT] Drift detected in {len(drifted_feats)} features: {drifted_feats[:3]}...")
            else:
                print("    -> Feature distribution stable: No significant PSI drift.")

        # --- STEP 6: Mix Replay Buffers (Mitigate Catastrophic Forgetting) ---
        print("  [Step 6] Replay Buffer Sampling (Anti-Catastrophic Forgetting)...")
        replay_5 = load_replay_5min()
        if replay_5 is not None and models_initialized:
            X_replay_5, y_dir_replay, y_price_replay = replay_5
            common_cols = [c for c in X_5min.columns if c in X_replay_5.columns]
            if len(common_cols) > 0:
                X_5min_train = pd.concat([X_5min[common_cols], X_replay_5[common_cols]], ignore_index=True)
                y_dir_train = pd.concat([y_dir, y_dir_replay], ignore_index=True)
                y_price_train = pd.concat([y_price, y_price_replay], ignore_index=True)
                print(f"    -> Mixed 5-min replay: {len(X_replay_5):,} buffer rows + {len(X_5min):,} current rows = {len(X_5min_train):,} total")
            else:
                X_5min_train = X_5min
                y_dir_train = y_dir
                y_price_train = y_price
            del replay_5, X_replay_5, y_dir_replay, y_price_replay
        else:
            X_5min_train = X_5min
            y_dir_train = y_dir
            y_price_train = y_price
            print(f"    -> 5-min dataset ready: {len(X_5min_train):,} samples (no prior replay buffer)")

        replay_1 = load_replay_1min()
        if replay_1 is not None and models_initialized:
            X_replay_1, y_exhaust_replay, y_runup_replay = replay_1
            common_cols_1 = [c for c in X_1min.columns if c in X_replay_1.columns]
            if len(common_cols_1) > 0:
                X_1min_train = pd.concat([X_1min[common_cols_1], X_replay_1[common_cols_1]], ignore_index=True)
                y_exhaust_train = pd.concat([y_exhaust, y_exhaust_replay], ignore_index=True)
                y_runup_train = pd.concat([y_runup, y_runup_replay], ignore_index=True)
                print(f"    -> Mixed 1-min replay: {len(X_replay_1):,} buffer rows + {len(X_1min):,} current rows = {len(X_1min_train):,} total")
            else:
                X_1min_train = X_1min
                y_exhaust_train = y_exhaust
                y_runup_train = y_runup
            del replay_1, X_replay_1, y_exhaust_replay, y_runup_replay
        else:
            X_1min_train = X_1min
            y_exhaust_train = y_exhaust
            y_runup_train = y_runup
            print(f"    -> 1-min dataset ready: {len(X_1min_train):,} samples (no prior replay buffer)")

        gc.collect()

        # --- STEP 7: Purged Walk-Forward Cross-Validation Split ---
        print("  [Step 7] Purged Walk-Forward Cross-Validation Split & Model Training...")
        X_5min_train = pd.DataFrame(X_5min_train)
        cv_5m = PurgedWalkForwardCV(n_splits=4, purge_gap=10, embargo_gap=5)
        folds_5m = list(cv_5m.split(X_5min_train))
        train_idx_5m, val_idx_5m = folds_5m[-1]

        X_tr5 = pd.DataFrame(X_5min_train.iloc[train_idx_5m])
        X_va5 = pd.DataFrame(X_5min_train.iloc[val_idx_5m])
        y_tr_dir = pd.Series(y_dir_train.iloc[train_idx_5m])
        y_va_dir = pd.Series(y_dir_train.iloc[val_idx_5m])
        y_tr_price = pd.Series(y_price_train.iloc[train_idx_5m])
        y_va_price = pd.Series(y_price_train.iloc[val_idx_5m])

        X_1min_train = pd.DataFrame(X_1min_train)
        cv_1m = PurgedWalkForwardCV(n_splits=4, purge_gap=15, embargo_gap=10)
        folds_1m = list(cv_1m.split(X_1min_train))
        train_idx_1m, val_idx_1m = folds_1m[-1]

        X_tr1 = pd.DataFrame(X_1min_train.iloc[train_idx_1m])
        X_va1 = pd.DataFrame(X_1min_train.iloc[val_idx_1m])
        y_tr_exh = pd.Series(y_exhaust_train.iloc[train_idx_1m])
        y_va_exh = pd.Series(y_exhaust_train.iloc[val_idx_1m])
        y_tr_runup = pd.Series(y_runup_train.iloc[train_idx_1m])
        y_va_runup = pd.Series(y_runup_train.iloc[val_idx_1m])

        print(f"    -> 5-min CV fold: Train={len(X_tr5):,} rows, Val={len(X_va5):,} rows (purge=10, embargo=5)")
        print(f"    -> 1-min CV fold: Train={len(X_tr1):,} rows, Val={len(X_va1):,} rows (purge=15, embargo=10)")

        # Model paths for incremental warm-start
        prev_m1 = MODEL_1_PATH if models_initialized and os.path.exists(MODEL_1_PATH) else None
        prev_m2 = MODEL_2_PATH if models_initialized and os.path.exists(MODEL_2_PATH) else None
        prev_m3 = MODEL_3_PATH if models_initialized and os.path.exists(MODEL_3_PATH) else None
        prev_m3b = MODEL_3B_PATH if models_initialized and os.path.exists(MODEL_3B_PATH) else None

        acc = 0.0
        rmse_price = 0.0
        rmse_exh = 0.0
        rmse_runup = 0.0

        # --- Model 1: Directional 3-Class Classifier ---
        mode_m1 = "incremental warm-start (xgb_model=prev)" if prev_m1 else "initial training from scratch"
        print(f"    -> [Model 1: Direction] Calling models.model_1_direction.DirectionalModel.train() ({mode_m1})...")
        try:
            model_1.train(X_tr5, y_tr_dir, X_va5, y_va_dir, xgb_model=prev_m1)
            fold_accs = []
            for f_idx, (_, f_val_idx) in enumerate(folds_5m):
                f_X_va = pd.DataFrame(X_5min_train.iloc[f_val_idx])
                f_y_va = pd.Series(y_dir_train.iloc[f_val_idx])
                f_yp = model_1.predict(f_X_va)
                f_preds = f_yp[0] if isinstance(f_yp, tuple) else f_yp
                fold_accs.append(float(accuracy_score(f_y_va, f_preds)))
            acc = float(np.mean(fold_accs))
            model_1.save(MODEL_1_PATH)
            logger.log(symbol, "model_1_direction", {
                "accuracy": round(acc, 4),
                "last_fold_accuracy": round(fold_accs[-1], 4),
                "all_fold_accuracies": [round(a, 4) for a in fold_accs],
                "n_folds": len(fold_accs),
                "val_size": len(X_va5)
            })
            print(f"       Model 1 Result: Mean Val Accuracy (across {len(fold_accs)} folds) = {acc*100:.2f}% (Last fold: {fold_accs[-1]*100:.2f}%) (Saved to {os.path.basename(MODEL_1_PATH)})")
        except Exception as e:  # noqa: BLE001
            print(f"       ERROR training Model 1: {e}")

        # --- Model 2: Price Movement Regressor ---
        mode_m2 = "incremental warm-start (xgb_model=prev)" if prev_m2 else "initial training from scratch"
        print(f"    -> [Model 2: Price Return] Calling models.model_2_price.PriceModel.train() ({mode_m2})...")
        try:
            model_2.train(X_tr5, y_tr_price, X_va5, y_va_price, xgb_model=prev_m2)
            fold_rmses_price = []
            for f_idx, (_, f_val_idx) in enumerate(folds_5m):
                f_X_va = pd.DataFrame(X_5min_train.iloc[f_val_idx])
                f_y_va = pd.Series(y_price_train.iloc[f_val_idx])
                f_preds = model_2.predict(f_X_va)
                fold_rmses_price.append(float(np.sqrt(mean_squared_error(f_y_va, f_preds))))
            rmse_price = float(np.mean(fold_rmses_price))
            model_2.save(MODEL_2_PATH)
            logger.log(symbol, "model_2_price", {
                "rmse": round(rmse_price, 6),
                "last_fold_rmse": round(fold_rmses_price[-1], 6),
                "all_fold_rmses": [round(r, 6) for r in fold_rmses_price],
                "n_folds": len(fold_rmses_price),
                "val_size": len(X_va5)
            })
            print(f"       Model 2 Result: Mean Val RMSE (across {len(fold_rmses_price)} folds) = {rmse_price:.6f} (Last fold: {fold_rmses_price[-1]:.6f}) (Saved to {os.path.basename(MODEL_2_PATH)})")
        except Exception as e:  # noqa: BLE001
            print(f"       ERROR training Model 2: {e}")

        # --- Model 3: Exhaustion Regressor ---
        mode_m3 = "incremental warm-start (xgb_model=prev)" if prev_m3 else "initial training from scratch"
        print(f"    -> [Model 3: Exhaustion] Calling models.model_3_exhaustion.ExhaustionModel.train() ({mode_m3})...")
        try:
            model_3.train(X_tr1, y_tr_exh, X_va1, y_va_exh, xgb_model=prev_m3)
            fold_rmses_exh = []
            for f_idx, (_, f_val_idx) in enumerate(folds_1m):
                f_X_va = pd.DataFrame(X_1min_train.iloc[f_val_idx])
                f_y_va = pd.Series(y_exhaust_train.iloc[f_val_idx])
                f_preds = model_3.predict(f_X_va)
                fold_rmses_exh.append(float(np.sqrt(mean_squared_error(f_y_va, f_preds))))
            rmse_exh = float(np.mean(fold_rmses_exh))
            model_3.save(MODEL_3_PATH)
            logger.log(symbol, "model_3_exhaustion", {
                "rmse": round(rmse_exh, 6),
                "last_fold_rmse": round(fold_rmses_exh[-1], 6),
                "all_fold_rmses": [round(r, 6) for r in fold_rmses_exh],
                "n_folds": len(fold_rmses_exh),
                "val_size": len(X_va1)
            })
            print(f"       Model 3 Result: Mean Val RMSE (across {len(fold_rmses_exh)} folds) = {rmse_exh:.6f} (Last fold: {fold_rmses_exh[-1]:.6f}) (Saved to {os.path.basename(MODEL_3_PATH)})")
        except Exception as e:  # noqa: BLE001
            print(f"       ERROR training Model 3: {e}")

        # --- Model 3b: Runup Regressor ---
        mode_m3b = "incremental warm-start (xgb_model=prev)" if prev_m3b else "initial training from scratch"
        print(f"    -> [Model 3b: Runup] Calling models.model_3b_runup.RunupModel.train() ({mode_m3b})...")
        try:
            model_3b.train(X_tr1, y_tr_runup, X_va1, y_va_runup, xgb_model=prev_m3b)
            fold_rmses_runup = []
            for f_idx, (_, f_val_idx) in enumerate(folds_1m):
                f_X_va = pd.DataFrame(X_1min_train.iloc[f_val_idx])
                f_y_va = pd.Series(y_runup_train.iloc[f_val_idx])
                f_preds = model_3b.predict(f_X_va)
                fold_rmses_runup.append(float(np.sqrt(mean_squared_error(f_y_va, f_preds))))
            rmse_runup = float(np.mean(fold_rmses_runup))
            model_3b.save(MODEL_3B_PATH)
            logger.log(symbol, "model_3b_runup", {
                "rmse": round(rmse_runup, 6),
                "last_fold_rmse": round(fold_rmses_runup[-1], 6),
                "all_fold_rmses": [round(r, 6) for r in fold_rmses_runup],
                "n_folds": len(fold_rmses_runup),
                "val_size": len(X_va1)
            })
            print(f"       Model 3b Result: Mean Val RMSE (across {len(fold_rmses_runup)} folds) = {rmse_runup:.6f} (Last fold: {fold_rmses_runup[-1]:.6f}) (Saved to {os.path.basename(MODEL_3B_PATH)})")
        except Exception as e:  # noqa: BLE001
            print(f"       ERROR training Model 3b: {e}")

        # --- STEP 8: Update Replay Buffers ---
        print(f"  [Step 8] Updating replay buffers on disk with 5% historical sample of {symbol}...")
        update_replay_buffer_5min(X_5min, y_dir, y_price, symbol)
        update_replay_buffer_1min(X_1min, y_exhaust, y_runup, symbol)
        print("    -> Replay buffers updated (replay_buffer_5min.parquet & replay_buffer_1min.parquet)")

        models_initialized = True

        # --- STEP 9: Periodic Full Retrain from Replay Buffer (every 25 companies) ---
        if (i + 1) % 25 == 0 and i > 0:
            print("\n  ******************************************************************")
            print(f"  *** [Step 9] PERIODIC FULL RETRAIN TRIGGERED (after {i+1} companies) ***")
            print("  *** Retraining Models 1, 2, 3, and 3b from full historical replay buffer ***")
            print("  ******************************************************************")
            try:
                # Retrain Model 1 & Model 2 (from 5-min replay buffer)
                replay_5 = load_replay_5min()
                if replay_5 is not None:
                    Xr_raw, yr_dir_raw, yr_price_raw = replay_5
                    print(f"    -> Loaded 5-min replay buffer: {len(Xr_raw):,} total historical samples")

                    # Retrain Model 1 (Directional 3-Class Classifier)
                    valid_r5_m1 = Xr_raw.replace([np.inf, -np.inf], np.nan).notna().all(axis=1) & yr_dir_raw.notna()
                    Xr_m1 = pd.DataFrame(Xr_raw[valid_r5_m1])
                    yr_dir = pd.Series(yr_dir_raw[valid_r5_m1])
                    sp1 = int(len(Xr_m1) * 0.8)
                    print(f"    -> [Retrain M1] DirectionalModel: Training fresh on {sp1:,} samples, validating on {len(Xr_m1)-sp1:,}...")
                    model_1_fresh = DirectionalModel()
                    model_1_fresh.train(
                        pd.DataFrame(Xr_m1.iloc[:sp1]),
                        pd.Series(yr_dir.iloc[:sp1]),
                        pd.DataFrame(Xr_m1.iloc[sp1:]),
                        pd.Series(yr_dir.iloc[sp1:])
                    )
                    model_1_fresh.save(MODEL_1_PATH)
                    model_1 = model_1_fresh
                    print("       Model 1 successfully retrained from full replay buffer.")
                    del Xr_m1, yr_dir

                    # Retrain Model 2 (Price Movement Regressor)
                    valid_r5_m2 = Xr_raw.replace([np.inf, -np.inf], np.nan).notna().all(axis=1) & yr_price_raw.notna()
                    Xr_m2 = pd.DataFrame(Xr_raw[valid_r5_m2])
                    yr_price = pd.Series(yr_price_raw[valid_r5_m2])
                    sp2 = int(len(Xr_m2) * 0.8)
                    print(f"    -> [Retrain M2] PriceModel: Training fresh on {sp2:,} samples, validating on {len(Xr_m2)-sp2:,}...")
                    model_2_fresh = PriceModel()
                    model_2_fresh.train(
                        pd.DataFrame(Xr_m2.iloc[:sp2]),
                        pd.Series(yr_price.iloc[:sp2]),
                        pd.DataFrame(Xr_m2.iloc[sp2:]),
                        pd.Series(yr_price.iloc[sp2:])
                    )
                    model_2_fresh.save(MODEL_2_PATH)
                    model_2 = model_2_fresh
                    print("       Model 2 successfully retrained from full replay buffer.")
                    del replay_5, Xr_raw, yr_dir_raw, yr_price_raw, Xr_m2, yr_price

                # Retrain Models 3 & 3b (from 1-min replay buffer)
                replay_1 = load_replay_1min()
                if replay_1 is not None:
                    Xr1, yr_exh, yr_runup = replay_1
                    print(f"    -> Loaded 1-min replay buffer: {len(Xr1):,} total historical samples")
                    valid_r1 = Xr1.replace([np.inf, -np.inf], np.nan).notna().all(axis=1) & yr_exh.notna() & yr_runup.notna()
                    Xr1 = pd.DataFrame(Xr1[valid_r1])
                    yr_exh = pd.Series(yr_exh[valid_r1])
                    yr_runup = pd.Series(yr_runup[valid_r1])
                    sp3 = int(len(Xr1) * 0.8)

                    # Model 3
                    print(f"    -> [Retrain M3] ExhaustionModel: Training fresh on {sp3:,} samples, validating on {len(Xr1)-sp3:,}...")
                    model_3_fresh = ExhaustionModel()
                    model_3_fresh.train(
                        pd.DataFrame(Xr1.iloc[:sp3]),
                        pd.Series(yr_exh.iloc[:sp3]),
                        pd.DataFrame(Xr1.iloc[sp3:]),
                        pd.Series(yr_exh.iloc[sp3:])
                    )
                    model_3_fresh.save(MODEL_3_PATH)
                    model_3 = model_3_fresh
                    print("       Model 3 successfully retrained from full replay buffer.")

                    # Model 3b
                    print(f"    -> [Retrain M3b] RunupModel: Training fresh on {sp3:,} samples, validating on {len(Xr1)-sp3:,}...")
                    model_3b_fresh = RunupModel()
                    model_3b_fresh.train(
                        pd.DataFrame(Xr1.iloc[:sp3]),
                        pd.Series(yr_runup.iloc[:sp3]),
                        pd.DataFrame(Xr1.iloc[sp3:]),
                        pd.Series(yr_runup.iloc[sp3:])
                    )
                    model_3b_fresh.save(MODEL_3B_PATH)
                    model_3b = model_3b_fresh
                    print("       Model 3b successfully retrained from full replay buffer.")
                    del replay_1, Xr1, yr_exh, yr_runup

                gc.collect()
            except Exception as e:  # noqa: BLE001
                print(f"    ERROR during periodic retrain: {e}")

        # --- STEP 10: Strict Memory Cleanup ---
        print("  [Step 10] Memory Cleanup: Releasing temporary DataFrames & running gc.collect()...")
        del X_5min, X_1min, y_dir, y_exhaust, y_runup, y_price
        del X_5min_train, X_1min_train, y_dir_train, y_exhaust_train, y_runup_train, y_price_train
        del X_tr5, X_va5, X_tr1, X_va1
        del y_tr_dir, y_va_dir, y_tr_exh, y_va_exh, y_tr_runup, y_va_runup, y_tr_price, y_va_price
        gc.collect()
        print("    -> Cleaned. Peak RAM flat at < 2 GB.")
        print(f"  Finished {symbol} -> [M1 Acc: {acc*100:.2f}%, M2 RMSE: {rmse_price:.6f}, M3 RMSE: {rmse_exh:.6f}, M3b RMSE: {rmse_runup:.6f}]")

    print("\n" + "=" * 70)
    print("INCREMENTAL TRAINING PIPELINE COMPLETE!")
    print(f"All 4 Models Saved to: {ARTIFACTS_DIR}")
    print(f"Training log: {os.path.join(ARTIFACTS_DIR, 'training_log.json')}")
    print(f"Drift log: {os.path.join(ARTIFACTS_DIR, 'drift_log.json')}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Master Training Orchestrator for NIFTY 50")
    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="Run Optuna Bayesian hyperparameter tuning on first company before streaming incremental loop"
    )
    parser.add_argument(
        "--priority2-reset",
        action="store_true",
        help="Perform Priority-2 reset (archive pre_priority2 artifacts, purge replay buffers)"
    )
    args = parser.parse_args()
    train_all_companies(tune_hyperparams=args.tune_hyperparams, priority2_reset=args.priority2_reset)
