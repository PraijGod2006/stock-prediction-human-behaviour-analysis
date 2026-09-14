"""
POST-RETRAIN DEVICE & PERFORMANCE AUDIT SCRIPT
Measures exact timing and GPU utilization for one full company processing cycle.
"""

import gc
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, mean_squared_error

sys.path.insert(0, r"d:\CODE\rajasthani\trading_pipeline")

from correlation.cross_asset import inject_cross_asset_features
from feature_engine.pipeline import build_features_1min, build_features_5min
from models.model_1_direction import DirectionalModel
from models.model_2_price import PriceModel
from models.model_3_exhaustion import ExhaustionModel
from train_master import clean_data, load_replay_1min, load_replay_5min, update_replay_buffer_1min, update_replay_buffer_5min
from validation.purged_cv import PurgedWalkForwardCV

PARQUET_PATH = r"d:\CODE\rajasthani\DATA\parquet\RELIANCE.parquet"
SYMBOL = "RELIANCE"
PEER_MAP_PATH = r"d:\CODE\rajasthani\trading_pipeline\artifacts\peer_map.json"
PARQUET_DIR = r"d:\CODE\rajasthani\DATA\parquet"

# Background GPU Monitor Thread
gpu_samples = []
stop_gpu_monitor = False

def gpu_monitor_thread():
    while not stop_gpu_monitor:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            ).decode().strip()
            parts = [int(p.strip()) for p in out.split(',')]
            gpu_samples.append((time.perf_counter(), parts[0], parts[1]))
        except Exception:
            pass
        time.sleep(0.2)

monitor = threading.Thread(target=gpu_monitor_thread, daemon=True)
monitor.start()

print("=" * 80)
print("STARTING DEVICE & PERFORMANCE AUDIT ON RELIANCE")
print("=" * 80)

timings = {}
t_total_start = time.perf_counter()

# --- STAGE 1: Parquet Read Time ---
t0 = time.perf_counter()
df_raw = pl.read_parquet(PARQUET_PATH, memory_map=False).to_pandas()
timings['1. Parquet read time'] = time.perf_counter() - t0
print(f"[Stage 1] Parquet Read: {timings['1. Parquet read time']:.4f}s ({len(df_raw):,} rows)")

# --- STAGE 2: Feature Engineering Time ---
t0 = time.perf_counter()
df_1min = build_features_1min(df_raw)
df_5min = build_features_5min(df_raw)
timings['2. Feature engineering time'] = time.perf_counter() - t0
print(f"[Stage 2] Feature Engineering: {timings['2. Feature engineering time']:.4f}s (1m={len(df_1min):,}, 5m={len(df_5min):,})")

# --- STAGE 3: Cross-Asset Peer Injection Time ---
t0 = time.perf_counter()
df_5min = inject_cross_asset_features(df_5min, SYMBOL, PEER_MAP_PATH, PARQUET_DIR)
timings['3. Cross-asset peer injection time'] = time.perf_counter() - t0
print(f"[Stage 3] Cross-Asset Injection: {timings['3. Cross-asset peer injection time']:.4f}s (cols={len(df_5min.columns)})")

# --- Target Preparation ---
m1_init = DirectionalModel()
m2_init = PriceModel()
m3_init = ExhaustionModel()

y_dir_df = m1_init.prepare_target(df_5min)
y_price_df = m2_init.prepare_target(df_5min, df_1min=df_1min)

target_dir_s = pd.Series(y_dir_df['target'], name='target_dir')
target_price_s = pd.Series(y_price_df['target'], name='target_price')
targets_5m = pd.concat([target_dir_s, target_price_s], axis=1).dropna()
common_idx_5m = df_5min.index.intersection(targets_5m.index)

drop_cols = [
    'date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'mid_price', 'future_close_5m', 'target', '__target_dir',
    '__target_price', '__target_exhaust', '__symbol'
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

df_1min_target = m3_init.prepare_target(df_1min)
y_exhaust_raw = pd.Series(df_1min_target['target'])
X_1min_raw = pd.DataFrame(df_1min_target[[c for c in df_1min_target.columns if c not in drop_cols]])
X_1min, y_exhaust = clean_data(X_1min_raw, y_exhaust_raw)

# --- STAGE 5: Replay Buffer Load + Merge Time ---
t0 = time.perf_counter()
replay_5 = load_replay_5min()
if replay_5 is not None:
    X_replay_5, y_rep_dir, y_rep_price = replay_5
    common_cols_5 = [c for c in X_5min.columns if c in X_replay_5.columns]
    X_5min_train = pd.concat([X_5min[common_cols_5], X_replay_5[common_cols_5]], ignore_index=True)
    y_dir_train = pd.concat([y_dir, y_rep_dir], ignore_index=True)
    y_price_train = pd.concat([y_price, y_rep_price], ignore_index=True)
else:
    X_5min_train = X_5min
    y_dir_train = y_dir
    y_price_train = y_price

replay_1 = load_replay_1min()
if replay_1 is not None:
    X_replay_1, y_rep_exh = replay_1
    common_cols_1 = [c for c in X_1min.columns if c in X_replay_1.columns]
    X_1min_train = pd.concat([X_1min[common_cols_1], X_replay_1[common_cols_1]], ignore_index=True)
    y_exhaust_train = pd.concat([y_exhaust, y_rep_exh], ignore_index=True)
else:
    X_1min_train = X_1min
    y_exhaust_train = y_exhaust
timings['5. Replay buffer load + merge time'] = time.perf_counter() - t0
print(f"[Stage 5] Replay Load & Merge: {timings['5. Replay buffer load + merge time']:.4f}s (5m={len(X_5min_train):,}, 1m={len(X_1min_train):,})")

# --- STAGE 4: Purged CV Fold Generation Time ---
t0 = time.perf_counter()
cv_5m = PurgedWalkForwardCV(n_splits=4, purge_gap=10, embargo_gap=5)
folds_5m = list(cv_5m.split(X_5min_train))
train_idx_5m, val_idx_5m = folds_5m[-1]
X_tr5 = pd.DataFrame(X_5min_train.iloc[train_idx_5m])
X_va5 = pd.DataFrame(X_5min_train.iloc[val_idx_5m])
y_tr_dir = pd.Series(y_dir_train.iloc[train_idx_5m])
y_va_dir = pd.Series(y_dir_train.iloc[val_idx_5m])
y_tr_price = pd.Series(y_price_train.iloc[train_idx_5m])
y_va_price = pd.Series(y_price_train.iloc[val_idx_5m])

cv_1m = PurgedWalkForwardCV(n_splits=4, purge_gap=15, embargo_gap=10)
folds_1m = list(cv_1m.split(X_1min_train))
train_idx_1m, val_idx_1m = folds_1m[-1]
X_tr1 = pd.DataFrame(X_1min_train.iloc[train_idx_1m])
X_va1 = pd.DataFrame(X_1min_train.iloc[val_idx_1m])
y_tr_exh = pd.Series(y_exhaust_train.iloc[train_idx_1m])
y_va_exh = pd.Series(y_exhaust_train.iloc[val_idx_1m])
timings['4. Purged CV fold generation time'] = time.perf_counter() - t0
print(f"[Stage 4] Purged CV Fold Generation: {timings['4. Purged CV fold generation time']:.4f}s")

# --- DTYPE INSPECTION FOR PART 2 ITEM 3 ---
print("\n--- DTYPE AUDIT (Checking for object / non-numeric types) ---")
print(f"X_tr5 dtypes: {set(X_tr5.dtypes)}")
print(f"X_tr1 dtypes: {set(X_tr1.dtypes)}")
non_numeric_5 = [c for c, d in X_tr5.dtypes.items() if not np.issubdtype(d, np.number)]
non_numeric_1 = [c for c, d in X_tr1.dtypes.items() if not np.issubdtype(d, np.number)]
print(f"Non-numeric columns in X_tr5: {non_numeric_5}")
print(f"Non-numeric columns in X_tr1: {non_numeric_1}")

# --- STAGE 6: Model 1 Fit Time ---
model_1 = DirectionalModel()
print(f"\n[Model 1 EXACT PARAMS PASSED TO FIT]: {model_1.params}")
t_m1_start = time.perf_counter()
model_1.train(X_tr5, y_tr_dir, X_va5, y_va_dir)
t_m1_end = time.perf_counter()
timings['6. Model 1 fit time'] = t_m1_end - t_m1_start
print(f"[Stage 6] Model 1 Fit: {timings['6. Model 1 fit time']:.4f}s (Booster configuration device: {model_1.model.get_params().get('device')})")

# --- STAGE 7: Model 2 Fit Time ---
model_2 = PriceModel()
print(f"\n[Model 2 EXACT PARAMS PASSED TO FIT]: {model_2.params}")
t_m2_start = time.perf_counter()
model_2.train(X_tr5, y_tr_price, X_va5, y_va_price)
t_m2_end = time.perf_counter()
timings['7. Model 2 fit time'] = t_m2_end - t_m2_start
print(f"[Stage 7] Model 2 Fit: {timings['7. Model 2 fit time']:.4f}s (Booster configuration device: {model_2.model.get_params().get('device')})")

# --- STAGE 8: Model 3 Fit Time ---
model_3 = ExhaustionModel()
print(f"\n[Model 3 EXACT PARAMS PASSED TO FIT]: {model_3.params}")
t_m3_start = time.perf_counter()
model_3.train(X_tr1, y_tr_exh, X_va1, y_va_exh)
t_m3_end = time.perf_counter()
timings['8. Model 3 fit time'] = t_m3_end - t_m3_start
print(f"[Stage 8] Model 3 Fit: {timings['8. Model 3 fit time']:.4f}s (Booster configuration device: {model_3.model.get_params().get('device')})")

# --- STAGE 9: Save / Serialize Time ---
t0 = time.perf_counter()
# Temporary save for benchmark
os.makedirs(r"d:\CODE\rajasthani\trading_pipeline\scratch", exist_ok=True)
model_1.save(r"d:\CODE\rajasthani\trading_pipeline\scratch\bench_m1.json")
model_2.save(r"d:\CODE\rajasthani\trading_pipeline\scratch\bench_m2.json")
model_3.save(r"d:\CODE\rajasthani\trading_pipeline\scratch\bench_m3.json")
del X_5min, X_1min, y_dir, y_exhaust, y_price
del X_5min_train, X_1min_train, y_dir_train, y_exhaust_train, y_price_train
del X_tr5, X_va5, X_tr1, X_va1
del y_tr_dir, y_va_dir, y_tr_exh, y_va_exh, y_tr_price, y_va_price
gc.collect()
timings['9. Save/serialize time'] = time.perf_counter() - t0
print(f"[Stage 9] Save/Serialize & GC: {timings['9. Save/serialize time']:.4f}s")

# Stop GPU monitoring
stop_gpu_monitor = True
monitor.join(timeout=1.0)

t_total = time.perf_counter() - t_total_start

print("\n" + "=" * 80)
print("GPU UTILIZATION LOG DURING FIT CALLS")
print("=" * 80)
m1_samples = [s for s in gpu_samples if t_m1_start - 0.5 <= s[0] <= t_m1_end + 0.5]
m2_samples = [s for s in gpu_samples if t_m2_start - 0.5 <= s[0] <= t_m2_end + 0.5]
m3_samples = [s for s in gpu_samples if t_m3_start - 0.5 <= s[0] <= t_m3_end + 0.5]

def summarize_gpu(name, samples):
    if not samples:
        print(f"{name}: No samples captured")
        return
    utils = [s[1] for s in samples]
    mems = [s[2] for s in samples]
    print(f"{name}: Peak GPU Util = {max(utils)}% (mean {np.mean(utils):.1f}%), Peak VRAM = {max(mems)} MiB (start {mems[0]} MiB -> peak {max(mems)} MiB)")

summarize_gpu("Model 1 Fit Window", m1_samples)
summarize_gpu("Model 2 Fit Window", m2_samples)
summarize_gpu("Model 3 Fit Window", m3_samples)

print("\n" + "=" * 80)
print("WALL-CLOCK TIME BREAKDOWN (SINGLE COMPANY: RELIANCE)")
print("=" * 80)
print(f"{'Stage Name':<45} | {'Time (s)':<10} | {'% of Total':<10}")
print("-" * 72)
for name, elapsed in timings.items():
    pct = (elapsed / t_total) * 100
    print(f"{name:<45} | {elapsed:<10.4f} | {pct:<10.2f}%")
print("-" * 72)
print(f"{'Total Measured Processing Time':<45} | {t_total:<10.4f} | 100.00%")
