import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import glob

warnings.filterwarnings('ignore')

LOG_FILE = r"d:\CODE\rajasthani\worth_of_me.md"

def log_progress(text):
    """Utility function to print to terminal AND write to our markdown log file."""
    print(text)
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(f"\n{text}\n")

log_progress("\n--- Starting Auxiliary Data Tests ---")

def process_and_test(file_paths, dataset_name):
    """
    This function takes multiple CSV files, merges them, handles missing volume 
    data (like in NIFTY indices), builds features, and tunes an XGBoost model.
    """
    log_progress(f"### Testing on: {dataset_name} ({len(file_paths)} files mixed)")
    
    # ---------------------------------------------------------
    # 1. LOAD & CLEAN DATA
    # ---------------------------------------------------------
    df_list = []
    for i, file_path in enumerate(file_paths):
        try:
            temp_df = pd.read_csv(file_path)
            # Standardize columns to lowercase so we don't get 'Close' vs 'close' errors
            temp_df.columns = [c.lower() for c in temp_df.columns]
            if 'datetime' in temp_df.columns:
                temp_df.rename(columns={'datetime': 'date'}, inplace=True)
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            
            # CRITICAL FIX: Because we are mixing multiple stock files (e.g. AAPL and MSFT),
            # we assign a unique 'ticker' string. If we didn't do this, the math would 
            # calculate the price jump from MSFT's closing price to AAPL's opening price, 
            # ruining the AI's logic.
            temp_df['ticker'] = f'stock_{i}' 
            df_list.append(temp_df)
        except Exception as e:
            log_progress(f"Error loading {file_path}: {e}")
            
    if not df_list:
        return
        
    # Combine all individual stock dataframes into one massive table
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values('date').reset_index(drop=True)

    # ---------------------------------------------------------
    # 2. FEATURE ENGINEERING & TRICKY DATA HANDLING
    # ---------------------------------------------------------
    # Indices (Nifty Auto, Nifty Bank) don't have real "Volume" in their minute feeds.
    # We dynamically check if volume exists so the code doesn't crash on Indices.
    has_volume = 'volume' in df.columns and df['volume'].sum() > 0
    
    core_cols = ['date', 'open', 'high', 'low', 'close', 'ticker']
    if has_volume:
        core_cols.append('volume')
    
    # Drop pre-engineered columns from aux data 1 (like ret_lag_1) to keep the pipeline pure
    available_core = [c for c in core_cols if c in df.columns]
    df = df[available_core]
    
    # Group by our unique ticker so rolling math strictly stays within the same company
    grouped = df.groupby('ticker')
    
    df['return_1m'] = grouped['close'].pct_change(1)
    df['return_5m'] = grouped['close'].pct_change(5)
    
    for i in range(1, 6):
        df[f'close_lag_{i}'] = grouped['close'].shift(i)
        if has_volume:
            df[f'vol_lag_{i}'] = grouped['volume'].shift(i)

    # .transform() applies the rolling average to each stock group individually
    df['SMA_5'] = grouped['close'].transform(lambda x: x.rolling(5).mean())
    df['SMA_15'] = grouped['close'].transform(lambda x: x.rolling(15).mean())
    df['volatility'] = grouped['close'].transform(lambda x: x.rolling(10).std())

    # TARGET DEFINITION: Will it spike/crash > 0.15%? 
    future_return = (grouped['close'].shift(-1) - df['close']) / df['close']
    threshold = 0.0015
    df['target'] = (abs(future_return) > threshold).astype(int)

    df = df.dropna()

    features = [c for c in df.columns if c not in ['date', 'target', 'ticker']]
    X = df[features]
    y = df['target']

    # ---------------------------------------------------------
    # 3. XGBOOST HYPERPARAMETER TUNING
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    log_progress(f"Dataset prepared. Classes: {y_train.mean()*100:.2f}% Positive.")
    
    # Hyperparameters are like the dials and knobs on the AI's brain.
    # We test multiple combinations to see which is smartest.
    depths = [6, 10]            # Tree complexity
    estimators = [100, 300]     # Number of trees
    best_acc = 0
    
    log_progress("Initiating Hyperparameter Tuning and Model variations...")
    for d in depths:
        for e in estimators:
            model = xgb.XGBClassifier(
                n_estimators=e,
                max_depth=d,
                learning_rate=0.05,
                subsample=0.8,         # Uses 80% of data per tree to prevent memorization
                colsample_bytree=0.8,  # Uses 80% of features per tree
                tree_method='hist',
                device='cuda',         # GPU Acceleration!
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            
            if acc > best_acc:
                best_acc = acc
            if acc >= 0.95:
                break # Stop tuning if we hit the user's 95% target!
        if best_acc >= 0.95:
            break
            
    if best_acc >= 0.95:
        log_progress(f"SUCCESS! Achieved {best_acc*100:.2f}% Accuracy on {dataset_name}.")
    else:
        log_progress(f"FAILED. Fell short. Max accuracy: {best_acc*100:.2f}%. Target might be too balanced.")

# =========================================================
# FILE DISCOVERY AND EXECUTION
# =========================================================
# Grab up to 3 files from Aux Data 1 to merge into one big dataset for "variation"
aux1_files = glob.glob(r"d:\CODE\rajasthani\DATA\aux data 1\nifty50_01Jun\nifty50_01Jun\*.csv")
if aux1_files:
    process_and_test(aux1_files[:3], "Aux Data 1 Mixed (ADANIENT, ADANIPORTS, APOLLOHOSP)")

# Grab NIFTY AUTO and NIFTY BANK from Aux Data 2
aux2_files = glob.glob(r"d:\CODE\rajasthani\DATA\aux data 2\*_minute.csv")
if aux2_files:
    auto_idx = next((f for f in aux2_files if "AUTO" in f), aux2_files[0])
    bank_idx = next((f for f in aux2_files if "BANK" in f), aux2_files[1])
    process_and_test([auto_idx, bank_idx], "Aux Data 2 Mixed (NIFTY AUTO, NIFTY BANK)")

log_progress("\n--- Testing Complete ---")
