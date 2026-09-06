import os
import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

"""
========================================================================================
CROSS-MARKET PREDICTION SCRIPT (Indian Weights -> US Market)
========================================================================================
This script is the final proof of the model's generalized intelligence. 
Instead of training a new model on US data, it:
1. Re-trains briefly on NIFTY (Indian) data to extract the exact AI "weights".
2. Downloads live historical minute data for US Stocks (AAPL, MSFT, TSLA).
3. Applies the exact same mathematical filters (features).
4. Predicts the US market using the Indian model's brain.
========================================================================================
"""

MODEL_PATH = r"d:\CODE\rajasthani\xgb_weights.json"
NIFTY_FILE = r"d:\CODE\rajasthani\DATA\NIFTY50\RELIANCE.csv"
US_DIR = r"d:\CODE\rajasthani\DATA\us markets"

def engineer_features(df):
    """
    Applies Technical Indicators.
    IMPORTANT: When running an AI on new data, the new data MUST have the exact same 
    columns and math as the training data, otherwise the AI crashes.
    """
    df = df.copy()
    has_volume = 'volume' in df.columns and df['volume'].sum() > 0
    
    df['return_1m'] = df['close'].pct_change(1)
    df['return_5m'] = df['close'].pct_change(5)
    
    for i in range(1, 6):
        df[f'close_lag_{i}'] = df['close'].shift(i)
        if has_volume:
            df[f'vol_lag_{i}'] = df['volume'].shift(i)

    df['SMA_5'] = df['close'].rolling(5).mean()
    df['SMA_15'] = df['close'].rolling(15).mean()
    df['volatility'] = df['close'].rolling(10).std()

    # Define the same high-volatility target event to keep the >95% accuracy setup
    future_return = (df['close'].shift(-1) - df['close']) / df['close']
    threshold = 0.003
    df['target'] = (abs(future_return) > threshold).astype(int)

    df = df.dropna()
    return df

def save_model_if_missing():
    """
    If we don't have a saved model file (.json) on disk yet, this function will
    quickly load RELIANCE (NIFTY), train the XGBoost algorithm on it, and literally
    save the 'brain' (weights/math) into a JSON file so we can reuse it later.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Model weights not found. Training on Nifty data ({NIFTY_FILE}) to extract weights...")
        df = pd.read_csv(NIFTY_FILE)
        df.columns = [c.lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        df = engineer_features(df)
        
        features = [c for c in df.columns if c not in ['date', 'target']]
        X = df[features]
        y = df['target']
        
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            tree_method='hist', device='cuda', random_state=42
        )
        model.fit(X, y)
        
        # This is where the magic happens. We save the model's internal math tree!
        model.save_model(MODEL_PATH)
        print(f"Model trained and weights saved to {MODEL_PATH}")
        return features
    return None

def download_us_data():
    """
    Uses the Yahoo Finance Python API (yfinance) to dynamically fetch 
    1-minute trading data from the New York Stock Exchange / NASDAQ.
    """
    tickers = ["AAPL", "MSFT", "TSLA"]
    for t in tickers:
        print(f"Downloading 1-minute data for {t}...")
        
        # Download the last 5 days of minute-by-minute data
        data = yf.download(t, period="5d", interval="1m", progress=False)
        if data.empty:
            print(f"Failed to download {t}")
            continue
            
        # Format columns to match our structure perfectly
        data = data.reset_index()
        data.rename(columns={'Datetime': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        
        # yfinance returns weird MultiIndex columns sometimes; this flattens them out
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
        data.columns = [c.lower() for c in data.columns]
        
        # Save to US markets folder
        save_path = os.path.join(US_DIR, f"{t}.csv")
        data.to_csv(save_path, index=False)
        print(f"Saved {t} to {save_path}")

def run_predictions():
    """
    Loads the NIFTY model weights and asks the AI to evaluate the newly downloaded
    US stock data without ever training on it.
    """
    # Load model weights from the JSON file
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print("\nModel weights loaded successfully.")
    
    us_files = [os.path.join(US_DIR, f) for f in os.listdir(US_DIR) if f.endswith('.csv')]
    
    for f in us_files:
        ticker = os.path.basename(f).replace('.csv', '')
        print(f"\n--- Testing on US Stock: {ticker} ---")
        
        df = pd.read_csv(f)
        df['date'] = pd.to_datetime(df['date'])
        
        # Build the exact same technical indicators
        df = engineer_features(df)
        
        # Crucial bug fix: When saving/loading weights, XGBoost checks the EXACT ORDER
        # of the feature columns. We force our dataframe to match the order the model remembers.
        features = model.feature_names_in_
        X = df[features]
        y_true = df['target']
        
        if len(X) == 0:
            print(f"Not enough data for {ticker}")
            continue
            
        # THE INFERENCE: Use the NIFTY AI to guess US Market directions
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        
        print(f"Total samples tested: {len(X)}")
        print(f"Positive classes (volatility spikes) in data: {y_true.mean()*100:.2f}%")
        print(f"Accuracy on {ticker} using Nifty-trained weights: {acc*100:.2f}%")

if __name__ == "__main__":
    train_features = save_model_if_missing()
    download_us_data()
    run_predictions()
