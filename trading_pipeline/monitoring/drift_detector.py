"""
========================================================================================
MONITORING: Feature Distribution Drift Detection
========================================================================================
WHY MONITOR DRIFT?
Machine learning models learn patterns from historical data. But markets change over
time (this is called "regime change" — e.g., a pandemic crash, a bull run, or new
regulations). When the statistical properties of the input features shift significantly
from what the model was trained on, the model's predictions become unreliable.

HOW DO WE DETECT IT?
We use the Population Stability Index (PSI), a standard metric in credit risk modeling
and quantitative finance. PSI measures how much the distribution of a feature has
shifted between two time periods:

    PSI = SUM( (Actual% - Expected%) * ln(Actual% / Expected%) )

- PSI < 0.10 → No significant shift (model is fine)
- PSI 0.10 - 0.25 → Moderate shift (monitor closely)
- PSI > 0.25 → Significant shift (RETRAIN the model!)

We also support the Kolmogorov-Smirnov (KS) test as an alternative.
========================================================================================
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Optional


def calculate_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) between two distributions.
    
    PSI quantifies how much a feature's distribution has shifted from the
    baseline (training data) to the current data.
    
    Formula:
        PSI = SUM( (actual_pct_i - expected_pct_i) * ln(actual_pct_i / expected_pct_i) )
    
    Args:
        expected: Array of feature values from the training/baseline period.
        actual: Array of feature values from the current/live period.
        n_bins: Number of histogram bins to divide the distributions into.
    
    Returns:
        float: PSI value. < 0.10 = stable, 0.10-0.25 = moderate shift, > 0.25 = major shift.
    """
    # Remove NaN/Inf values
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    # Create bins from the expected (baseline) distribution
    # Using percentile-based bins ensures each bin has roughly equal population
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicate edges
    
    if len(breakpoints) < 2:
        return 0.0
    
    # Count how many values fall into each bin for both distributions
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    # Convert counts to percentages (add small epsilon to avoid division by zero)
    epsilon = 1e-6
    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon
    
    # PSI formula: sum of (actual% - expected%) * ln(actual% / expected%)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return float(psi)


class DriftDetector:
    """
    Monitors feature distribution drift across companies and over time.
    
    During the incremental training loop, this class:
    1. Stores the baseline feature distributions from the first batch of training data.
    2. For each subsequent company, computes PSI against the baseline.
    3. Flags companies where features have drifted significantly.
    4. Logs all metrics to a JSON file for post-analysis.
    
    Attributes:
        psi_threshold (float): PSI value above which a feature is considered "drifted".
        baseline_stats (dict): Stored baseline feature distributions.
        log_path (str): Path to the JSON drift log file.
    """
    
    def __init__(self, log_path: str, psi_threshold: float = 0.25):
        self.psi_threshold = psi_threshold
        self.baseline_stats: dict = {}
        self.log_path = log_path
        self.drift_log: list = []
        
        # Load existing log if it exists
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                self.drift_log = json.load(f)
    
    def set_baseline(self, X: pd.DataFrame, symbol: str = "BASELINE"):
        """
        Stores the baseline feature distributions from the initial training data.
        All future drift checks compare against this baseline.
        
        Args:
            X: Feature DataFrame from the initial training data.
            symbol: Label for this baseline snapshot.
        """
        self.baseline_stats = {}
        for col in X.columns:
            values = X[col].dropna().values
            if len(values) > 0:
                self.baseline_stats[col] = values
        
        print(f"Drift baseline set from {symbol} with {len(X.columns)} features.")
    
    def check_drift(self, X: pd.DataFrame, symbol: str) -> dict:
        """
        Checks each feature in X against the stored baseline using PSI.
        
        Args:
            X: Current feature DataFrame to check.
            symbol: Company/stock ticker being checked.
        
        Returns:
            dict with keys:
                'symbol': stock name
                'drifted_features': list of features that exceeded PSI threshold
                'psi_scores': dict of feature -> PSI value
                'needs_retrain': bool indicating if any feature drifted
        """
        if not self.baseline_stats:
            print("WARNING: No baseline set. Call set_baseline() first.")
            return {'symbol': symbol, 'drifted_features': [], 'psi_scores': {}, 'needs_retrain': False}
        
        psi_scores = {}
        drifted_features = []
        
        for col in X.columns:
            if col not in self.baseline_stats:
                continue
            
            actual = X[col].dropna().values
            expected = self.baseline_stats[col]
            
            psi = calculate_psi(expected, actual)
            psi_scores[col] = round(psi, 4)
            
            if psi > self.psi_threshold:
                drifted_features.append(col)
        
        needs_retrain = len(drifted_features) > 0
        
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'drifted_features': drifted_features,
            'psi_scores': psi_scores,
            'needs_retrain': needs_retrain,
            'max_psi': max(psi_scores.values()) if psi_scores else 0.0
        }
        
        # Log the result
        self.drift_log.append(result)
        self._save_log()
        
        if needs_retrain:
            print(f"  DRIFT DETECTED in {symbol}: {len(drifted_features)} features exceeded PSI threshold.")
        
        return result
    
    def _save_log(self):
        """Persists the drift log to disk as JSON."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.drift_log, f, indent=2)
