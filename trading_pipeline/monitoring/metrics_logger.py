"""
========================================================================================
MONITORING: Lightweight JSON-Based Metrics Logger
========================================================================================
This replaces the heavyweight MLflow dependency with a simple JSON-based logging system.
Every time a model is trained on a new company, the validation metrics are appended
here so you can track the model's performance over time without needing a database.

WHAT IT LOGS:
- Company name and timestamp
- Model ID (directional, price, exhaustion)
- Validation accuracy (classification) or RMSE (regression)
- Feature importances (top 10)
- PSI drift scores
- Whether a full retrain was triggered
========================================================================================
"""

import json
import os
from datetime import datetime
from typing import Optional


class MetricsLogger:
    """
    Simple, file-based metrics logger. Appends training metrics to a JSON file
    after each incremental training step.
    
    No external dependencies (no MLflow, no databases, no servers).
    Just a clean JSON array that you can open in any text editor or load in Python.
    
    Usage:
        logger = MetricsLogger("artifacts/training_log.json")
        logger.log(symbol="RELIANCE", model_id="model_1_direction", metrics={...})
        all_metrics = logger.load_metrics()
    """
    
    def __init__(self, log_path: str):
        """
        Args:
            log_path: Absolute path to the JSON log file.
                      Created automatically if it doesn't exist.
        """
        self.log_path = log_path
        self.entries: list = []
        
        # Load existing entries if the file exists
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    self.entries = json.load(f)
            except json.JSONDecodeError:
                self.entries = []
    
    def log(
        self,
        symbol: str,
        model_id: str,
        metrics: dict,
        feature_importances: Optional[dict] = None,
        psi_scores: Optional[dict] = None,
        fold_number: Optional[int] = None,
        retrain_triggered: bool = False,
        notes: str = ""
    ):
        """
        Logs a single training/validation step.
        
        Args:
            symbol: Stock ticker (e.g., "RELIANCE", "TCS").
            model_id: Which model was trained ("model_1_direction", "model_2_price", "model_3_exhaustion").
            metrics: Dict of metric values, e.g., {"accuracy": 0.56, "loss": 0.69}.
            feature_importances: Optional dict of top feature -> importance score.
            psi_scores: Optional dict of feature -> PSI drift value.
            fold_number: Which CV fold this metric is from (if using purged CV).
            retrain_triggered: Whether drift detection triggered a full retrain.
            notes: Any additional human-readable notes.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "model_id": model_id,
            "metrics": metrics,
            "fold_number": fold_number,
            "retrain_triggered": retrain_triggered,
            "notes": notes,
        }
        
        if feature_importances is not None:
            # Only store top 10 features to keep the log file manageable
            sorted_feats = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10])
            entry["top_features"] = sorted_feats
        
        if psi_scores is not None:
            entry["psi_scores"] = psi_scores
        
        self.entries.append(entry)
        self._save()
        
        # Print a compact summary
        metric_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        print(f"  [LOG] {symbol} | {model_id} | {metric_str}")
    
    def load_metrics(self, model_id: Optional[str] = None, symbol: Optional[str] = None) -> list:
        """
        Loads and optionally filters the logged metrics.
        
        Args:
            model_id: Filter by model (e.g., "model_1_direction"). None = all models.
            symbol: Filter by stock ticker. None = all symbols.
        
        Returns:
            List of log entries matching the filters.
        """
        results = self.entries
        
        if model_id:
            results = [e for e in results if e.get("model_id") == model_id]
        if symbol:
            results = [e for e in results if e.get("symbol") == symbol]
        
        return results
    
    def get_latest(self, model_id: str) -> Optional[dict]:
        """Returns the most recent log entry for a given model."""
        filtered = self.load_metrics(model_id=model_id)
        return filtered[-1] if filtered else None
    
    def _save(self):
        """Persists the log to disk."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.entries, f, indent=2)
