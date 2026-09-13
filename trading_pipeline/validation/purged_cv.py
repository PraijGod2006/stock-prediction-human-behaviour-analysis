"""
========================================================================================
PURGED WALK-FORWARD CROSS-VALIDATION WITH OPTUNA
========================================================================================
WHY NOT RANDOM TRAIN/TEST SPLIT?
In normal machine learning, you can randomly shuffle data into train/test sets.
But financial time-series data has "autocorrelation" — each minute's price is heavily
influenced by the previous minutes. If we randomly split, the model accidentally
sees future patterns in the training set (this is called "data leakage").

WHAT IS PURGED WALK-FORWARD CV?
It is a time-respecting validation scheme inspired by Marcos Lopez de Prado's book
"Advances in Financial Machine Learning":

1. We split data into sequential folds (e.g., months).
2. We train on fold 1, then test on fold 3 — but we SKIP fold 2 entirely.
   - The "purge gap" (fold 2) removes any rows whose labels overlap with the
     training data (e.g., a 10-minute forward drawdown label computed at the
     end of fold 1 would leak into fold 2).
   - The "embargo gap" adds extra safety margin after the purge.
3. We slide the window forward and repeat.

WHY OPTUNA?
Instead of manually trying hyperparameter combinations (grid search), Optuna uses
Bayesian optimization — it learns from previous trials which hyperparameters work
best, and intelligently suggests the next combination to try. This finds better
models in fewer iterations.
========================================================================================
"""


import numpy as np
import optuna
import pandas as pd

# Silence Optuna's verbose trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class PurgedWalkForwardCV:
    """
    Implements time-ordered walk-forward cross-validation with purge and embargo gaps.
    
    This prevents lookahead bias that occurs with random/shuffled splits on
    autocorrelated financial time-series data.
    
    Parameters:
        n_splits (int): Number of walk-forward folds.
        purge_gap (int): Number of rows to purge between train and validation sets.
                         This removes rows whose forward-looking labels overlap with
                         the end of the training set.
        embargo_gap (int): Additional rows to skip after the purge for extra safety.
    """
    
    def __init__(self, n_splits: int = 5, purge_gap: int = 10, embargo_gap: int = 5):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
    
    def split(self, X: pd.DataFrame):
        """
        Generates train/validation index pairs for each fold.
        
        The data is divided into (n_splits + 1) sequential chunks.
        For each fold i:
          - Train on chunks [0, ..., i]
          - Purge the last `purge_gap` rows from training before validation
          - Validate on chunk [i + 1]
          - Embargo `embargo_gap` rows immediately after validation, preventing 
            them from being used in future training sets
        
        Yields:
            (train_indices, val_indices): Arrays of integer indices for each fold.
        """
        n_samples = len(X)
        # Size of each chunk (the validation window)
        chunk_size = n_samples // (self.n_splits + 1)
        embargo_set = set()
        
        for i in range(self.n_splits):
            # Training: everything from the start up to the end of chunk i
            train_end = chunk_size * (i + 1)
            
            # Apply purge: remove the last purge_gap rows from training
            # These rows have forward-looking labels that overlap with validation
            purged_train_end = max(0, train_end - self.purge_gap)
            
            # Validation: starts exactly at train_end
            val_start = train_end
            val_end = min(n_samples, chunk_size * (i + 2))
            
            # Safety check: validation must have data
            if val_start >= val_end:
                continue
            
            # Exclude embargoed indices from training
            train_idx = np.array([j for j in range(purged_train_end) if j not in embargo_set])
            val_idx = np.arange(val_start, val_end)
            
            # Add embargo gap after this fold's validation
            embargo_end = min(n_samples, val_end + self.embargo_gap)
            embargo_set.update(range(val_end, embargo_end))
            
            yield train_idx, val_idx
    
    def get_n_splits(self) -> int:
        return self.n_splits


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    model_class,
    n_trials: int = 50,
    n_splits: int = 5,
    purge_gap: int = 10,
    embargo_gap: int = 5,
    is_classifier: bool = True
) -> dict:
    """
    Uses Optuna Bayesian optimization to find the best XGBoost hyperparameters
    evaluated via purged walk-forward cross-validation.
    
    This is the core function that replaces naive grid search. Instead of trying
    every combination, Optuna learns from previous trials and converges on the
    best hyperparameters much faster.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        model_class: One of DirectionalModel, PriceModel, or ExhaustionModel
        n_trials: Number of Optuna trials to run (more = better search, slower)
        n_splits: Number of walk-forward CV folds
        purge_gap: Rows to purge between train and validation
        embargo_gap: Additional embargo rows after purge
        is_classifier: True for classification (accuracy), False for regression (RMSE)
    
    Returns:
        dict: Best hyperparameters found by Optuna
    """
    cv = PurgedWalkForwardCV(n_splits=n_splits, purge_gap=purge_gap, embargo_gap=embargo_gap)
    
    def objective(trial: optuna.Trial) -> float:
        """
        Optuna calls this function for each trial. It suggests hyperparameters,
        trains and evaluates the model via purged CV, and returns the score.
        
        Optuna learns which parameter ranges produce good scores and focuses
        its search there (this is the "Bayesian" part).
        """
        # --- Optuna suggests hyperparameters from these ranges ---
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        
        # --- Evaluate across all purged CV folds ---
        scores = []
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            import xgboost as xgb
            
            if is_classifier:
                model = xgb.XGBClassifier(
                    **params,
                    tree_method='hist',
                    device='cuda',
                    random_state=42,
                    eval_metric='logloss',
                    early_stopping_rounds=30,
                    verbosity=0
                )
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict(X_val)
                # Accuracy score
                score = (preds == y_val.values).mean()
            else:
                model = xgb.XGBRegressor(
                    **params,
                    tree_method='hist',
                    device='cuda',
                    random_state=42,
                    eval_metric='rmse',
                    early_stopping_rounds=30,
                    verbosity=0
                )
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict(X_val)
                # Negative RMSE (Optuna maximizes, so we negate the error)
                rmse = np.sqrt(((preds - y_val.values) ** 2).mean())
                score = -rmse
            
            scores.append(score)
        
        # Return the average score across all folds
        return np.mean(scores)
    
    # --- Run the Optuna study ---
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best trial score: {study.best_trial.value:.4f}")
    print(f"Best hyperparameters: {study.best_trial.params}")
    
    return study.best_trial.params
