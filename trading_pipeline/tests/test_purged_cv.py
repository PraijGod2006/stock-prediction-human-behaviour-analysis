import pandas as pd
import numpy as np

from validation.purged_cv import PurgedWalkForwardCV

def test_purged_walk_forward_cv_gaps():
    # 1. Create synthetic data
    n_samples = 1000
    df = pd.DataFrame({'feature': np.random.randn(n_samples)})
    
    purge_gap = 15
    embargo_gap = 25
    n_splits = 5
    
    # 2. Instantiate PurgedWalkForwardCV
    cv = PurgedWalkForwardCV(n_splits=n_splits, purge_gap=purge_gap, embargo_gap=embargo_gap)
    
    # 3. Generate all folds
    folds = list(cv.split(df))
    
    assert len(folds) == n_splits, f"Expected {n_splits} folds, got {len(folds)}"
    
    previous_val_ends = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_set = set(train_idx)
        val_set = set(val_idx)
        
        # 4(a) No index appears in both a training set and its paired validation set
        overlap = train_set.intersection(val_set)
        assert len(overlap) == 0, f"Fold {fold_idx}: Train and validation sets overlap. Overlapping indices: {overlap}"
        
        # 4(b) No training index falls within purge_gap bars immediately before its paired validation set's start
        if len(val_idx) > 0:
            val_start = min(val_idx)
            purge_zone = set(range(val_start - purge_gap, val_start))
            purge_overlap = train_set.intersection(purge_zone)
            assert len(purge_overlap) == 0, f"Fold {fold_idx}: Train set contains indices from the purge gap. Overlapping indices: {purge_overlap}"
        
        # 4(c) No training index in a LATER fold falls within embargo_gap bars immediately after an EARLIER fold's validation set
        for prev_val_end in previous_val_ends:
            embargo_zone = set(range(prev_val_end, prev_val_end + embargo_gap))
            embargo_overlap = train_set.intersection(embargo_zone)
            assert len(embargo_overlap) == 0, f"Fold {fold_idx}: Train set contains indices from an earlier fold's embargo gap. Overlapping indices: {embargo_overlap}"
        
        if len(val_idx) > 0:
            previous_val_ends.append(max(val_idx) + 1)

    print("ALL 3 PURGED CV LEAKAGE ASSERTIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_purged_walk_forward_cv_gaps()
