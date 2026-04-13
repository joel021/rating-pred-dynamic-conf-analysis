import pandas as pd
import numpy as np

def time_ordered_folds(
    df: pd.DataFrame,
    timestamp_col: str,
    n_folds: int,
    shuffle_within_folds: bool = True,
    random_state: int | None = None
):
    """
    Split a DataFrame into n sequential time-ordered folds.
    
    Properties:
    - Fold 0 contains the oldest samples
    - Fold n-1 contains the newest samples
    - No temporal leakage across folds
    - Optional shuffling within each fold
    
    Parameters
    ----------
    df : pd.DataFrame
    timestamp_col : str
        Column used for temporal ordering
    n_folds : int
    shuffle_within_folds : bool
    random_state : int or None
    
    Returns
    -------
    List[pd.DataFrame]
        List of folds in temporal order
    """
    
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    # 1. Sort globally by time
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    
    n = len(df_sorted)
    
    # 2. Compute fold boundaries (balanced splits)
    fold_sizes = np.full(n_folds, n // n_folds)
    fold_sizes[: n % n_folds] += 1  # distribute remainder
    
    folds = []
    start = 0
    
    rng = np.random.default_rng(random_state)
    
    for fold_size in fold_sizes:
        end = start + fold_size
        
        fold = df_sorted.iloc[start:end].copy()
        
        # 3. Shuffle within fold (no temporal leakage)
        if shuffle_within_folds:
            fold = fold.sample(frac=1, random_state=rng.integers(1e9)).reset_index(drop=True)
        
        folds.append(fold)
        start = end

    return folds