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


def split_ratings(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str,
    fit_ratio: float = 0.75,
    shuffle: bool = False,
    random_state: int | None = None
):
    """
    Split ratings into fit and test sets ensuring:
    - Min 2 interactions per user in fit_df
    - All users and items in test_df are present in fit_df
    """
    user_counts = df[user_col].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    df = df[df[user_col].isin(valid_users)].copy()

    if shuffle:
        rng = np.random.default_rng(random_state)
        df = df.sample(frac=1, random_state=rng.integers(1e9) if random_state is None else random_state).reset_index(drop=True)
    else:
        df = df.sort_values(timestamp_col).reset_index(drop=True)

    fit_indices = []
    test_indices = []

    for u, group in df.groupby(user_col):
        group_indices = group.index.tolist()
        n_group = len(group_indices)
        n_fit = max(2, int(n_group * fit_ratio))
        if n_fit >= n_group:
            fit_indices.extend(group_indices)
        else:
            fit_indices.extend(group_indices[:n_fit])
            test_indices.extend(group_indices[n_fit:])

    fit_df = df.loc[fit_indices].copy()
    fit_items = set(fit_df[item_col].unique())

    final_test_indices = []
    for idx in test_indices:
        item = df.at[idx, item_col]
        if item in fit_items:
            final_test_indices.append(idx)
        else:
            fit_indices.append(idx)

    fit_df = df.loc[fit_indices].copy()
    test_df = df.loc[final_test_indices].copy()

    return fit_df.reset_index(drop=True), test_df.reset_index(drop=True)