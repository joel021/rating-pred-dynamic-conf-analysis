import numpy as np
import pandas as pd

def find_best_conf_threshold(df_true_pred: pd.DataFrame, ndcg_calculator, k: int):
    """
    Search for the best confidence threshold by evaluating over a range of quantiles.
    
    Parameters
    ----------
    df_true_pred : pd.DataFrame
        DataFrame containing predicted ratings and confidence scores.
    ndcg_calculator : callable
        Function to calculate metrics given df, k, and threshold.
    k : int
        Top-k cutoff for evaluation.
        
    Returns
    -------
    best_quantile : float
    best_conf_mean : np.ndarray
    best_conf_std : np.ndarray
    best_conf_threshold : float
    """
    best_q = None
    best_mean = None
    best_std = None
    best_threshold = None
    best_score = -float('inf')
    
    # Grid search over quantiles from 0.65 to 0.99
    quantiles = np.linspace(0.65, 0.99, 35)
    
    for q in quantiles:
        c_t = df_true_pred['conf_pred'].quantile(q)
        mean, std = ndcg_calculator(df_true_pred, k, c_t)
        score = mean[0]
        if score > best_score:
            best_score = score
            best_q = q
            best_mean = mean
            best_std = std
            best_threshold = c_t
            
    return best_q, best_mean, best_std, best_threshold
