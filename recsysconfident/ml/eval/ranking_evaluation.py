import warnings
import numpy as np
from scipy.stats import entropy

import torch

import pandas as pd
from pandas import DataFrame

from recsysconfident.constants import RELEVANCE_RATIO, PRED_BATCH_SIZE
from recsysconfident.environment import Environment
from recsysconfident.ml.distance_metrics import mae, rmse
from recsysconfident.ml.ranking.rank_metrics import ConfAwareRankingMetrics


def ranking_scores(candidates_norm_df: DataFrame, environ: Environment, k=10) -> dict:

    conf_rank_calculator = ConfAwareRankingMetrics(environ.dataset_info)
    rank_scores_mean, rank_scores_std = conf_rank_calculator.users_mean_std_rank_metrics(candidates_norm_df, k)

    scores_dict = {
        f"mNDCG@{k}": f"{rank_scores_mean[0]:.5f}",
        f"stdNDCG@{k}": f"{rank_scores_std[0]:.5f}",
        f"MAP@{k}": f"{rank_scores_mean[1]:.5f}",
        f"stdMAP@{k}": f"{rank_scores_std[1]:.5f}",
    }
    return scores_dict

def kl_from_columns(df, col_p, col_q, bins=50, eps=1e-12) -> dict:
    x = df[col_p].to_numpy()
    y = df[col_q].to_numpy()

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    p_hist, bin_edges = np.histogram(x, bins=bins, range=(min_val, max_val), density=False)
    q_hist, _         = np.histogram(y, bins=bin_edges, density=False)

    p = p_hist.astype(float)
    q = q_hist.astype(float)

    p = p / p.sum()
    q = q / q.sum()

    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()

    return {"kl_diverence": entropy(p, q)}

def evaluate_batched(model, split_df: pd.DataFrame, environ, device: str, k_values=[3, 10]) -> dict:
    user_col = environ.dataset_info.user_col
    item_col = environ.dataset_info.item_col
    relevance_col = environ.dataset_info.relevance_col
    rmin, rmax = environ.dataset_info.rate_range[:2]
    n_items = environ.dataset_info.n_items
    items_per_user = environ.dataset_info.items_per_user
    
    unique_users = split_df[user_col].unique()
    
    df_grouped = split_df.groupby(user_col)
    pos_items_dict = df_grouped[item_col].apply(list).to_dict()
    pos_rates_dict = df_grouped[relevance_col].apply(list).to_dict()

    model.eval()
    if hasattr(model, 'switch_to_ranking'):
        model.switch_to_ranking()

    results = {f"{metric}@{k}": [] for metric in ["mNDCG", "MAP"] for k in k_values}
    user_batch_size = 256
    max_k = max(k_values)
    
    all_items_tensor = torch.arange(n_items, dtype=torch.int32, device=device) if environ.num_negatives is None else None

    with torch.inference_mode():
        for i in range(0, len(unique_users), user_batch_size):
            batch_users = unique_users[i:i + user_batch_size]

            if environ.num_negatives is None:
                n_u = len(batch_users)
                u_tensor = torch.tensor(batch_users, dtype=torch.int32, device=device).repeat_interleave(n_items)
                i_tensor = all_items_tensor.repeat(n_u)

                preds_list = []
                for j in range(0, len(u_tensor), PRED_BATCH_SIZE):
                    preds, _ = model.predict(u_tensor[j:j + PRED_BATCH_SIZE], i_tensor[j:j + PRED_BATCH_SIZE])
                    preds_list.append(preds)
                
                all_preds = torch.cat(preds_list).view(n_u, n_items)
                padded_true = torch.zeros((n_u, n_items), device=device)

                for b_idx, u in enumerate(batch_users):
                    pos_i = pos_items_dict.get(u, [])
                    pos_r = pos_rates_dict.get(u, [])
                    global_pos_set = items_per_user.get(u, (set(), []))[0]
                    
                    train_pos = list(global_pos_set - set(pos_i))
                    if train_pos:
                        all_preds[b_idx, train_pos] = -float('inf')
                    
                    if pos_i:
                        padded_true[b_idx, pos_i] = torch.tensor(pos_r, dtype=torch.float32, device=device)
                
                padded_true = (padded_true - rmin) / (rmax - rmin)
                padded_preds = all_preds
                max_len = n_items

            else:
                batch_u, batch_i, batch_r = [], [], []
                boundaries = []
                idx = 0
                
                for u in batch_users:
                    pos_i = pos_items_dict.get(u, [])
                    pos_r = pos_rates_dict.get(u, [])
                    global_pos_set = items_per_user.get(u, (set(), []))[0]
                    
                    candidate_negatives = np.setdiff1d(np.arange(n_items), np.array(list(global_pos_set)), assume_unique=True)
                    if len(candidate_negatives) >= environ.num_negatives:
                        neg_i = np.random.choice(candidate_negatives, size=environ.num_negatives, replace=False).tolist()
                    else:
                        neg_i = candidate_negatives.tolist()
                        
                    u_items = pos_i + neg_i
                    u_ratings = pos_r + [rmin] * len(neg_i)
                    n_u_items = len(u_items)
                    
                    batch_u.extend([u] * n_u_items)
                    batch_i.extend(u_items)
                    batch_r.extend(u_ratings)
                    
                    boundaries.append((idx, idx + n_u_items))
                    idx += n_u_items

                if not batch_u:
                    continue

                u_tensor = torch.tensor(batch_u, dtype=torch.int32, device=device)
                i_tensor = torch.tensor(batch_i, dtype=torch.int32, device=device)
                r_tensor = torch.tensor(batch_r, dtype=torch.float32, device=device)
                r_tensor = (r_tensor - rmin) / (rmax - rmin)

                preds_list = []
                for j in range(0, len(u_tensor), PRED_BATCH_SIZE):
                    preds, _ = model.predict(u_tensor[j:j + PRED_BATCH_SIZE], i_tensor[j:j + PRED_BATCH_SIZE])
                    preds_list.append(preds)
                    
                flat_preds = torch.cat(preds_list)
                
                max_len = max(end - start for start, end in boundaries)
                n_users_batch = len(boundaries)
                
                padded_preds = torch.full((n_users_batch, max_len), -float('inf'), device=device)
                padded_true = torch.zeros((n_users_batch, max_len), device=device)
                
                for b_idx, (start, end) in enumerate(boundaries):
                    length = end - start
                    padded_preds[b_idx, :length] = flat_preds[start:end]
                    padded_true[b_idx, :length] = r_tensor[start:end]

            k_actual = min(max_k, max_len)
            _, top_idx = torch.topk(padded_preds, k_actual, dim=1)
            sorted_true = torch.gather(padded_true, 1, top_idx)
            binary_true = (sorted_true >= RELEVANCE_RATIO).float()
            
            total_relevant = (padded_true >= RELEVANCE_RATIO).float().sum(dim=1)
            ideal_true, _ = torch.sort(padded_true, dim=1, descending=True)

            for k in k_values:
                k_curr = min(k, max_len)
                k_true = sorted_true[:, :k_curr]
                k_bin = binary_true[:, :k_curr]
                
                positions = torch.arange(1, k_curr + 1, device=device).float().unsqueeze(0)
                cum_hits = torch.cumsum(k_bin, dim=1)
                
                ap_divisor = total_relevant.clamp(min=1e-9)
                ap = (cum_hits * k_bin / positions).sum(dim=1) / ap_divisor
                ap = torch.where(total_relevant == 0, torch.zeros_like(ap), ap)
                results[f"MAP@{k}"].extend(ap.cpu().tolist())
                
                ideal_true_k = ideal_true[:, :k_curr]
                discounts = torch.log2(torch.arange(2, k_curr + 2, device=device).float()).unsqueeze(0)
                
                dcg = (k_true / discounts).sum(dim=1)
                idcg = (ideal_true_k / discounts).sum(dim=1)
                
                ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))
                results[f"mNDCG@{k}"].extend(ndcg.cpu().tolist())

    final_results = {}
    for k in k_values:
        ndcgs = results.get(f"mNDCG@{k}", [0.0])
        aps = results.get(f"MAP@{k}", [0.0])
        
        if ndcgs and aps:
            final_results[f"mNDCG@{k}"] = f"{np.mean(ndcgs):.5f}"
            final_results[f"stdNDCG@{k}"] = f"{np.std(ndcgs):.5f}"
            final_results[f"MAP@{k}"] = f"{np.mean(aps):.5f}"
            final_results[f"stdMAP@{k}"] = f"{np.std(aps):.5f}"
        else:
            final_results[f"mNDCG@{k}"] = "0.00000"
            final_results[f"stdNDCG@{k}"] = "0.00000"
            final_results[f"MAP@{k}"] = "0.00000"
            final_results[f"stdMAP@{k}"] = "0.00000"

    if hasattr(model, 'switch_to_rating'):
        model.switch_to_rating()
        
    return final_results

def evaluate(split_df: pd.DataFrame, environ: Environment, model=None, device='cpu') -> dict:
    
    dist_divergence_metric = kl_from_columns(split_df, environ.dataset_info.relevance_col, environ.dataset_info.r_pred_col)
    distance_metrics = get_distance_metrics(split_df, environ)
    
    if model is not None:
        ranking_metrics = evaluate_batched(model, split_df, environ, device, [3, 10])
    else:
        warnings.warn(
            "evaluate() was called without a model. Since split_df does not contain negative samples "
            "under the new workflow, ranking metrics (NDCG/MAP) cannot be computed correctly and will be set to 0.0.",
            UserWarning
        )
        ranking_metrics = {
            "mNDCG@10": "0.00000", "stdNDCG@10": "0.00000", "MAP@10": "0.00000", "stdMAP@10": "0.00000",
            "mNDCG@3": "0.00000", "stdNDCG@3": "0.00000", "MAP@3": "0.00000", "stdMAP@3": "0.00000"
        }

    return {**dist_divergence_metric, **distance_metrics, **ranking_metrics}

def get_distance_metrics(split_df: pd.DataFrame, environ: Environment):

    y_true = split_df[environ.dataset_info.relevance_col].values
    y_pred = split_df[environ.dataset_info.r_pred_col].values
    mae_score = mae(y_true, y_pred)
    rmse_score = rmse(y_true, y_pred)

    return {
        "rmse": rmse_score,
        "mae": mae_score
    }

