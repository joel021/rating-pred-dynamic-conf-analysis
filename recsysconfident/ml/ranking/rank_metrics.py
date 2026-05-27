"""
package: recsysconfident.ml.ranking.conf_aware_rank_metrics
"""
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.constants import RELEVANCE_RATIO


class ConfAwareRankingMetrics:

    def __init__(self, data_info: DatasetInfo, r_t: float = RELEVANCE_RATIO, alpha: float = 5):
        self.data_info = data_info
        self.r_t = r_t
        self.alpha = alpha

    def binarize(self, relevances):
        return (relevances >= self.r_t).astype(int)

    def _get_true_pred_scores(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {}

        df_sorted = df.sort_values(by=[self.data_info.user_col, self.data_info.r_pred_col], ascending=[True, False])
        
        users = df_sorted[self.data_info.user_col].values
        relevances = df_sorted[self.data_info.relevance_col].values
        preds = df_sorted[self.data_info.r_pred_col].values
        
        unique_users, indices, counts = np.unique(users, return_index=True, return_counts=True)
        
        user_true_pred_scores = {}
        for user, idx, count in zip(unique_users, indices, counts):
            user_true_pred_scores[user] = (
                relevances[idx:idx+count],
                preds[idx:idx+count]
            )
        return user_true_pred_scores

    def conf_filter(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        if self.data_info.conf_pred_col in df.columns:
            return df[df[self.data_info.conf_pred_col] >= threshold]
        return df

    def rank_metrics(self, norm_df: pd.DataFrame, k: int, conf_threshold: float = -1) -> list:
        if conf_threshold >= 0:
            norm_df = self.conf_filter(norm_df, conf_threshold)

        user_true_pred_scores = self._get_true_pred_scores(norm_df)
        metrics = []
        for user_key in user_true_pred_scores.keys():
            true_ratings, pred_ratings = user_true_pred_scores[user_key]
            binary_true = self.binarize(true_ratings)
            
            # NDCG
            try:
                ndcg = ndcg_score([true_ratings], [pred_ratings], k=k)
            except Exception:
                ndcg = 0.0

            # MAP (Average Precision at K)
            try:
                total_relevant = np.sum(binary_true)
                if total_relevant == 0:
                    ap = 0.0
                else:
                    ap = 0.0
                    num_pos = 0
                    for i in range(1, min(k, len(binary_true)) + 1):
                        if binary_true[i - 1] == 1:
                            num_pos += 1
                            ap += num_pos / i
                    ap /= min(total_relevant, k)
            except Exception:
                ap = 0.0

            # Precision@K
            try:
                sliced_len = len(binary_true[:k])
                precision = np.sum(binary_true[:k]) / sliced_len if sliced_len > 0 else 0.0
            except Exception:
                precision = 0.0

            # Recall@K
            try:
                total_relevant = np.sum(binary_true)
                recall = np.sum(binary_true[:k]) / total_relevant if total_relevant > 0 else 0.0
            except Exception:
                recall = 0.0

            metrics.append([ndcg, ap, precision, recall])
        return metrics

    def users_mean_std_rank_metrics(self, candidates_norm_df: pd.DataFrame, k: int, conf_threshold: float = -1) -> tuple:
        users_scores = self.rank_metrics(candidates_norm_df, k, conf_threshold)
        if not users_scores:
            return np.zeros(4), np.zeros(4)
        scores = np.array(users_scores)

        mean_metrics = np.mean(scores, axis=0)
        std_metrics = np.std(scores, axis=0)

        return mean_metrics, std_metrics
