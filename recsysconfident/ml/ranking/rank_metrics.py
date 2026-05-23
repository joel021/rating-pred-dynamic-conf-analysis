"""
package: recsysconfident.ml.ranking.conf_aware_rank_metrics
"""
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, average_precision_score, recall_score

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


class ConfAwareRankingMetrics:

    def __init__(self, data_info: DatasetInfo, r_t: float=0.75, alpha: float = 5):
        self.data_info = data_info
        self.r_t = r_t
        self.alpha = alpha

    def binarize(self, relevances):
        return (relevances >= self.r_t).astype(int)

    def _get_true_pred_scores(self, df: pd.DataFrame) -> dict:

        user_true_pred_scores = (
            df.groupby(self.data_info.user_col)
            .apply(lambda x: (
                x.sort_values(by=self.data_info.r_pred_col, ascending=False)
                .loc[:, [self.data_info.relevance_col, self.data_info.r_pred_col]]
                .pipe(lambda s: (s[self.data_info.relevance_col].values, s[self.data_info.r_pred_col].values))
            ))
            .to_dict()
        )
        return user_true_pred_scores

    def conf_filter(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        if self.data_info.conf_pred_col in df.columns:
            return df[df[self.data_info.conf_pred_col] >= threshold]
        return df

    def rank_metrics(self, norm_df: pd.DataFrame, k: int, conf_threshold: float = -1) -> list:
        from sklearn.metrics import precision_score
        
        if conf_threshold >= 0:
            norm_df = self.conf_filter(norm_df, conf_threshold)

        user_true_pred_scores = self._get_true_pred_scores(norm_df)
        metrics = []
        for user_key in user_true_pred_scores.keys():
            true_ratings, pred_ratings = user_true_pred_scores[user_key]
            binary_pred = self.binarize(pred_ratings)
            binary_true = self.binarize(true_ratings)
            
            # NDCG
            try:
                ndcg = ndcg_score([true_ratings], [pred_ratings], k=k)
            except Exception:
                ndcg = 0.0

            # MAP
            try:
                ap = average_precision_score(binary_true[:k], binary_pred[:k])
                if np.isnan(ap):
                    ap = 0.0
            except Exception:
                ap = 0.0

            # Precision
            try:
                precision = precision_score(binary_true[:k], binary_pred[:k], zero_division=0)
            except Exception:
                precision = 0.0

            # Recall
            try:
                recall = recall_score(binary_true[:k], binary_pred[:k], zero_division=0)
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
