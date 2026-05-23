import unittest
import pandas as pd
import numpy as np

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.ranking.rank_metrics import ConfAwareRankingMetrics


class TestRankMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.df = pd.DataFrame({
            "userId": [1, 1, 1, 1, 1],
            "itemId": [1, 2, 3, 4, 5],
            "rating": np.asarray([1.0, 0.0, 0.0, 0.0, 1.0]),
            "r_pred": np.asarray([0.1, 0.05, 0.5, 0.6, 0.9]),
            "conf_pred": [1.0, 0.4, 0.5, 0.6, 0.1]
        }).sort_values(by="r_pred")
        data_info = DatasetInfo(user_col="userId",
                                item_col="itemId",
                                rating_col="rating",
                                interactions_file="ratings.csv",
                                columns=["userId","itemId", "rating","conf_pred"],
                                rate_range=[1,5],
                                database_name="ml-1m",
                                run_data_uri="./runs/data/ml-1m",
                                metadata_columns=None)
        self.rank_metrics = ConfAwareRankingMetrics(data_info)

    def test_rank_metrics_at_k_conf(self):

        users_scores = self.rank_metrics.rank_metrics(self.df, 3, 0.5)
        assert len(users_scores) == 1, "The number of users should 1."

    def test_rank_metrics_at_k(self):

        users_scores = self.rank_metrics.rank_metrics(self.df, 3, -1)
        assert len(users_scores) == 1, "The number of users should 1."

    def test_rank_metrics_all_metrics(self):

        users_scores = self.rank_metrics.rank_metrics(self.df, 3, -1)
        assert len(users_scores[0]) == 4, "There is 4 rank metrics"

    def test_users_mean_std_rank_metrics(self):
        mean, std = self.rank_metrics.users_mean_std_rank_metrics(self.df, 3, -1)
        assert len(mean) == 4, "There is mean of 4 rank metrics."

    def test_users_mean_std_rank_metrics_std(self):
        mean, std = self.rank_metrics.users_mean_std_rank_metrics(self.df, 3, -1)
        assert len(std) == 4, "There is std of 4 rank metrics."

    def test_conf_filter(self):

        filtered_df = self.rank_metrics.conf_filter(self.df, 0.5)
        print(filtered_df)

    def test_metric_values(self):
        users_scores = self.rank_metrics.rank_metrics(self.df, 3, -1)
        ndcg, ap, precision, recall = users_scores[0]
        self.assertAlmostEqual(precision, 1.0 / 3.0)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(ap, 0.5)
