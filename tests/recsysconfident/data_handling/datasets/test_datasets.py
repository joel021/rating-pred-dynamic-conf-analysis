import unittest
import pandas as pd
import os

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


class TestPositiveOnlyInteractions(unittest.TestCase):

    def setUp(self):

        script_dir = str(os.path.dirname(os.path.abspath(__file__)))
        ratings_uri = f'{str(script_dir[0:script_dir.index("tests")])}/tests/data/ratings-1m.csv'
        root_uri = f'{str(script_dir[0:script_dir.index("tests")])}/tests/data'
        self.dataset = DatasetInfo(
            user_col='userId',
            item_col='itemId',
            rating_col='rating',
            interactions_file='ratings-1m.csv',
            columns=['userId', 'itemId', 'rating'],
            rate_range=[1, 5, 1],
            database_name='test_db',
            run_data_uri=f'{root_uri}/test_db',
            metadata_columns=None,
            root_uri=root_uri,
            timestamp_col='timestamp'
        )
        self.ratings = pd.read_csv(ratings_uri)
        self.dataset.build(self.ratings, None, True)

    def test_all_in_fit_belong_to_items_per_users(self):
        users = self.dataset.items_per_user.keys()
        fit_users = set(self.dataset.df_folds[0]['userId'].unique())

        assert len(fit_users - users) == 0

    def test_all_in_val_belong_to_items_per_users(self):
        users = self.dataset.items_per_user.keys()
        val_users = set(self.dataset.df_folds[1]['userId'].unique())

        assert len(val_users - users) == 0

    def test_all_in_test_belong_to_items_per_users(self):
        users = self.dataset.items_per_user.keys()
        test_users = set(self.dataset.df_folds[2]['userId'].unique())

        assert len(test_users - users) == 0


