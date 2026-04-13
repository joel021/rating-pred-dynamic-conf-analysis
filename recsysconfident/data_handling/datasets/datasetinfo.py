"""
package: recsysconfident.data_handling.datasets.datasets.py
"""
from typing import Any, Dict, List, Optional, Tuple

from pandas import DataFrame, read_csv, concat
import os
from recsysconfident.data_handling.splitting import time_ordered_folds
from recsysconfident.utils.datasets import filter_positives, map_ids


class DatasetInfo:


    def __init__(self, user_col: str, item_col: str, rating_col: str, interactions_file: str, columns: List[str],
                 rate_range: List[float], database_name: str, run_uri: str, metadata_columns: List[str],
                 folds: int=7,
                 items_file: Optional[str] = None, sep: str = ",", has_head: bool = False,
                 timestamp_col: Optional[str] = None, batch_size: int = 1024, root_uri: str = "./"):

        # Columns
        self.user_col: str = user_col
        self.item_col: str = item_col
        self.relevance_col: str = rating_col
        self.conf_pred_col: str = "conf_pred"
        self.r_pred_col: str = "r_pred"
        self.timestamp_col: Optional[str] = timestamp_col

        # Files and Paths
        self.root_uri: str = root_uri
        self.interactions_file: str = interactions_file
        self.items_file: Optional[str] = items_file
        self.database_name: str = database_name
        self.run_uri: str = run_uri

        # Dataset Structure Info
        self.columns: List[str] = columns
        self.metadata_columns: List[str] = metadata_columns
        self.rate_range: List[float] = rate_range
        self.sep: str = sep
        self.has_head: bool = has_head

        # DataFrames (Internal State)
        self.df_folds: Optional[List] = None
        self.items_df: Optional[DataFrame] = None
        self.items_per_user: Dict[Any, Tuple[set, list]] = {}

        # Sizing and Training Params
        self.n_users: int = 0
        self.n_items: int = 0
        self.batch_size: int = batch_size
        self.ratio_t: float = 0.78  # Threshold for filtering positives
        self.folds = folds

    def build(self, ratings_df: DataFrame, items_df: Optional[DataFrame], shuffle: bool) -> None:
        """
        Initializes the dataset state with raw dataframes, performs splitting,
        and computes initial user-item interaction sets.
        """
        self.ratings_df = ratings_df
        self.items_df = items_df

        self._split_interactions(shuffle)
        self.items_per_user = self._get_user_item_sets(self.ratings_df)

        print(f"{len(list(self.items_per_user.keys()))} mapped users sequentially!")


    def _split_interactions(self, shuffle: bool) -> "DatasetInfo":
        """
        Splits the interactions into fit, validation, and test sets.
        Loads existing splits if files are present.
        """
        items_path = f"{self.run_uri}/items.csv"

        os.makedirs(self.run_uri, exist_ok=True)

        if os.path.exists(f"{self.run_uri}/fold-0.csv") and os.path.exists(f"{self.run_uri}/fold-{self.folds-1}.csv"):
            # Load existing splits
            self.df_folds = []
            for fold in range(self.folds):
                self.df_folds.append(read_csv(f"{self.run_uri}/fold-{fold}.csv"))

            self.ratings_df = concat(self.df_folds, ignore_index=True)
            self.n_users = len(self.ratings_df[self.user_col].unique())
            self.n_items = len(self.ratings_df[self.item_col].unique())

            if os.path.exists(items_path):
                self.items_df = read_csv(items_path)
                self.items_df.set_index(self.item_col, inplace=True, drop=True)
        else:
            self.ratings_df = filter_positives(self.ratings_df, self.relevance_col, self.ratio_t)

            self.ratings_df, self.items_df = map_ids(self.ratings_df, self.items_df, self.user_col, self.item_col)
            self.n_users = len(self.ratings_df[self.user_col].unique())
            self.n_items = len(self.ratings_df[self.item_col].unique())

            if self.items_df is not None:
                self.items_df.to_csv(items_path, index=False)
                self.items_df.set_index(self.item_col, inplace=True, drop=True)

            self.df_folds = time_ordered_folds(self.ratings_df, self.timestamp_col, n_folds=self.folds, shuffle_within_folds=shuffle)
            
            for i in range(self.folds):
                self.df_folds[i].to_csv(f"{self.split_run_uri}/fold-{i}.csv", index=False)


        return self


    def _get_user_item_sets(self, df: DataFrame) -> Dict:
        """
        Computes a dictionary mapping users to their rated items and relevance scores.
        """
        user_item_dict = (
            df.groupby(self.user_col)
            .apply(lambda x: (set(x[self.item_col].tolist()), x[self.relevance_col].tolist()))
            .to_dict()
        )
        return user_item_dict

    # --- Public Accessors/Getters ---

    def get_splits(self) -> List[DataFrame]:
        """
        Returns the fit, validation, and test dataframes.
        """
        return self.df_folds
