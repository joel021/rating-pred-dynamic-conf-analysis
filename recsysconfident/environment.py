import json
import os

from recsysconfident.data_handling.datasets.csv_reader import CsvReader
from recsysconfident.ml.models.cgp_rank import get_cgprank_and_dataloader
from recsysconfident.ml.models.dropout_uncertainty_model import get_MCDropoutRecModel_and_dataloader
import torch

from recsysconfident.data_handling.datasets.amazon_products import AmazonProductsReader
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.data_handling.datasets.movie_lens_reader import MovieLensReader
from recsysconfident.ml.models.distribution_based.lightgcn_conf import get_lightgcn_conf_model_and_dataloader
from recsysconfident.ml.models.distribution_based.cp_mf import get_cpmf_model_and_dataloader


class Environment:

    def __init__(self, model_name: str,
                 database_name: str,
                 instance_dir: str,
                 batch_size: int = 1024,
                 split_position: int = -1,
                 root_path:str="./",
                 min_inter_per_user: int=10):
        self.work_dir: str = None
        self.dataset_info: DatasetInfo = None
        self.batch_size = batch_size
        self.model_name = model_name
        self.database_name = database_name
        self.split_position = split_position
        self.root_path = root_path
        self.min_inter_per_user = min_inter_per_user

        self.load_df_info()
        self.instance_dir = instance_dir
        self.model_uri = f"{self.instance_dir}/model-{self.split_position}.pth"

        self.setup_splits_path()

    def setup_splits_path(self):

        os.makedirs(name=f"{self.root_path}/runs", exist_ok=True)
        splits = os.listdir(f"{self.root_path}/runs")
        if self.split_position == -1:
            self.split_position = len(splits)

        os.makedirs(name=f"{self.root_path}/runs/data_splits/{self.database_name}/{self.split_position}", exist_ok=True)

    def load_df_info(self):

        if os.path.isfile(f"{self.root_path}/data/{self.database_name}/info.json"):

            with open(f"{self.root_path}/data/{self.database_name}/info.json") as f:
                info = json.load(f)
            self.dataset_info = DatasetInfo(**info, database_name=self.database_name, batch_size=self.batch_size, root_uri=self.root_path)
        else:
            raise FileNotFoundError("Info file does not exists. Check if the dataset name is correct.")

    def read_split_datasets(self, shuffle: bool):

        self.database_name_fn = {
            "ml-1m": MovieLensReader(self.dataset_info).read,
            "amazon-movies-tvs": AmazonProductsReader(self.dataset_info).read,
            "netflix-prize": CsvReader(self.dataset_info).read,
        }

        self.model_name_fn = {
            "dropout": get_MCDropoutRecModel_and_dataloader,
            "cpmf": get_cpmf_model_and_dataloader,
            "prlightgcn": get_lightgcn_conf_model_and_dataloader,
            "cgprank": get_cgprank_and_dataloader
        }

        if not self.database_name in self.database_name_fn:
            raise FileNotFoundError(f"Database {self.database_name} does not exist.")

        ratings_df = self.database_name_fn[self.database_name]()
        items_df = None
        if self.dataset_info.metadata_columns:
            items_df = CsvReader(self.dataset_info).read_items()

            not_data_items = set(ratings_df[self.dataset_info.item_col].unique()) - set(
                items_df[self.dataset_info.item_col].unique())
            if len(not_data_items) > 0:
                print(f"Warning: {len(not_data_items)} items in ratings are missing from items_df metadata.")

        self.dataset_info.build(ratings_df, items_df, shuffle)
        print(f"Gathered dataset with {len(self.dataset_info.ratings_df)} interactions, {self.dataset_info.n_users} users"
              f" and {self.dataset_info.n_items} items.")

        print("Interactions dataset built.")
        return self

    def get_model_dataloaders(self, shuffle: bool, fold) -> tuple:

        self.read_split_datasets(shuffle)
        if not self.model_name in self.model_name_fn:
            raise ValueError(f"Invalid model name: {self.model_name}")

        model, fit_dl, val_dl = self.model_name_fn[self.model_name](self.dataset_info, fold)

        if os.path.isfile(self.model_uri):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(self.model_uri, weights_only=True, map_location=device))
            print(f"Loaded model weights from {self.model_uri}")

        return model, fit_dl, val_dl
