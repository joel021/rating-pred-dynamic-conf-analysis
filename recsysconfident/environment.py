import json
import os
import torch

from recsysconfident.data_handling.datasets.csv_reader import CsvReader
from recsysconfident.ml.models.cgp_rank import get_cgprank_and_dataloader
from recsysconfident.ml.models.dropout_uncertainty_model import get_MCDropoutRecModel_and_dataloader

from recsysconfident.data_handling.datasets.amazon_products import AmazonProductsReader
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.data_handling.datasets.movie_lens_reader import MovieLensReader
from recsysconfident.ml.models.distribution_based.lightgcn_conf import get_lightgcn_conf_model_and_dataloader
from recsysconfident.ml.models.distribution_based.cp_mf import get_cpmf_model_and_dataloader
from recsysconfident.ml.models.k_nearest_neighbors import get_knn_cosine_basic, get_knn_pearson_baseline_basic


class Environment:

    def __init__(self, model_name: str,
                 database_name: str,
                 split_position,
                 batch_size: int = 1024,
                 root_path:str="./",
                 min_inter_per_user: int=10):
        self.work_dir: str = None
        self.dataset_info: DatasetInfo = None
        self.batch_size = batch_size
        self.model_name = model_name
        self.database_name = database_name
        self.root_path = root_path
        self.min_inter_per_user = min_inter_per_user
        self.instance_dir = None
        self.split_position = split_position
        
        self.setup_instance_dir(None)

        self.load_df_info()
        self.model_uri = None

    def set_split_position(self, split_position):
        self.split_position = split_position
        self.setup_instance_dir(None)

    def setup_instance_dir(self, instance_dir: str):

        if instance_dir is None:
            self.work_dir = f"./runs/{self.database_name}-{self.model_name}"
            instance_dir = f"{self.work_dir}-{self.split_position}"

        self.instance_dir = instance_dir
        os.makedirs(name=instance_dir, exist_ok=True)

    def load_df_info(self):
        
        if os.path.isfile(f"{self.root_path}/data/{self.database_name}/info.json"):

            with open(f"{self.root_path}/data/{self.database_name}/info.json") as f:
                info = json.load(f)
            run_data_uri = f"{self.root_path}/runs/data/{self.database_name}"
            os.makedirs(name=run_data_uri, exist_ok=True)
            self.dataset_info = DatasetInfo(**info, run_data_uri=run_data_uri,
                                            database_name=self.database_name, 
                                            batch_size=self.batch_size, root_uri=self.root_path)
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
            "knn-cosine-basic": get_knn_cosine_basic,
            "knn-pearson-baseline": get_knn_pearson_baseline_basic
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

    def get_model_dataloaders(self, shuffle: bool) -> tuple:
        
        self.model_uri = f"{self.instance_dir}/model-{self.split_position}.pth"
        self.read_split_datasets(shuffle)
        if not self.model_name in self.model_name_fn:
            raise ValueError(f"Invalid model name: {self.model_name}")

        model, fit_dl, val_dl = self.model_name_fn[self.model_name](self.dataset_info, self.split_position)

        if os.path.isfile(self.model_uri):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(self.model_uri, weights_only=True, map_location=device))
            print(f"Loaded model weights from {self.model_uri}")

        return model, fit_dl, val_dl
