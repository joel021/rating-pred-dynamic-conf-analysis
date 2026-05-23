import pandas as pd

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


class CsvReader:

    def __init__(self, dataset_info: DatasetInfo):
        self.info = dataset_info

    def read(self):
        file_uri = f"{self.info.root_uri}/data/{self.info.database_name}/{self.info.interactions_file}"
        header = 0 if self.info.has_head else None
        
        df = pd.read_csv(file_uri, sep=self.info.sep, header=header)
        if not self.info.has_head:
            df.columns = self.info.columns
        return df

    def read_items(self):
        file_uri = f"{self.info.root_uri}/data/{self.info.database_name}/{self.info.items_file}"
        return pd.read_csv(file_uri)
