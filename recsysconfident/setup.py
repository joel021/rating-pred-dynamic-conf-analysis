import json
import os
import time


class Setup:

    def __init__(self, model_name: str,
                 database_name: str,
                 folds: int=7,
                 split_position: int = 0,
                 fit_mode: int = 0,
                 batch_size: int = 1024,
                 learning_rate: float = 0.001,
                 patience: int = 5,
                 rate_range: list = None,
                 timestamp: str = None,
                 num_negatives=72,
                 reevaluate:bool = False,
                 hyperparameters: dict = None,
                 setup_name: str = None):

        self.folds = folds
        self.model_name = model_name
        self.database_name = database_name
        self.split_position = split_position
        self.fit_mode = fit_mode
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.num_negatives = num_negatives
        self.reevaluate = reevaluate
        self.hyperparameters = hyperparameters
        self.setup_name = setup_name

        self.set_rate_range(rate_range)

    def set_rate_range(self, rate_range: list[float]):

        if not rate_range:
            if os.path.isfile(f"./data/{self.database_name}/info.json"):
                with open(f"./data/{self.database_name}/info.json") as f:
                    info = json.load(f)
                self.rate_range = info.get("rate_range", None)

                if not self.rate_range:
                    raise Exception(f"No rate_range specified in {self.database_name}")
            else:
                raise Exception(f"No info.json found for {self.database_name}")
        else:
            self.rate_range = rate_range

    def to_dict(self) -> dict:
        res = {
            'model_name': self.model_name,
            'database_name': self.database_name,
            'folds': self.folds,
            'fit_mode': self.fit_mode,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'rate_range': self.rate_range,
            'num_negatives': self.num_negatives,
            'timestamp': time.strftime('%Y-%m-%d-%H-%M-%S')
        }
        if self.hyperparameters is not None:
            res['hyperparameters'] = self.hyperparameters
        if self.setup_name is not None:
            res['setup_name'] = self.setup_name
        return res
