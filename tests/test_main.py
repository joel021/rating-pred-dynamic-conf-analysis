import glob
import json
import os
import shutil
import unittest
import pandas as pd

from recsysconfident.setup import Setup
from main import main


class TestMainFlow(unittest.TestCase):

    def setUp(self):
        self.db_name = "ml-100k-test"
        self.model_name = "mf"
        
        # Clean up existing runs/data
        self.cleanup_paths()

        # Create dummy ml-100k dataset
        os.makedirs(f"./data/{self.db_name}", exist_ok=True)
        
        # We need each user to have at least 15 interactions in test_fold_df so they survive splitting & filtering (15 >= 10)
        # Generate dummy ratings data: 3 users, 200 total items, each user rates 100 items
        ratings_data = []
        # User 1 rates items 1 to 100
        for item_id in range(1, 101):
            ratings_data.append([1, item_id, 4.0 if item_id % 2 == 0 else 3.0, 1000 + item_id])
        # User 2 rates items 51 to 150
        for item_id in range(51, 151):
            ratings_data.append([2, item_id, 4.0 if item_id % 2 == 0 else 3.0, 1000 + item_id])
        # User 3 rates items 101 to 200
        for item_id in range(101, 201):
            ratings_data.append([3, item_id, 4.0 if item_id % 2 == 0 else 3.0, 1000 + item_id])
                
        df = pd.DataFrame(ratings_data, columns=["userId", "movieId", "rating", "timestamp"])
        df.to_csv(f"./data/{self.db_name}/ratings.dat", sep="\t", index=False, header=False)

        # Create info.json
        info = {
            "user_col": "userId",
            "item_col": "movieId",
            "rating_col": "rating",
            "timestamp_col": "timestamp",
            "columns": ["userId", "movieId", "rating", "timestamp"],
            "interactions_file": "ratings.dat",
            "sep": "\t",
            "has_head": False,
            "rate_range": [1.0, 5.0, 1.0],
            "metadata_columns": None
        }
        with open(f"./data/{self.db_name}/info.json", "w") as f:
            json.dump(info, f, indent=4)

    def tearDown(self):
        # Clean up files created during the test
        self.cleanup_paths()
        if os.path.exists(f"./data/{self.db_name}"):
            shutil.rmtree(f"./data/{self.db_name}")

    def cleanup_paths(self):
        # Remove run instance folders
        run_dirs = glob.glob(f"./runs/{self.db_name}-{self.model_name}-*")
        for d in run_dirs:
            if os.path.isdir(d):
                shutil.rmtree(d)
                
        # Remove run data folders
        db_data_dir = f"./runs/data/{self.db_name}"
        if os.path.exists(db_data_dir):
            shutil.rmtree(db_data_dir)

    def test_main_execution(self):
        setup_dict = {
            "model_name": self.model_name,
            "database_name": self.db_name,
            "folds": 3,  # Use 3 folds for quick testing
            "batch_size": 8,
            "patience": 1,
            "learning_rate": 0.01,
            "num_negatives": 2,
            "reevaluate": False
        }
        
        setup = Setup(**setup_dict)
        
        # Run main workflow
        main(setup)
        
        # Verify that output files were generated for fold 0 and fold 1 (setup.folds - 1)
        for fold in range(2):
            instance_dir = f"./runs/{self.db_name}-{self.model_name}-{fold}"
            
            # Check setup JSON
            setup_file = f"{instance_dir}/setup-{fold}.json"
            self.assertTrue(os.path.isfile(setup_file), f"Setup file missing: {setup_file}")
            
            # Check metrics JSON
            metrics_file = f"{instance_dir}/metrics-{fold}.json"
            self.assertTrue(os.path.isfile(metrics_file), f"Metrics file missing: {metrics_file}")
            
            # Check eval error CSV
            eval_file = f"{instance_dir}/eval_error_conf-{fold}.csv"
            self.assertTrue(os.path.isfile(eval_file), f"Eval error file missing: {eval_file}")

    def test_main_execution_num_negatives_none(self):
        setup_dict = {
            "model_name": self.model_name,
            "database_name": self.db_name,
            "folds": 3,
            "batch_size": 8,
            "patience": 1,
            "learning_rate": 0.01,
            "num_negatives": None,
            "reevaluate": False
        }
        
        setup = Setup(**setup_dict)
        
        # Run main workflow
        main(setup)
        
        # Verify that output files were generated for fold 0 and fold 1 (setup.folds - 1)
        for fold in range(2):
            instance_dir = f"./runs/{self.db_name}-{self.model_name}-{fold}"
            
            # Check setup JSON
            setup_file = f"{instance_dir}/setup-{fold}.json"
            self.assertTrue(os.path.isfile(setup_file), f"Setup file missing: {setup_file}")
            
            # Check metrics JSON
            metrics_file = f"{instance_dir}/metrics-{fold}.json"
            self.assertTrue(os.path.isfile(metrics_file), f"Metrics file missing: {metrics_file}")
            
            # Check eval error CSV
            eval_file = f"{instance_dir}/eval_error_conf-{fold}.csv"
            self.assertTrue(os.path.isfile(eval_file), f"Eval error file missing: {eval_file}")

    def test_main_execution_reevaluate(self):
        from unittest.mock import patch

        setup_dict = {
            "model_name": self.model_name,
            "database_name": self.db_name,
            "folds": 3,
            "batch_size": 8,
            "patience": 1,
            "learning_rate": 0.01,
            "num_negatives": 2,
            "reevaluate": False
        }
        
        setup = Setup(**setup_dict)
        
        # 1. Run main workflow with reevaluate=False to train the model first
        main(setup)
        
        # Verify model was indeed trained and saved
        for fold in range(2):
            instance_dir = f"./runs/{self.db_name}-{self.model_name}-{fold}"
            model_file = f"{instance_dir}/model-{fold}.pth"
            self.assertTrue(os.path.isfile(model_file), f"Model file missing: {model_file}")

        # 2. Run main workflow with reevaluate=True and patch/mock setup_fit to ensure it's not called
        setup_dict["reevaluate"] = True
        setup_reev = Setup(**setup_dict)

        with patch("main.setup_fit") as mock_setup_fit:
            main(setup_reev)
            # Ensure setup_fit was never called
            mock_setup_fit.assert_not_called()

        # Check that setup, metrics, and eval error files are still present
        for fold in range(2):
            instance_dir = f"./runs/{self.db_name}-{self.model_name}-{fold}"
            setup_file = f"{instance_dir}/setup-{fold}.json"
            self.assertTrue(os.path.isfile(setup_file))
            metrics_file = f"{instance_dir}/metrics-{fold}.json"
            self.assertTrue(os.path.isfile(metrics_file))
            eval_file = f"{instance_dir}/eval_error_conf-{fold}.csv"
            self.assertTrue(os.path.isfile(eval_file))

