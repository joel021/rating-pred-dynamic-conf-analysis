import unittest
import torch
import scipy.sparse as sp
from recsysconfident.setup import Setup
from recsysconfident.environment import Environment
from recsysconfident.ml.models.wasserstein_regularized.cp_mf import CPMF
from recsysconfident.ml.models.wasserstein_regularized.dropout_uncertainty_model import MCDropoutRecModel
from recsysconfident.ml.models.wasserstein_regularized.lbd import LBD
from recsysconfident.ml.models.wasserstein_regularized.lightgcn import LightGCN
from recsysconfident.ml.models.wasserstein_regularized.lightgcn_conf import PRLightGCN
from recsysconfident.ml.models.wasserstein_regularized.mf import MatrixFactorizationModel
from recsysconfident.ml.models.wasserstein_regularized.ord_rec_mf import OrdRec

class TestWassersteinLambda(unittest.TestCase):

    def test_setup_hyperparameters_propagation(self):
        # 1. Test Setup stores and serializes hyperparameters and setup_name
        setup_dict = {
            "model_name": "mf_wasserstein",
            "database_name": "ml-100k",
            "folds": 3,
            "hyperparameters": {
                "wasserstein_lambda": 0.5
            },
            "setup_name": "my_custom_setup_key"
        }
        setup = Setup(**setup_dict)
        self.assertEqual(setup.hyperparameters, {"wasserstein_lambda": 0.5})
        self.assertEqual(setup.setup_name, "my_custom_setup_key")
        
        serialized = setup.to_dict()
        self.assertIn("hyperparameters", serialized)
        self.assertEqual(serialized["hyperparameters"]["wasserstein_lambda"], 0.5)
        self.assertIn("setup_name", serialized)
        self.assertEqual(serialized["setup_name"], "my_custom_setup_key")

    def test_environment_hyperparameters_propagation(self):
        # 2. Test Environment receives hyperparameters and setup_name, and creates setup_name run directory
        env = Environment(
            model_name="mf_wasserstein",
            database_name="ml-100k",
            split_position=0,
            hyperparameters={"wasserstein_lambda": 0.75},
            setup_name="my_custom_setup_key"
        )
        self.assertEqual(env.hyperparameters, {"wasserstein_lambda": 0.75})
        self.assertEqual(env.setup_name, "my_custom_setup_key")
        self.assertEqual(env.work_dir, "./runs/my_custom_setup_key")
        self.assertEqual(env.instance_dir, "./runs/my_custom_setup_key-0")
        
        # Verify fallback logic
        env_fallback = Environment(
            model_name="mf_wasserstein",
            database_name="ml-100k",
            split_position=1
        )
        self.assertIsNone(env_fallback.setup_name)
        self.assertEqual(env_fallback.work_dir, "./runs/ml-100k-mf_wasserstein")
        self.assertEqual(env_fallback.instance_dir, "./runs/ml-100k-mf_wasserstein-1")

    def test_model_wasserstein_lambda_parsing(self):
        # 3. Test that each model correctly parses wasserstein_lambda and defaults to 1.0
        rate_range = [1.0, 5.0, 1.0]
        
        # Test CPMF
        cpmf_default = CPMF(num_users=10, num_items=10, latent_dim=5, rate_range=rate_range)
        self.assertEqual(cpmf_default.wasserstein_lambda, 1.0)
        cpmf_custom = CPMF(num_users=10, num_items=10, latent_dim=5, rate_range=rate_range, hyperparameters={"wasserstein_lambda": 2.5})
        self.assertEqual(cpmf_custom.wasserstein_lambda, 2.5)

        # Test MCDropoutRecModel
        dropout_default = MCDropoutRecModel(n_users=10, n_items=10, r_max=5.0, r_min=1.0, items_per_user={})
        self.assertEqual(dropout_default.wasserstein_lambda, 1.0)
        dropout_custom = MCDropoutRecModel(n_users=10, n_items=10, r_max=5.0, r_min=1.0, items_per_user={}, hyperparameters={"wasserstein_lambda": 0.2})
        self.assertEqual(dropout_custom.wasserstein_lambda, 0.2)

        # Test LBD
        lbd_default = LBD(num_users=10, num_items=10, num_hidden=5, n_ratings=5, rmin=1.0, rmax=5.0)
        self.assertEqual(lbd_default.wasserstein_lambda, 1.0)
        lbd_custom = LBD(num_users=10, num_items=10, num_hidden=5, n_ratings=5, rmin=1.0, rmax=5.0, hyperparameters={"wasserstein_lambda": 0.05})
        self.assertEqual(lbd_custom.wasserstein_lambda, 0.05)

        # Test LightGCN
        # Simple dummy graph structure
        adj_matrix = sp.csr_matrix((20, 20))
        lightgcn_default = LightGCN(Graph=adj_matrix, n_users=10, n_items=10, emb_dim=8, n_layers=2, keep_prob=1.0, A_split=False, rmin=1.0, rmax=5.0)
        self.assertEqual(lightgcn_default.wasserstein_lambda, 1.0)
        lightgcn_custom = LightGCN(Graph=adj_matrix, n_users=10, n_items=10, emb_dim=8, n_layers=2, keep_prob=1.0, A_split=False, rmin=1.0, rmax=5.0, hyperparameters={"wasserstein_lambda": 1.5})
        self.assertEqual(lightgcn_custom.wasserstein_lambda, 1.5)

        # Test PRLightGCN
        prlightgcn_default = PRLightGCN(Graph=adj_matrix, n_users=10, n_items=10, emb_dim=8, n_layers=2, keep_prob=1.0, A_split=False, rmin=1.0, rmax=5.0, step=1.0)
        self.assertEqual(prlightgcn_default.wasserstein_lambda, 1.0)
        prlightgcn_custom = PRLightGCN(Graph=adj_matrix, n_users=10, n_items=10, emb_dim=8, n_layers=2, keep_prob=1.0, A_split=False, rmin=1.0, rmax=5.0, step=1.0, hyperparameters={"wasserstein_lambda": 0.8})
        self.assertEqual(prlightgcn_custom.wasserstein_lambda, 0.8)

        # Test MatrixFactorizationModel
        mf_default = MatrixFactorizationModel(num_users=10, num_items=10, num_factors=8, rmin=1.0, rmax=5.0)
        self.assertEqual(mf_default.wasserstein_lambda, 1.0)
        mf_custom = MatrixFactorizationModel(num_users=10, num_items=10, num_factors=8, rmin=1.0, rmax=5.0, hyperparameters={"wasserstein_lambda": 3.14})
        self.assertEqual(mf_custom.wasserstein_lambda, 3.14)

        # Test OrdRec
        ordrec_default = OrdRec(num_users=10, num_items=10, num_factors=8, items_per_user={}, rmax=5.0, rmin=1.0)
        self.assertEqual(ordrec_default.wasserstein_lambda, 1.0)
        ordrec_custom = OrdRec(num_users=10, num_items=10, num_factors=8, items_per_user={}, rmax=5.0, rmin=1.0, hyperparameters={"wasserstein_lambda": 12.0})
        self.assertEqual(ordrec_custom.wasserstein_lambda, 12.0)
