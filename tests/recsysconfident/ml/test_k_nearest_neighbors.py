import unittest
import torch
import pandas as pd
import numpy as np
from recsysconfident.ml.models.neighborhood_based.k_nearest_neighbors import SparseKNNRecommender

class TestKNearestNeighbors(unittest.TestCase):
    def setUp(self):
        # Create a small dataset with 5 users and 4 items
        # Ratings:
        # User 0: [4.0, 0.0, 3.0, 5.0]
        # User 1: [0.0, 2.0, 4.0, 0.0]
        # User 2: [5.0, 3.0, 0.0, 2.0]
        # User 3: [1.0, 0.0, 2.0, 0.0]
        # User 4: [0.0, 0.0, 5.0, 4.0]
        self.n_users = 5
        self.n_items = 4
        
        ratings_data = [
            [0, 0, 4.0], [0, 2, 3.0], [0, 3, 5.0],
            [1, 1, 2.0], [1, 2, 4.0],
            [2, 0, 5.0], [2, 1, 3.0], [2, 3, 2.0],
            [3, 0, 1.0], [3, 2, 2.0],
            [4, 2, 5.0], [4, 3, 4.0]
        ]
        
        self.train_df = pd.DataFrame(ratings_data, columns=["userId", "movieId", "rating"])
        self.user_col = "userId"
        self.item_col = "movieId"
        self.rating_col = "rating"

    def test_msd_similarity_equivalence(self):
        # Compute using optimized SparseKNNRecommender
        recommender = SparseKNNRecommender(
            train_df=self.train_df,
            user_col=self.user_col,
            item_col=self.item_col,
            rating_col=self.rating_col,
            n_users=self.n_users,
            n_items=self.n_items,
            k=2,
            metric="msd",
            estimator="basic",
            device="cpu",
            chunk_size=2
        )
        
        optimized_sim = recommender.sim
        
        # Calculate MSD reference (slow loop-based implementation)
        R = recommender.R_dense
        mask = (R > 0).float()
        expected_sim = torch.zeros((self.n_users, self.n_users))
        
        for i in range(self.n_users):
            for k in range(self.n_users):
                if i == k:
                    expected_sim[i, k] = 0.0
                    continue
                
                common_mask = mask[i] * mask[k]
                common_count = common_mask.sum()
                if common_count > 0:
                    msd_val = ((R[i] - R[k]) ** 2 * common_mask).sum() / (common_count + 1e-9)
                    expected_sim[i, k] = 1.0 / (msd_val + 1.0)
                else:
                    expected_sim[i, k] = 0.0
                    
        # Check equivalence
        torch.testing.assert_close(optimized_sim, expected_sim, rtol=1e-5, atol=1e-5)

    def test_pearson_baseline_equivalence(self):
        # Compute using optimized SparseKNNRecommender
        recommender = SparseKNNRecommender(
            train_df=self.train_df,
            user_col=self.user_col,
            item_col=self.item_col,
            rating_col=self.rating_col,
            n_users=self.n_users,
            n_items=self.n_items,
            k=2,
            metric="pearson_baseline",
            estimator="baseline",
            shrinkage=10,
            device="cpu",
            chunk_size=2
        )
        
        optimized_sim = recommender.sim
        
        # Calculate Pearson Baseline reference (slow loop-based implementation)
        R = recommender.R_dense
        mask = (R > 0).float()
        mu = recommender.global_mean
        bu = recommender.user_means - mu
        bi = recommender.item_means - mu
        
        baseline = mu + bu.unsqueeze(1) + bi.unsqueeze(0)
        Xc = (R - baseline) * mask
        
        expected_sim = torch.zeros((self.n_users, self.n_users))
        
        for i in range(self.n_users):
            for k in range(self.n_users):
                if i == k:
                    expected_sim[i, k] = 0.0
                    continue
                
                common_mask = mask[i] * mask[k]
                common_count = common_mask.sum()
                
                # compute pearson baseline correlation
                # num = sum_{j in common} Xc[i, j] * Xc[k, j]
                # but wait! The baseline formula uses the centered ratings (Xc)
                num = (Xc[i] * Xc[k]).sum()
                den = torch.sqrt((Xc[i]**2).sum()) * torch.sqrt((Xc[k]**2).sum())
                rho = num / (den + 1e-9)
                
                shrink = (common_count - 1) / (common_count - 1 + 10)
                expected_sim[i, k] = shrink * rho
                
        # Check equivalence
        torch.testing.assert_close(optimized_sim, expected_sim, rtol=1e-5, atol=1e-5)

    def test_predictions_non_empty(self):
        recommender = SparseKNNRecommender(
            train_df=self.train_df,
            user_col=self.user_col,
            item_col=self.item_col,
            rating_col=self.rating_col,
            n_users=self.n_users,
            n_items=self.n_items,
            k=2,
            metric="msd",
            estimator="basic",
            device="cpu"
        )
        
        u_ids = torch.tensor([0, 1, 2], dtype=torch.int32)
        i_ids = torch.tensor([1, 0, 2], dtype=torch.int32)
        
        preds, certs = recommender.predict(u_ids, i_ids)
        
        self.assertEqual(preds.shape, (3,))
        self.assertEqual(certs.shape, (3,))
        self.assertTrue(torch.all(preds >= 0.0))
        self.assertTrue(torch.all(certs >= 0.0))
