import torch
import torch.nn as nn
import pandas as pd

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


def get_knn_cosine_basic(info: DatasetInfo, fold):

    train_folds_df = pd.concat(info.df_folds[:fold+1], ignore_index=True)

    test_fold_df = info.df_folds[fold+1]

    eval_data = [(torch.from_numpy(test_fold_df[info.user_col].values.astype(int)).int(),
                      torch.from_numpy(test_fold_df[info.item_col].values.astype(int)).int(),
                      torch.from_numpy(test_fold_df[info.relevance_col].values.astype(float)))]
    
    model = SparseKNNRecommender(
        train_data_df=train_folds_df,
        user_col=info.user_col,
        item_col=info.item_col,
        rating_col=info.relevance_col,
        n_users=info.n_users,
        n_items=info.n_items,
        metric='cosine',
        estimator='basic'
    )

    return model, [], eval_data

def get_knn_pearson_baseline_basic(info: DatasetInfo, fold):

    train_folds_df = pd.concat(info.df_folds[:fold+1], ignore_index=True)

    test_fold_df = info.df_folds[fold+1]

    eval_data = [(torch.from_numpy(test_fold_df[info.user_col].values.astype(int)).int(),
                      torch.from_numpy(test_fold_df[info.item_col].values.astype(int)).int(),
                      torch.from_numpy(test_fold_df[info.relevance_col].values.astype(float)))]
    
    model = SparseKNNRecommender(
        train_data_df=train_folds_df,
        user_col=info.user_col,
        item_col=info.item_col,
        rating_col=info.relevance_col,
        n_users=info.n_users,
        n_items=info.n_items,
        metric='pearson_baseline',
        estimator='baseline'
    )

    return model, [], eval_data


class SparseKNNRecommender():

    def __init__(self, train_data_df, user_col, item_col, rating_col, n_users, n_items, k=40, metric='cosine', estimator='basic', shr=100):
        self.train_data_df = train_data_df
        self.user_col = user_col
        self.item_col = item_col

        self.k = k
        self.n_items = n_items
        self.metric = metric
        self.estimator = estimator
        self.shr = shr
        
        u_ids = torch.from_numpy(train_data_df[user_col].values).long()
        i_ids = torch.from_numpy(train_data_df[item_col].values).long()
        vals = torch.from_numpy(train_data_df[rating_col].values).float()
        
        self.item_users = {item: [] for item in range(n_items)}
        self.item_ratings = {item: [] for item in range(n_items)}
        
        for u, item, r in zip(u_ids.tolist(), i_ids.tolist(), vals.tolist()):
            self.item_users[item].append(u)
            self.item_ratings[item].append(r)
            
        indices = torch.stack([u_ids, i_ids])
        self.train_sparse = torch.sparse_coo_tensor(
            indices, vals, (n_users, n_items)
        ).coalesce()
        
        self.global_mean = vals.mean()
        
        user_sums = torch.zeros(n_users)
        user_counts = torch.zeros(n_users)
        user_sums.scatter_add_(0, u_ids, vals)
        user_counts.scatter_add_(0, u_ids, torch.ones_like(vals))
        self.user_means = user_sums / (user_counts + 1e-9)
        
        user_sq_sums = torch.zeros(n_users)
        user_sq_sums.scatter_add_(0, u_ids, vals ** 2)
        user_var = (user_sq_sums / (user_counts + 1e-9)) - (self.user_means ** 2)
        self.user_stds = torch.sqrt(torch.clamp(user_var, min=0.0)) + 1e-9
        
        item_sums = torch.zeros(n_items)
        item_counts = torch.zeros(n_items)
        item_sums.scatter_add_(0, i_ids, vals)
        item_counts.scatter_add_(0, i_ids, torch.ones_like(vals))
        self.item_means = item_sums / (item_counts + 1e-9)

    def _compute_similarity(self, x, y, idx):
        mask = (x > 0) & (y > 0)
        
        if self.metric == 'cosine':
            x_m = x * mask
            y_m = y * mask
            num = (x_m * y_m).sum(dim=1)
            den = torch.sqrt((x_m**2).sum(dim=1)) * torch.sqrt((y_m**2).sum(dim=1))
            return num / (den + 1e-9)
            
        elif self.metric == 'msd':
            diff = (x - y) * mask
            msd_dist = (diff ** 2).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            return 1 / (msd_dist + 1)
            
        elif self.metric == 'pearson':
            mu_x = x.sum() / ((x > 0).sum() + 1e-9)
            mu_y = self.user_means[idx].unsqueeze(1)
            x_c = (x - mu_x) * mask
            y_c = (y - mu_y) * mask
            num = (x_c * y_c).sum(dim=1)
            den = torch.sqrt((x_c**2).sum(dim=1)) * torch.sqrt((y_c**2).sum(dim=1))
            return num / (den + 1e-9)
            
        elif self.metric == 'pearson_baseline':
            mu = self.global_mean
            b_u = mu + (x.sum() / ((x > 0).sum() + 1e-9) - mu) + (self.item_means - mu)
            b_v = mu + (self.user_means[idx].unsqueeze(1) - mu) + (self.item_means - mu)
            
            x_c = (x - b_u) * mask
            y_c = (y - b_v) * mask
            num = (x_c * y_c).sum(dim=1)
            den = torch.sqrt((x_c**2).sum(dim=1)) * torch.sqrt((y_c**2).sum(dim=1))
            rho = num / (den + 1e-9)
            
            n_corated = mask.sum(dim=1)
            return ((n_corated - 1) / (n_corated - 1 + self.shr)) * rho

        return torch.zeros(y.size(0))

    def forward(self, x_u, i):
        candidate_u_ids = self.item_users.get(i, [])
        
        if not candidate_u_ids:
            return torch.tensor(0.0)
            
        target_ratings = torch.tensor(self.item_ratings[i], dtype=torch.float32)
        idx = torch.tensor(candidate_u_ids, dtype=torch.long)
        
        candidate_matrix = torch.index_select(self.train_sparse, 0, idx).to_dense()
        
        similarities = self._compute_similarity(x_u.unsqueeze(0), candidate_matrix, idx)
        
        actual_k = min(self.k, len(candidate_u_ids))
        knn_sims, knn_top_indices = torch.topk(similarities, actual_k, largest=True)
        
        knn_ratings = target_ratings[knn_top_indices]
        knn_u_ids = idx[knn_top_indices]
        
        sim_sum = torch.sum(torch.abs(knn_sims)) + 1e-9
        
        std_ui = torch.std(knn_ratings, unbiased=False) + 1e-9
        certainty = 1.0 / (std_ui + 1.0)

        if self.estimator == 'basic':
            return torch.tensor([(knn_sims * knn_ratings).sum() / sim_sum, certainty])
            
        mu_u = x_u.sum() / ((x_u > 0).sum() + 1e-9)
        
        if self.estimator == 'means':
            mu_v = self.user_means[knn_u_ids]
            return torch.tensor([mu_u + (knn_sims * (knn_ratings - mu_v)).sum() / sim_sum, certainty])
            
        if self.estimator == 'zscore':
            x_u_mask = x_u > 0
            var_u = ((x_u[x_u_mask] - mu_u) ** 2).mean() if x_u_mask.any() else torch.tensor(0.0)
            std_u = torch.sqrt(var_u) + 1e-9
            
            mu_v = self.user_means[knn_u_ids]
            std_v = self.user_stds[knn_u_ids]
            
            return torch.tensor([mu_u + std_u * (knn_sims * ((knn_ratings - mu_v) / std_v)).sum() / sim_sum, certainty])
            
        if self.estimator == 'baseline':
            mu = self.global_mean
            b_i = self.item_means[i] - mu
            b_u = mu_u - mu
            b_ui = mu + b_u + b_i
            
            b_v = self.user_means[knn_u_ids] - mu
            b_vi = mu + b_v + b_i
            
            return torch.tensor([b_ui + (knn_sims * (knn_ratings - b_vi)).sum() / sim_sum, certainty])
            
        return torch.tensor([0.0, 0.0])

    def loss(self, *args):
        return nn.MSELoss()(torch.tensor([0]), torch.tensor([0]))

    def eval_loss(self, *args):
        return self.loss()
    
    def predict(self, u_ids, i_ids):
        batch_size = u_ids.size(0)
        results = torch.zeros((batch_size, 2))
        
        for b in range(batch_size):
            u = u_ids[b].item()
            i = i_ids[b].item()
            
            u_exists = u < self.user_means.size(0) and self.user_means[u] > 0
            i_exists = i < self.item_means.size(0) and self.item_means[i] > 0
            
            if not u_exists and not i_exists:
                results[b, 0] = self.global_mean
            elif not u_exists:
                results[b, 0] = self.item_means[i]
            elif not i_exists:
                results[b, 0] = self.user_means[u]
            else:
                x_u = torch.index_select(self.train_sparse, 0, torch.tensor([u], dtype=torch.long)).to_dense().squeeze(0)
                pred = self.forward(x_u, i)
                
                if pred[0] == 0.0:
                    results[b, 0] = self.user_means[u]
                else:
                    results[b] = pred
                    
        return results[:, 0], results[:, 1]

    def eval(self):
        return self
    
    def train(self, mode):
        return self
    
    def to(self, device):
        return self

    def train_method(self, **args):

        return {}

