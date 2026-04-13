"""
Using implementation found in https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/model.py
"""
from torch import nn
import torch

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.GCN_utils import get_adj_matrix, normalize_adj, scipy_to_torch_sparse
from recsysconfident.ml.models.simple_confidence.simple_conf_model import SimpleConfModel


def get_lightgcn_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    adj = get_adj_matrix(info.ratings_df, info)
    norm_adj = normalize_adj(adj)
    Graph = scipy_to_torch_sparse(norm_adj)

    model = LightGCN(
                Graph,
                info.n_users,
                 info.n_items,
                 64,
                 3,
                 keep_prob=0.6,
                 A_split=False,
                 rmin=info.rate_range[0],
                 rmax=info.rate_range[1])
    return model, fit_dataloader, eval_dataloader, test_dataloader

class LightGCN(SimpleConfModel):

    def __init__(self, Graph, n_users, n_items, emb_dim, n_layers, keep_prob: float, A_split, rmin, rmax, dropout=True):
        super(LightGCN, self).__init__()
        self.Graph = Graph
        self.rmax = rmax
        self.rmin = rmin
        self.num_users = n_users
        self.num_items = n_items
        self.latent_dim = emb_dim
        self.n_layers = n_layers
        self.keep_prob = keep_prob
        self.A_split = A_split
        self.dropout = dropout
        self.__init_weight()
        self.criterion = nn.MSELoss()
        print("Light GCN instantiated")

    def __init_weight(self):

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.coalesce().indices().t()
        values = x.coalesce().values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        self.Graph = self.Graph.to(self.embedding_item.weight.device)
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getEmbedding(self, users, pos_items, neg_items):

        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def forward(self, users, items):

        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return torch.stack([gamma, torch.zeros_like(gamma)], dim=1)

    def regularization(self):
        return 0