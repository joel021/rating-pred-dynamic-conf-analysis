import torch
from torch_geometric.nn import GATConv
import torch.nn as nn

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.simple_confidence.simple_conf_model import SimpleConfModel


def get_ddgat_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    edges = torch.stack([torch.from_numpy(info.fit_df[info.user_col].values),
                         torch.from_numpy(info.fit_df[info.item_col].values)])

    model = DeepDGAT(info.n_users,
                 info.n_items,
                 emb_dim=64,
                 edge_index=edges,
                 heads=1,
                 rmin=info.rate_range[0],
                 rmax=info.rate_range[1])
    return model, fit_dataloader, eval_dataloader, test_dataloader

class DeepDGAT(SimpleConfModel):

    def __init__(self, n_users: int, n_items: int, emb_dim: int, edge_index, heads: int, rmin: float, rmax: float):
        super().__init__()
        self.n_users = n_users
        self.rmin = rmin
        self.rmax = rmax
        self.register_buffer('edge_index', edge_index)
        self.criterion = nn.MSELoss()

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform(self.user_emb.weight)
        nn.init.xavier_uniform(self.item_emb.weight)

        self.gat = GATConv(emb_dim, emb_dim, heads=heads, concat=False)

        self.rating_net = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user_indices, item_indices):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        x_updated = self.gat(x, self.edge_index)

        u_feat = x_updated[user_indices]
        i_feat = x_updated[item_indices + self.n_users]

        r_pred = self.rating_net(torch.cat([u_feat, i_feat], dim=1)).squeeze()
        return torch.stack([r_pred, torch.zeros_like(r_pred)], dim=1)

    def regularization(self):
        return 0
