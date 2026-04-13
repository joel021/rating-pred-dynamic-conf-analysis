import torch
import torch.nn as nn

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.simple_confidence.simple_conf_model import SimpleConfModel


def get_dmf_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = DMF(info.n_users,
                 info.n_items,
                 64,
                 1,
                 rmin=info.rate_range[0],
                 rmax=info.rate_range[1])
    return model, fit_dataloader, eval_dataloader, test_dataloader

class DMF(SimpleConfModel):

    def __init__(self, n_users: int, n_items: int, emb_dim: int, hidden_layers: int, rmin: float, rmax: float):
        super().__init__()
        self.n_users = n_users
        self.rmin = rmin
        self.rmax = rmax

        self.user_embed = nn.Embedding(n_users, emb_dim)
        self.item_embed = nn.Embedding(n_items, emb_dim)

        nn.init.xavier_uniform(self.user_embed.weight)
        nn.init.xavier_uniform(self.item_embed.weight)

        user_modules = []
        item_modules = []
        input_dim = emb_dim

        for h in hidden_layers:
            user_modules.append(nn.Linear(input_dim, h))
            user_modules.append(nn.ReLU())
            item_modules.append(nn.Linear(input_dim, h))
            item_modules.append(nn.ReLU())
            input_dim = h

        self.user_net = nn.Sequential(*user_modules)
        self.item_net = nn.Sequential(*item_modules)
        self.criterion = nn.MSELoss()

    def forward(self, user_indices, item_indices):
        u = self.user_embed(user_indices)
        v = self.item_embed(item_indices)

        p = self.user_net(u)
        q = self.item_net(v)

        dot_product = torch.sum(p * q, dim=1)
        norm_p = torch.norm(p, p=2, dim=1)
        norm_q = torch.norm(q, p=2, dim=1)

        cosine_sim = dot_product / (norm_p * norm_q + 1e-8)
        r_pred = ((cosine_sim + 1) / 2) * (self.rmax - self.rmin) + self.r_min

        return torch.stack([r_pred, torch.zeros_like(r_pred)], dim=1)

    def regularization(self):
        return 0
