import torch
import torch.nn as nn
from torch_betainc import betainc

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.ml.losses import SoftHistogramWasserstein
from recsysconfident.ml.models.torchmodel import TorchModel

def beta_cdf(x_batch, alpha, beta, eps=1e-7):
    x = x_batch.clamp(eps, 1 - eps)
    a = alpha.unsqueeze(-1)
    b = beta.unsqueeze(-1)
    return betainc(a, b, x)

def get_lbd_wasserstein_model_and_dataloader(info: DatasetInfo, fold: int):
    fit_dataloader, eval_dataloader = ui_ids_label(info, fold)

    if not (info.rate_range is None) and len(info.rate_range) == 3:
        Rmin, Rmax, Rstep = info.rate_range
        n_ratings = int(round((Rmax - Rmin) / Rstep)) + 1 if Rmin == 0.0 else int(round((Rmax - Rmin) / Rstep))
    else:
        n_ratings = 10

    model = LBD(
        num_users=info.n_users,
        num_items=info.n_items,
        num_hidden=512,
        n_ratings=n_ratings,
        rmax=info.rate_range[1],
        rmin=info.rate_range[0]
    )

    return model, fit_dataloader, eval_dataloader

class LBD(TorchModel):
    def __init__(self, num_users: int, num_items: int, num_hidden: int, n_ratings: int, rmax: float = 5.0, rmin: float = 0.0):
        super().__init__(None)

        self.num_users = num_users
        self.num_items = num_items
        self.num_hidden = num_hidden
        self.n_ratings = n_ratings
        self.rmax = torch.scalar_tensor(rmax)
        self.rmin = torch.scalar_tensor(rmin)
        self.R_step = (rmax - rmin) / (n_ratings - 1) if n_ratings > 1 else 1.0

        self.uid_features = nn.Embedding(num_users + 1, num_hidden)
        self.iid_features = nn.Embedding(num_items + 1, num_hidden)

        self.a = nn.Embedding(num_users + num_items + 1, 1)
        self.b = nn.Embedding(num_users + num_items + 1, 1)

        self.a_0 = nn.Parameter(torch.tensor(0.1))
        self.b_0 = nn.Parameter(torch.tensor(0.3))

        self.bin_a = nn.Embedding(num_users + 1, n_ratings)
        self.bin_b = nn.Embedding(num_items + 1, n_ratings)

        self.epslon = torch.scalar_tensor(0.001)
        self.initialize_weights()
        self.soft_hist_wasserstein = SoftHistogramWasserstein(self.rmin.item(), self.rmax.item())

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.uid_features.weight)
        nn.init.xavier_uniform_(self.iid_features.weight)
        nn.init.xavier_uniform_(self.a.weight)
        nn.init.xavier_uniform_(self.b.weight)
        nn.init.xavier_uniform_(self.bin_a.weight)
        nn.init.xavier_uniform_(self.bin_b.weight)

    def forward(self, u_ids, i_ids):
        U = self.uid_features(u_ids)
        V = self.iid_features(i_ids)

        mu = 0.5 * (1 + nn.functional.cosine_similarity(U, V, dim=-1))
        nu = torch.norm(U + V, dim=1)

        alpha = nu * mu
        beta = nu * (1 - mu)

        a_user = self.a(u_ids).squeeze(-1)
        a_item = self.a(i_ids + self.num_users).squeeze(-1)
        b_user = self.b(u_ids).squeeze(-1)
        b_item = self.b(i_ids + self.num_users).squeeze(-1)

        alpha_prime = torch.maximum(self.a_0 + a_user + a_item + alpha, self.epslon)
        beta_prime = torch.maximum(self.b_0 + b_user + b_item + beta, self.epslon)

        theta_u = self.bin_a(u_ids)
        theta_i = self.bin_b(i_ids)
        w_logits = theta_u + theta_i

        mu_conf = torch.stack([mu, nu, alpha_prime, beta_prime], dim=1)

        return mu_conf, w_logits

    def regularization(self):
        U_l2 = torch.norm(self.uid_features.weight, p=2) ** 2
        V_l2 = torch.norm(self.iid_features.weight, p=2) ** 2
        a_l2 = torch.norm(self.a.weight, p=2) ** 2
        b_l2 = torch.norm(self.b.weight, p=2) ** 2
        bin_a_l2 = torch.norm(self.bin_a.weight, p=2) ** 2
        bin_b_l2 = torch.norm(self.bin_b.weight, p=2) ** 2

        return (U_l2 + V_l2 + a_l2 + b_l2 + bin_a_l2 + bin_b_l2) * 0.0001

    def get_pmf(self, user_ids, item_ids):
        outputs, w_logits = self.forward(user_ids, item_ids)
        alpha = torch.clamp(outputs[:, 2], min=self.epslon, max=10000.0)
        beta = torch.clamp(outputs[:, 3], min=self.epslon, max=10000.0)

        W_ij_r = torch.softmax(w_logits, dim=-1)

        zero_edge = torch.zeros_like(W_ij_r[:, :1])
        normalized_edges = torch.cat(
            [zero_edge, torch.cumsum(W_ij_r, dim=1)], dim=1
        )

        cdf_at_edges = beta_cdf(normalized_edges, alpha, beta)
        bin_probs = cdf_at_edges[:, 1:] - cdf_at_edges[:, :-1]

        return bin_probs, outputs

    def predict(self, user, item):
        bin_probs, outputs = self.get_pmf(user, item)
        nu = outputs[:, 1]
        
        R = torch.linspace(self.rmin.item(), self.rmax.item(), self.n_ratings, device=bin_probs.device)
        ratings = torch.sum(bin_probs * R.unsqueeze(0), dim=1)
        
        return ratings, nu

    def eval_loss(self, user_ids, item_ids, true_labels):
        bin_probs, _ = self.get_pmf(user_ids, item_ids)
        R = torch.linspace(self.rmin.item(), self.rmax.item(), self.n_ratings, device=bin_probs.device)
        ratings = torch.sum(bin_probs * R.unsqueeze(0), dim=1)
        
        return torch.sqrt(torch.nn.functional.mse_loss(ratings, true_labels, reduction='mean'))

    def loss(self, user_ids, item_ids, labels, optimizer):
        optimizer.zero_grad()

        bin_probs, _ = self.get_pmf(user_ids, item_ids)
        label_index = torch.round((labels - self.rmin) / self.R_step).long().clamp(0, self.n_ratings - 1)

        true_bin_probs = bin_probs.gather(1, label_index.unsqueeze(1)).squeeze(1)
        true_bin_probs = torch.clamp(true_bin_probs, min=1e-10, max=1.0)
        loss = -torch.log(true_bin_probs).mean()
        
        R = torch.linspace(self.rmin.item(), self.rmax.item(), self.n_ratings, device=bin_probs.device)
        ratings = torch.sum(bin_probs * R.unsqueeze(0), dim=1)

        loss = loss + self.regularization() + self.soft_hist_wasserstein(ratings, labels)

        loss.backward()
        optimizer.step()

        return loss
