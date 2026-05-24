import torch
import torch.nn as nn
import torch.distributions as d

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.torchmodel import TorchModel


def get_cpmf_wasserstein_model_and_dataloader(info: DatasetInfo, fold):

    fit_dataloader, eval_dataloader = ui_ids_label(info, fold)

    model = CPMF(
        num_users=info.n_users,
        num_items=info.n_items,
        latent_dim=20,
        rate_range=info.rate_range
    )

    return model, fit_dataloader, eval_dataloader


class CPMF(TorchModel):

    def __init__(self, num_users, num_items, latent_dim, rate_range: list):
        super().__init__(None)

        self.rmin = rate_range[0]
        self.rmax = rate_range[1]
        self.delta_r = rate_range[2] / 2

        # Latent factors
        self.user_factors = nn.Embedding(num_users, latent_dim)
        self.item_factors = nn.Embedding(num_items, latent_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor((self.rmin + self.rmax) / 2))

        # Variance parameters (γ_u, γ_v), initialized to 1.0
        self.user_gamma = nn.Embedding(num_users, 1)
        self.item_gamma = nn.Embedding(num_items, 1)
        nn.init.ones_(self.user_gamma.weight)
        nn.init.ones_(self.item_gamma.weight)

        self.alpha = nn.Parameter(torch.tensor(1.))

        # Regularization coefficients
        self.lambda_u = 0.001
        self.lambda_v = 0.001

        # Initialize factor and bias weights
        nn.init.xavier_uniform_(self.user_factors.weight)
        nn.init.xavier_uniform_(self.item_factors.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        self.switch_to_rating()

    def forward(self, user_ids, item_ids):
        u = self.user_factors(user_ids)  # (batch, k)
        v = self.item_factors(item_ids)  # (batch, k)
        dot = torch.sum(u * v, dim=1, keepdim=True)

        user_bias = self.user_bias(user_ids)
        item_bias = self.item_bias(item_ids)
        mean = (dot + user_bias + item_bias + self.global_bias).squeeze()

        # Softplus ensures γ > 0
        gamma_u = torch.clamp(self.user_gamma(user_ids), min=0.00001)
        gamma_v = torch.clamp(self.item_gamma(item_ids), min=0.00001)
        alpha = torch.exp(self.alpha)

        precision = alpha * gamma_u * gamma_v
        variance = 1.0 / precision
        std = torch.sqrt(variance).squeeze()

        return torch.stack([mean, std], dim=1)

    def loss(self, user_ids, item_ids, labels, optimizer):
        optimizer.zero_grad()
        pred_scores = self.forward(user_ids, item_ids)
        mu = pred_scores[:, 0]
        sigma = pred_scores[:, 1]
        nll = -d.Normal(mu, sigma).log_prob(labels).mean()

        loss = nll + self.regularization()
        loss.backward()
        optimizer.step()
        return loss

    def eval_loss(self, user_ids, item_ids, labels):
        pred_scores = self.forward(user_ids, item_ids)
        mu = pred_scores[:, 0]
        return torch.sqrt(torch.nn.functional.mse_loss(labels, mu, reduction='mean'))

    def regularization(self, user_ids=None, item_ids=None):
        reg = 0.0

        # L2 regularization on user and item factors/biases
        reg += self.lambda_u * torch.sum(self.user_factors.weight ** 2)
        reg += self.lambda_v * torch.sum(self.item_factors.weight ** 2)
        reg += 0.01 * torch.sum(self.user_bias.weight ** 2)
        reg += 0.01 * torch.sum(self.item_bias.weight ** 2)

        return reg

    def sharpe_ratio(self, R, sigma):
        return (R - 0.7 * self.rmax) / (sigma + 0.00001)

    def raking_predict(self, user_ids, item_ids):
        scores = self.forward(user_ids, item_ids)

        mu = scores[:, 0]
        sigma = scores[:, 1]
        dist = d.Normal(mu, sigma)

        confidence = dist.cdf(mu + self.delta_r) - dist.cdf(mu - self.delta_r)
        score = self.sharpe_ratio(mu, sigma)

        return score, confidence

    def rating_predict(self, user_ids, item_ids):
        scores = self.forward(user_ids, item_ids)

        mu = scores[:, 0]
        sigma = scores[:, 1]
        dist = d.Normal(mu, sigma)

        confidence = dist.cdf(mu + self.delta_r) - dist.cdf(mu - self.delta_r)

        return mu, confidence

    def switch_to_ranking(self):
        self.predict = self.raking_predict

    def switch_to_rating(self):
        self.predict = self.rating_predict
