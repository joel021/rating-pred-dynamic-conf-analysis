import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO


from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import gp_data_dl
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


def get_cgprank_and_dataloader(info: DatasetInfo, fold):

    fit_dataloader, eval_dataloader, inducing_points, num_data = gp_data_dl(info, fold)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CGPRankRatingPred(
                        inducing_points=inducing_points,
                        device = device,
                        r_min = info.rate_range[0],
                        r_max = info.rate_range[1],
                        n_users=info.n_users,
                        n_items=info.n_items,
                        num_data=num_data
                )
    
    return model, fit_dataloader, eval_dataloader


class CGPRankRatingPred(ApproximateGP):

    def __init__(self, inducing_points, n_users, n_items, r_min, r_max, device, rank=8, num_data=1_000_000):
        self.device = device
        inducing_points = inducing_points.to(device)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        ).to(device)

        super().__init__(variational_strategy)

        # Likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        # ELBO
        self.mll = VariationalELBO(self.likelihood, self, num_data=num_data).to(device)

        # Mean / Kernel
        self.mean_module = gpytorch.means.ConstantMean().to(device)

        self.user_kernel = gpytorch.kernels.IndexKernel(num_tasks=n_users+1, rank=rank).to(device)
        self.item_kernel = gpytorch.kernels.IndexKernel(num_tasks=n_items+1, rank=rank).to(device)

        self.covar_module = self.user_kernel * self.item_kernel

        # Rating normalization
        self.r_min = r_min
        self.r_max = r_max

    def to(self, device):

        if not (self.device == device):
            self.device = device
            return self.to(device)

        return self

    def train(self, mode=True, *args):

        if mode and not self.training:
            self.training = True
            return self.train(mode)
        
        elif not mode and self.training:
            self.likelihood.train(False)

            self.mll.train(False)
            self.mean_module.train(False)
            self.user_kernel.train(False)
            self.item_kernel.train(False)
            self.covar_module.train(False)

    def eval(self):

        if self.training:
            self.train(False)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def _normalize(self, y):
        return (y - self.r_min) / (self.r_max - self.r_min)

    def _denormalize(self, y):
        return y * (self.r_max - self.r_min) + self.r_min

    def loss(self, u_ids, i_ids, labels, optimizer):
        self.train()

        optimizer.zero_grad()

        inputs = torch.stack([u_ids, i_ids], dim=-1).float()

        labels = self._normalize(labels)

        output = self.forward(inputs)

        loss = -self.mll(output, labels)
        loss.backward()
        optimizer.step()

        return loss

    def eval_loss(self, u_ids, i_ids, labels):
        self.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            inputs = torch.stack([u_ids, i_ids], dim=-1).float()

            output = self(inputs)
            preds = self.likelihood(output).mean

            preds_denorm = self._denormalize(preds)

            mse = torch.mean((preds_denorm - labels) ** 2)

        return mse

    def predict(self, user_ids, item_ids):
        self.eval()
        self.likelihood.eval()

        inputs = torch.stack([user_ids, item_ids], dim=-1).long()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = self.forward(inputs)
            preds = self.likelihood(output)

            mean = preds.mean
            mean = self._denormalize(mean)

            var = preds.variance
            return mean, 1 / (1 + var)

    def predict_top_k(self, u_id, k=10, beta=2.0):
        """
        CGPRANK-style selection using:
        - UCB
        - Sequential hallucinated updates via GP conditioning
        """

        self.eval()
        self.likelihood.eval()

        all_item_ids = torch.arange(self.n_items, device=device)
        device = u_id.device

        user_tensor = torch.full((len(all_item_ids), 1), u_id, device=device)
        candidates = torch.cat([user_tensor, all_item_ids.unsqueeze(-1)], dim=-1).float()

        selected_indices = []

        # Start with base model
        fantasy_model = self

        for _ in range(k):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = fantasy_model.likelihood(fantasy_model(candidates))

                mu = posterior.mean
                sigma = posterior.variance.sqrt()

            # UCB scoring
            scores = mu + (beta ** 0.5) * sigma

            # Mask selected
            if selected_indices:
                scores[selected_indices] = -float("inf")

            best_idx = torch.argmax(scores).item()
            selected_indices.append(best_idx)

            x_new = candidates[best_idx].unsqueeze(0)
            y_new = mu[best_idx].unsqueeze(0)  # use predicted mean

            # Condition model on fantasy observation
            fantasy_model = fantasy_model.condition_on_observations(x_new, y_new)

        return all_item_ids[selected_indices]
