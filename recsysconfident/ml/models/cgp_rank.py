
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import VariationalELBO

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import gp_data_dl
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


def get_cgprank_and_dataloader(info: DatasetInfo, fold):

    fit_dataloader, eval_dataloader, inducing_points = gp_data_dl(info, fold)

    model = LargeScaleCGPRank(
                        inducing_points=inducing_points,
                        r_min = info.rate_range[0],
                        r_max = info.rate_range[1],
                        n_users=info.n_users,
                        n_items=info.n_items
                )
    
    return model, fit_dataloader, eval_dataloader


class LargeScaleCGPRank(ApproximateGP):

    def __init__(self, inducing_points, n_users, n_items, r_max, r_min, rank=8):
        # The VariationalStrategy manages the relationship between inducing points and the data

        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        
        self.r_min = r_min
        self.r_max = r_max

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Identity-based kernels for collaborative filtering
        self.user_kernel = gpytorch.kernels.IndexKernel(num_indices=n_users, rank=rank)
        self.item_kernel = gpytorch.kernels.IndexKernel(num_indices=n_items, rank=rank)
        self.covar_module = self.user_kernel * self.item_kernel
        
        # Define the MLL (Marginal Log Likelihood) objective for training
        self.mll = VariationalELBO(self.likelihood, self, num_data=10_000_000) 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def loss(self, u_ids, i_ids, labels, optimizer):
        self.train()
        self.likelihood.train()
        optimizer.zero_grad()
        
        normalized_labels = (labels - self.r_min) / (self.r_max - self.r_min)
        
        inputs = torch.stack([u_ids, i_ids], dim=-1)
        output = self(inputs)
        
        loss = -self.mll(output, normalized_labels)
        loss.backward()
        optimizer.step()
        return loss

    def eval_loss(self, u_ids, i_ids, labels):
        self.eval()
        self.likelihood.eval()
        
        normalized_labels = (labels - self.r_min) / (self.r_max - self.r_min)
        
        inputs = torch.stack([u_ids, i_ids], dim=-1)
        with torch.no_grad():
            output = self(inputs)
            loss = -self.mll(output, normalized_labels)
        return loss

    def predict(self, u_ids, i_ids):
        self.eval()
        self.likelihood.eval()
        
        inputs = torch.stack([u_ids, i_ids], dim=-1)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self(inputs))
            
            norm_pred = observed_pred.mean
            norm_uncertainty = observed_pred.stddev
            
            pred = norm_pred * (self.r_max - self.r_min) + self.r_min
            uncertainty = norm_uncertainty * (self.r_max - self.r_min)
            
        return pred, uncertainty
