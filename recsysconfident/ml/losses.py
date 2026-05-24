import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_mse_with_weight_penalty(y_true, y_pred, reduction='sum'):
    y_true = y_true[:, 0].float()
    y_pred, sample_weight = y_pred[:, 0].float(), y_pred[:, 1].float()

    pred_loss = sample_weight * (y_true - y_pred) ** 2
    det_loss = torch.log(1 / torch.clamp(sample_weight, min=1e-6, max=1e6))
    sample_loss = pred_loss + det_loss

    if reduction == 'none':
        return sample_loss
    elif reduction == 'mean':
        return torch.mean(sample_loss)
    else:  # 'sum'
        return torch.sum(sample_loss)

class WeightedMSEWithWeightPenalty(nn.Module):
    def __init__(self, reduction='sum'):
        super(WeightedMSEWithWeightPenalty, self).__init__()
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        return weighted_mse_with_weight_penalty(y_true, y_pred, reduction=self.reduction)

def custom_mse(y_true, y_pred, reduction='sum'):
    y_true = y_true.float()
    if len(y_pred.shape) > len(y_true.shape):
        y_pred = y_pred[:, 0]
    sample_loss = (y_true - y_pred) ** 2

    if reduction == 'none':
        return sample_loss
    elif reduction == 'mean':
        return torch.mean(sample_loss)
    else:  # 'sum'
        return torch.sum(sample_loss)

class CustomRMSE(nn.Module):
    def __init__(self, reduction='sum'):
        super(CustomRMSE, self).__init__()
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        return torch.sqrt(custom_mse(y_true, y_pred, reduction=self.reduction))

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, outputs, targets):
        return torch.sqrt(nn.MSELoss()(outputs, targets))

class KDELoss(nn.Module):
    def __init__(self, r_min, r_max, num_bins,
                 bandwidth=0.5,
                 lambda_kl=0.1):
        super().__init__()

        self.bandwidth = bandwidth
        self.lambda_kl = lambda_kl

        self.register_buffer(
            'grid',
            torch.linspace(r_min, r_max, num_bins)
        )

    def forward(self, x, y):

        grid = self.grid.view(1, -1)

        x_view = x.view(-1, 1)
        y_view = y.view(-1, 1)

        # KDE estimates
        p_x = torch.exp(
            -0.5 * ((x_view - grid) / self.bandwidth) ** 2
        )
        p_x = p_x.mean(dim=0)
        p_x = p_x / (p_x.sum() + 1e-8)

        p_y = torch.exp(
            -0.5 * ((y_view - grid) / self.bandwidth) ** 2
        )
        p_y = p_y.mean(dim=0)
        p_y = p_y / (p_y.sum() + 1e-8)

        # Stability
        p_x = p_x.clamp_min(1e-8)
        p_x = p_x / p_x.sum()
        
        p_y = p_y.clamp_min(1e-8)
        p_y = p_y / p_y.sum()

        # KL divergence
        kl_div = torch.sum(
            p_y * torch.log(p_y / p_x)
        )

        return self.lambda_kl * kl_div
    
class SoftHistogramWasserstein(nn.Module):

    def __init__(self,
                 r_min=1.0,
                 r_max=5.0,
                 num_bins=None, #suggested num_bins = relevance range + 1 = (r_max - r_min) + 1
                 bandwidth=0.5 #suggested as 0.5 because the relevances are usually discrete integers like 1, 2, 3, 4, 5
                 ):

        super().__init__()

        if (num_bins == None) :
            num_bins = r_max - r_min + 1
            
        self.bandwidth = bandwidth

        self.register_buffer(
            'bins',
            torch.linspace(r_min, r_max, num_bins)
        )

    def soft_histogram(self, x):

        x = x.view(-1, 1)
        bins = self.bins.view(1, -1)

        weights = torch.exp(
            -0.5 * ((x - bins) / self.bandwidth) ** 2
        )

        weights = weights / (
            weights.sum(dim=1, keepdim=True) + 1e-8
        )

        hist = weights.mean(dim=0)

        hist = hist / (hist.sum() + 1e-8)

        return hist

    def forward(self, pred, target):

        p = self.soft_histogram(pred)
        q = self.soft_histogram(target)

        cdf_p = torch.cumsum(p, dim=0)
        cdf_q = torch.cumsum(q, dim=0)

        wasserstein = torch.mean(
            torch.abs(cdf_p - cdf_q)
        )

        return wasserstein
    