import torch
import torch.nn as nn
import torch.nn.functional as F

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.torchmodel import TorchModel


def get_MCDropoutRecModel_and_dataloader(info: DatasetInfo, fold: int):

    fit_dataloader, eval_dataloader = ui_ids_label(info, fold)

    model = MCDropoutRecModel(
        n_users=info.n_users,
        n_items=info.n_items,
        r_min=info.rate_range[0],
        r_max=info.rate_range[1],
        emb_dim=64,
        hidden_dim=64
    )

    return model, fit_dataloader, eval_dataloader

class MCDropoutRecModel(TorchModel):

    def __init__(
        self,
        n_users,
        n_items,
        r_max,
        r_min,
        items_per_user=None,
        emb_dim=64,
        hidden_dim=64,
        dropout=0.2,
        l2_reg=1e-6,
        mc_samples=50,
    ):
        super().__init__(items_per_user)

        self.r_max = r_max
        self.r_min = r_min

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        self.fc1 = nn.Linear(2 * emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

        self.l2_reg = l2_reg
        self.mc_samples = mc_samples

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)

        x = torch.cat([u, i], dim=-1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        pred = self.out(x).squeeze(-1)
        return torch.stack([pred, torch.zeros_like(pred)])

    def regularization(self):
        reg = 0.0
        for param in self.parameters():
            reg += torch.sum(param ** 2)
        return self.l2_reg * reg

    def _predict_point(self, user_ids, item_ids):
        self.eval()
        with torch.no_grad():
            return self.forward(user_ids, item_ids)
        
    def _enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _normalize(self, x):
        return (x - self.r_min) / (self.r_max - self.r_min + 1e-8)

    def _denormalize(self, x):
        return x * (self.r_max - self.r_min) + self.r_min

    def predict(self, user_ids, item_ids):
        """
        Returns:
            mean: [B] (denormalized)
            certainty: [B]
        """

        scale = (self.r_max - self.r_min)

        self.eval()
        self._enable_dropout()

        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                preds.append(self.forward(user_ids, item_ids)[:, 0])

        preds = torch.stack(preds, dim=0)

        # Mean & variance in normalized space
        mean_norm = preds.mean(dim=0)
        var_norm = preds.var(dim=0)

        # Denormalize mean
        mean = self._denormalize(mean_norm)

        var = var_norm * (scale ** 2)

        # Certainty (log-precision style)
        certainty = -torch.log(var + 1e-8)

        return mean, certainty
    
    def eval_loss(self, user_ids, item_ids, labels):
        """
        Deterministic evaluation (normalized space)
        """

        labels_norm = self._normalize(labels)

        preds = self._predict_point(user_ids, item_ids)

        loss = F.mse_loss(preds[:, 0], labels_norm, reduction="mean")
        return loss
    
    def loss(self, user_ids, item_ids, labels, optimizer):

        optimizer.zero_grad()
        self.train()

        labels_norm = self._normalize(labels)

        preds = self.forward(user_ids, item_ids)

        mse = F.mse_loss(preds[:, 0], labels_norm, reduction="mean")
        reg = self.regularization()

        loss = mse + reg

        loss.backward()
        optimizer.step()

        return loss
