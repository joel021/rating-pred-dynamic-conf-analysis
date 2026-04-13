import torch

from recsysconfident.ml.models.torchmodel import TorchModel


class SimpleConfModel(TorchModel):

    def __init__(self, items_per_user):
        super().__init__(items_per_user)

    def l2(self, layer):
        l2_loss = torch.norm(layer.weight, p=2) ** 2  # L2 norm squared for weights
        return l2_loss
    
    def forward(self, user_ids, item_ids):
        pass

    def predict(self, user_ids, item_ids):
        outputs = self.forward(user_ids, item_ids)
        prediction, confidence = outputs[:, 0], outputs[:, 1]
        return prediction, confidence

    def eval_loss(self, user_ids, item_ids, labels):
        outputs = self.forward(user_ids, item_ids)

        return torch.sqrt(torch.nn.functional.mse_loss(labels, outputs[:, 0], reduction='mean'))

    def loss(self, user_ids, item_ids, labels, optimizer):
        optimizer.zero_grad()
        outputs = self.forward(user_ids, item_ids)
        loss = self.criterion(labels, outputs[:, 0]) + self.regularization() * 0.0001
        loss.backward()
        optimizer.step()
        return loss
