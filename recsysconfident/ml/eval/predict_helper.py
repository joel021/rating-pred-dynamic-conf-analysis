import torch
from recsysconfident.ml.models.torchmodel import TorchModel


def predict(model: TorchModel, data_loader, device: str = 'cpu'):
    # Create a DataLoader to iterate through the dataset
    model = model.to(device)
    model.eval()

    y_pred_list = []
    y_true_list = []
    pred_confs_list = []

    with torch.no_grad():
        for data in data_loader:

            users_ids, items_ids, ratings = data
            users_ids, items_ids, ratings = users_ids.to(device), items_ids.to(device), ratings.to(device)
            output, mconfs = model.predict(users_ids, items_ids)
            
            # Keep on CPU to avoid accumulating on GPU memory
            pred_confs_list.append(mconfs.cpu())
            y_pred_list.append(output.cpu())
            y_true_list.append(ratings.cpu())

    pred_confs = torch.cat(pred_confs_list, dim=0).view(-1).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).view(-1).numpy()
    y_true = torch.cat(y_true_list, dim=0).view(-1).numpy()
    return y_true, y_pred, pred_confs
