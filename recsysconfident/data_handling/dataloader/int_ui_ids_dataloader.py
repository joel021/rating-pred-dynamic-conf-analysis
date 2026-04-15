"""
pakcage: recsysconfident.data_handling.dataloader.int_ui_ids_dataloader.py
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


def gp_data_dl(info: DatasetInfo, fold):

    train_folds_df = pd.concat(info.df_folds[:fold+1], ignore_index=True)
    test_fold_df = info.df_folds[fold+1]

    fit_dataloader = DataLoader(
        TensorDataset(torch.from_numpy(train_folds_df[info.user_col].values.astype(int)).int(),
                      torch.from_numpy(train_folds_df[info.item_col].values.astype(int)).int(),
                      torch.from_numpy(train_folds_df[info.relevance_col].values.astype(float)).float()),
        batch_size=info.batch_size,
        shuffle=False)

    eval_dataloader = DataLoader(
        TensorDataset(torch.from_numpy(test_fold_df[info.user_col].values.astype(int)).int(),
                      torch.from_numpy(test_fold_df[info.item_col].values.astype(int)).int(),
                      torch.from_numpy(test_fold_df[info.relevance_col].values.astype(float)).float()),
        batch_size=info.batch_size,
        shuffle=False)
    
    sample_df = train_folds_df.sample(n=min(1000, len(train_folds_df)))

    inducing_points = torch.stack([
        torch.tensor(sample_df[info.user_col].values, dtype=torch.int),
        torch.tensor(sample_df[info.item_col].values, dtype=torch.int)
    ], dim=-1)

    return fit_dataloader, eval_dataloader, inducing_points, len(train_folds_df)

def ui_ids_label(info: DatasetInfo, fold):

    train_folds_df = pd.concat(info.df_folds[:fold+1], ignore_index=True)
    test_fold_df = info.df_folds[fold+1]

    fit_dataloader = DataLoader(
        TensorDataset(torch.from_numpy(train_folds_df[info.user_col].values.astype(int)).int(),
                      torch.from_numpy(train_folds_df[info.item_col].values.astype(int)).int(),
                      torch.from_numpy(train_folds_df[info.relevance_col].values.astype(float)).float()),
        batch_size=info.batch_size,
        shuffle=False)

    eval_dataloader = DataLoader(
        TensorDataset(torch.from_numpy(test_fold_df[info.user_col].values.astype(int)).int(),
                      torch.from_numpy(test_fold_df[info.item_col].values.astype(int)).int(),
                      torch.from_numpy(test_fold_df[info.relevance_col].values.astype(float)).float()),
        batch_size=info.batch_size,
        shuffle=False)
    
    return fit_dataloader, eval_dataloader
