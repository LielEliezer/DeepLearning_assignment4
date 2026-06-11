import torch
from torch.utils.data import TensorDataset, DataLoader


def get_loaders(X_train):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        drop_last=True
    )
    return train_loader