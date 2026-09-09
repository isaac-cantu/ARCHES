from torch.utils.data import TensorDataset, DataLoader
import torch

def split_data(X, y, train=0.7, val=0.15):

    n = X.shape[0]
    idx = torch.randperm(n)

    n_train = int(train * n)
    n_val   = int(val * n)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val     = X[val_idx], y[val_idx]
    X_test, y_test   = X[test_idx], y[test_idx]

    # -------- normalización correcta --------
    X_mean = X_train.mean(dim=0)
    X_std  = X_train.std(dim=0) + 1e-8

    X_train = (X_train - X_mean) / X_std
    X_val   = (X_val   - X_mean) / X_std
    X_test  = (X_test  - X_mean) / X_std

    # -------- clipping extra (anti-explosión) --------
    X_train = torch.clamp(X_train, -5, 5)
    X_val   = torch.clamp(X_val, -5, 5)
    X_test  = torch.clamp(X_test, -5, 5)

    train_ds = TensorDataset(X_train, y_train.view(-1,1))
    val_ds   = TensorDataset(X_val,   y_val.view(-1,1))
    test_ds  = TensorDataset(X_test,  y_test.view(-1,1))

    return train_ds, val_ds, test_ds