"""Regression: a last batch with a single sample used to crash training
(0-d tensors from .squeeze(), and BatchNorm in train mode)."""
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from models.mlp import ShowerMLP  # noqa: E402
from optimizers.optimizer_selector import OptimizerSelector  # noqa: E402
from trainers.trainer import Trainer  # noqa: E402


def _train(n_train, n_val, bs, tmp_path, batchnorm=False, drop_last=False):
    torch.manual_seed(0)
    X = torch.randn(n_train + n_val + 1, 5)
    y = torch.rand(n_train + n_val + 1, 1) * 100
    tr = DataLoader(TensorDataset(X[:n_train], y[:n_train]), batch_size=bs, shuffle=True,
                    drop_last=drop_last)
    va = DataLoader(TensorDataset(X[n_train:n_train + n_val], y[n_train:n_train + n_val]), batch_size=bs)
    te = DataLoader(TensorDataset(X[-bs - 1:], y[-bs - 1:]), batch_size=bs)   # also ends with 1 sample
    m = ShowerMLP(input_dim=5, output_dim=1, n_layers=1, n_neurons=8, activation="relu",
                  dropout=0.0, batchnorm=batchnorm)
    opt = OptimizerSelector(optimizer_type="adam", lr=1e-3).get_optimizer(m)
    t = Trainer(m, "mse", opt, "cpu", str(tmp_path), scale="log1p")
    t.set_dataloaders(tr, va)
    t.train(epochs=2, early_stopping=False, patience=3)
    t.evaluate_test(te)
    return t


@pytest.mark.parametrize("n_train,n_val", [(129, 128), (128, 129), (130, 131)])
def test_single_sample_last_batch_does_not_crash(n_train, n_val, tmp_path):
    t = _train(n_train, n_val, 128, tmp_path)
    assert len(t.history["val"]["resolution"]) == 2
    assert torch.isfinite(t.test_metrics["rmse"])


def test_batchnorm_with_single_sample_last_train_batch(tmp_path):
    # run_search.run_one() uses drop_last in exactly this case
    t = _train(129, 128, 128, tmp_path, batchnorm=True, drop_last=True)
    assert len(t.history["train"]["loss"]) == 2


def test_validation_loss_is_mse_for_every_training_loss(tmp_path):
    """val_loss must not depend on the training loss (comparability across trials)."""
    torch.manual_seed(0)
    X = torch.randn(64, 5); y = torch.rand(64, 1) * 50
    losses = {}
    for loss in ("mse", "mae", "huber"):
        torch.manual_seed(1)
        m = ShowerMLP(input_dim=5, output_dim=1, n_layers=1, n_neurons=8, activation="relu",
                      dropout=0.0, batchnorm=False)
        opt = OptimizerSelector(optimizer_type="adam", lr=1e-3).get_optimizer(m)
        t = Trainer(m, loss, opt, "cpu", str(tmp_path), scale="log1p")
        loader = DataLoader(TensorDataset(X, y), batch_size=64)
        loss_val, _ = t.forward_step(X, y, "val")
        ref = torch.mean((t.output_transform.transform(y).view(-1) - m(X).view(-1)) ** 2)
        assert loss_val.item() == pytest.approx(ref.item(), rel=1e-5)