import torch

class Metrics:

    def __init__(self, y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8):
        self.y_pred = y_pred
        self.y_true = y_true
        self.eps = eps

    # --- errores básicos ---
    def mse(self):
        return torch.mean((self.y_pred - self.y_true) ** 2)

    def mae(self):
        return torch.mean(torch.abs(self.y_pred - self.y_true))

    def rmse(self):
        return torch.sqrt(self.mse())

    # --- error relativo ---
    def relative_error(self):
        return torch.mean(
            torch.abs(self.y_pred - self.y_true) / (torch.abs(self.y_true) + self.eps)
        )

    # --- métricas físicas ---
    def bias(self):
        return torch.mean(self.y_pred - self.y_true)

    def resolution(self):
        # std del error
        return torch.std(self.y_pred - self.y_true)

    # --- percentiles ---
    def percentile_68(self):
        return torch.quantile(torch.abs(self.y_pred - self.y_true), 0.68)

    def percentile_95(self):
        return torch.quantile(torch.abs(self.y_pred - self.y_true), 0.95)

    # --- correlación ---
    def correlation(self):
        pred_mean = torch.mean(self.y_pred)
        true_mean = torch.mean(self.y_true)

        cov = torch.mean((self.y_pred - pred_mean) * (self.y_true - true_mean))

        pred_std = torch.std(self.y_pred, unbiased=False)
        true_std = torch.std(self.y_true, unbiased=False)

        return cov / (pred_std * true_std + self.eps)

    # --- R^2 ---
    def r2(self):
        ss_res = torch.sum((self.y_true - self.y_pred) ** 2)
        ss_tot = torch.sum((self.y_true - torch.mean(self.y_true)) ** 2)
        return 1 - ss_res / (ss_tot + self.eps)

    # --- diferencia directa ---
    def diff(self):
        return self.y_pred - self.y_true

    # --- energy scale (útil en física) ---
    def energy_scale(self):
        # E_pred / E_true
        return torch.mean(self.y_pred / (self.y_true + self.eps))
    