import torch 

class OutputScale:

    def __init__(
        self,
        method="asinh",
        scale=10000.0
    ):

        self.method = method
        self.scale = scale

    def transform(self, y):

        if self.method == "log1p":
            return torch.log1p(y)

        elif self.method == "log":
            return torch.log(y)

        elif self.method == "asinh":
            return torch.asinh(y / self.scale)

        else:
            return y

    def inverse_transform(self, y):

        if self.method == "log1p":
            y_real = torch.expm1(y)

        elif self.method == "log":
            y_real = torch.exp(y)

        elif self.method == "asinh":
            y_real = self.scale * torch.sinh(y)

        else:
            y_real = y

        # -----------------------------
        # Numerical stability
        # -----------------------------
        y_real = torch.nan_to_num(
            y_real,
            nan=0.0,
            posinf=1e12,
            neginf=0.0
        )

        return y_real