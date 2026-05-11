import torch
import torch.nn as nn
from losses.pafl import PAFLLoss

class LossSelector:

    def __init__(self, loss_type:str = "mse", **kwargs):
        self.loss_type = loss_type.lower()
        self.kwargs = kwargs

    def get_loss(self):

        if self.loss_type == "mse":
            return nn.MSELoss()

        elif self.loss_type == "mae":
            return nn.L1Loss()

        elif self.loss_type == "huber":
            delta = self.kwargs.get("delta", 1.0)
            return nn.HuberLoss(delta=delta)

        elif self.loss_type == "logcosh":
            return self.log_cosh_loss

        elif self.loss_type == "relative":
            return self.relative_loss

        elif self.loss_type == "physics":
            return self.physics_loss
        
        elif self.loss_type == "pafl":

            return PAFLLoss(
                lambda_temp=self.kwargs.get(
                    "lambda_temp",
                    0.1
                ),

                lambda_rare=self.kwargs.get(
                    "lambda_rare",
                    0.1
                ),

                lambda_res=self.kwargs.get(
                    "lambda_res",
                    0.1
                ),

                alpha=self.kwargs.get(
                    "alpha",
                    0.2
                ),

                beta=self.kwargs.get(
                    "beta",
                    0.9
                )
            )

        else:
            raise ValueError(f"Loss '{self.loss_type}' not supported")

    # --- custom losses ---

    def log_cosh_loss(self, pred, target):
        return torch.mean(torch.log(torch.cosh(pred - target + 1e-12)))

    def relative_loss(self, pred, target):
        eps = 1e-8
        return torch.mean(
            torch.abs(pred - target) / (torch.abs(target) + eps)
        )

    def physics_loss(self, pred, target):
        """
        MSE + peso relativo tipo tanh 
        """
        diff = pred - target

        mse = torch.mean(diff**2)

        rel = diff / (torch.abs(target) + 1e-8)
        weight = torch.tanh(torch.abs(rel))

        weighted = torch.mean(weight * diff**2)

        lambda_ = self.kwargs.get("lambda", 0.1)

        return mse + lambda_ * weighted
    
    