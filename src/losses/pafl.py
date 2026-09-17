import torch
import torch.nn as nn


class PAFLLoss(nn.Module):

    def __init__(
        self,
        lambda_temp=0.2,
        lambda_rare=0.2,
        lambda_res=0.2,
        alpha=0.1,
        beta=0.9,
        eps=1e-8
    ):

        super().__init__()

        # Lambdas base
        self.lambda_temp = lambda_temp
        self.lambda_rare = lambda_rare
        self.lambda_res = lambda_res

        # Adaptación fuzzy
        self.alpha = alpha
        self.beta = beta

        self.eps = eps

        # EMA
        self.filtered_error = None
        self.prev_filtered = None

        self.prev_loss = None

    def update_lambda(self, current_loss):

        # ==========================================
        # EMA filtering
        # ==========================================

        if self.filtered_error is None:

            self.filtered_error = current_loss

        else:

            self.filtered_error = (
                self.beta * self.filtered_error
                +
                (1 - self.beta) * current_loss
            )

        # ==========================================
        # Delta
        # ==========================================

        if self.prev_filtered is None:

            delta = torch.tensor(0.0)

        else:

            delta = (
                self.filtered_error
                -
                self.prev_filtered
            )

        # ==========================================
        # Fuzzy factor
        # ==========================================

        factor = torch.tanh(delta)

        # ==========================================
        # Update lambdas
        # ==========================================

        self.lambda_temp *= (
            1 + self.alpha * factor.item()
        )

        self.lambda_rare *= (
            1 + self.alpha * factor.item()
        )

        self.lambda_res *= (
            1 + self.alpha * factor.item()
        )

        self.prev_filtered = self.filtered_error

    def forward(self, pred, target):

        # ==========================================
        # Error
        # ==========================================

        diff = pred - target

        # ==========================================
        # Base MSE
        # ==========================================

        mse = torch.mean(diff**2)

        # ==========================================
        # Temporal term
        # ==========================================

        if self.prev_loss is None:

            delta_loss = torch.tensor(
                0.0,
                device=pred.device
            )

        else:

            delta_loss = torch.relu(
                mse - self.prev_loss
            )

        # ==========================================
        # Rare-event term
        # ==========================================

        rel_error = diff / (
            torch.abs(pred) + self.eps
        )

        rare_weight = torch.tanh(
            torch.abs(rel_error)
        )

        rare_loss = torch.mean(
            rare_weight * diff**2
        )

        # ==========================================
        # Resolution term
        # ==========================================

        resolution_loss = torch.std(diff)

        # ==========================================
        # Total loss
        # ==========================================

        total_loss = (

            mse

            + self.lambda_temp * delta_loss

            + self.lambda_rare * rare_loss

            + self.lambda_res * resolution_loss
        )

        # ==========================================
        # Update fuzzy lambdas
        # ==========================================

        self.update_lambda(
            mse.detach()
        )

        self.prev_loss = mse.detach()

        return total_loss