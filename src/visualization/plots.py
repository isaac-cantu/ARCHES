import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class TrainPlots:

    def __init__(self, preds, targets, path, model_name, experiment_path):

        self.path = Path(path)

        self.preds = np.array(preds)
        self.targets = np.array(targets)

        self.model_name = model_name
        self.experiment_path = Path(experiment_path)

        self.plots_experiment = self.experiment_path / "plots"

        self.plots_experiment.mkdir(
            parents=True,
            exist_ok=True
        )

        self.plots_path = self.path / "plots"

        self.plots_path.mkdir(
            parents=True,
            exist_ok=True
        )


        history_path = self.path / "history.json"

        with open(history_path, "r") as f:

            self.history = json.load(f)

    def plot_loss(self):

        plt.figure(figsize=(8,5))

        plt.plot(
            self.history["train"]["loss"],
            label="Train"
        )

        plt.plot(
            self.history["val"]["loss"],
            label="Validation"
        )

        plt.axvline(self.history["best_epoch"]["epoch"] - 1,
            linestyle="--",
            label="Best Epoch"
        )

        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            self.plots_path / "loss.png"
        )

        plt.close()

    def plot_predictions(self):

        plt.figure(figsize=(6,6))

        plt.scatter(
            self.targets,
            self.preds,
            alpha=0.5
        )

        min_v = min(
            self.targets.min(),
            self.preds.min()
        )

        max_v = max(
            self.targets.max(),
            self.preds.max()
        )

        plt.plot(
            [min_v, max_v],
            [min_v, max_v],
            "--"
        )

        plt.title("Prediction vs True")
        plt.xlabel("True")
        plt.ylabel("Predicted")

        plt.tight_layout()

        plt.savefig(
            self.plots_path / "pred_vs_true.png"
        )

        plt.close()

    def plot_residuals(self):

        residuals = self.preds - self.targets

        plt.figure(figsize=(8,5))

        sns.histplot(
            residuals,
            bins=50,
            kde=True
        )

        plt.title("Residual")
        plt.xlabel("Residual")

        plt.tight_layout()

        plt.savefig(
            self.plots_path / "residuals.png"
        )

        plt.close()

    def general_plot(self, plot_name):

        fig, axes = plt.subplots(
            4,
            2,
            figsize=(14, 18)
        )

        fig.suptitle(self.model_name)

        axes = axes.flatten()

        # =========================================================
        # LOSS
        # =========================================================
        axes[0].plot(
            self.history["train"]["loss"],
            label="Train"
        )

        axes[0].plot(
            self.history["val"]["loss"],
            label="Validation"
        )

        axes[0].axvline(
            self.history["best_epoch"]["epoch"] - 1,
            linestyle="--",
            label="Best Epoch"
        )

        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        # =========================================================
        # PRED VS TRUE
        # =========================================================
        axes[1].scatter(
            self.targets,
            self.preds,
            alpha=0.5
        )

        min_v = min(
            self.targets.min(),
            self.preds.min()
        )

        max_v = max(
            self.targets.max(),
            self.preds.max()
        )

        axes[1].plot(
            [min_v, max_v],
            [min_v, max_v],
            "--"
        )

        axes[1].set_title("Prediction vs True")
        axes[1].set_xlabel("True")
        axes[1].set_ylabel("Predicted")

        # =========================================================
        # MSE
        # =========================================================
        axes[2].plot(
            self.history["train"]["mse"],
            label="Train"
        )

        axes[2].plot(
            self.history["val"]["mse"],
            label="Validation"
        )

        axes[2].axvline(
            self.history["best_epoch"]["epoch"] - 1,
            linestyle="--",
            label="Best Epoch"
        )

        axes[2].set_title("MSE")
        axes[2].legend()

        # =========================================================
        # MAE
        # =========================================================
        axes[3].plot(
            self.history["train"]["mae"],
            label="Train"
        )

        axes[3].plot(
            self.history["val"]["mae"],
            label="Validation"
        )

        axes[3].axvline(
            self.history["best_epoch"]["epoch"] - 1,
            linestyle="--",
            label="Best Epoch"
        )

        axes[3].set_title("MAE")
        axes[3].legend()

        # =========================================================
        # RMSE
        # =========================================================
        axes[4].plot(
            self.history["train"]["rmse"],
            label="Train"
        )

        axes[4].plot(
            self.history["val"]["rmse"],
            label="Validation"
        )

        axes[4].axvline(
            self.history["best_epoch"]["epoch"] - 1,
            linestyle="--",
            label="Best Epoch"
        )

        axes[4].set_title("RMSE")
        axes[4].legend()

        # =========================================================
        # R2
        # =========================================================
        axes[5].plot(
            self.history["train"]["r2"],
            label="Train"
        )

        axes[5].plot(
            self.history["val"]["r2"],
            label="Validation"
        )

        axes[5].axvline(
            self.history["best_epoch"]["epoch"] - 1,
            linestyle="--",
            label="Best Epoch"
        )

        axes[5].set_title("R²")
        axes[5].legend()

        # =========================================================
        # BIAS
        # =========================================================
        axes[6].plot(
            self.history["train"]["bias"],
            label="Train"
        )

        axes[6].plot(
            self.history["val"]["bias"],
            label="Validation"
        )

        axes[6].axvline(
            self.history["best_epoch"]["epoch"] - 1,
            linestyle="--",
            label="Best Epoch"
        )

        axes[6].set_title("Bias")
        axes[6].legend()

        # =========================================================
        # RESOLUTION
        # =========================================================
        axes[7].plot(
            self.history["train"]["resolution"],
            label="Train"
        )

        axes[7].plot(
            self.history["val"]["resolution"],
            label="Validation"
        )

        axes[7].axvline(
            self.history["best_epoch"]["epoch"] - 1,
            linestyle="--",
            label="Best Epoch"
        )

        axes[7].set_title("Resolution")
        axes[7].legend()

        plt.tight_layout()

        plt.savefig(
            self.plots_path / "training_metrics.png",
            dpi=300
        )

        plt.savefig(
            self.experiment_path / "plots" / f"{plot_name}.png",
            dpi=300
        )


        plt.close()

    def plot_all(self, plot_name):

        self.plot_loss()

        self.plot_predictions()

        self.plot_residuals()

        self.general_plot(plot_name)

