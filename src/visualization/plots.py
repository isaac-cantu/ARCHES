import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

class TrainPlots:

    def __init__(self, preds, targets, path, model_name, experiment_path):

        self.path = Path(path)

        self.preds = preds
        self.targets = targets

        self.model_name = model_name
        self.experiment_path = Path(experiment_path)

        self.plots_experiment = self.experiment_path / "plots"

        self.plots_experiment.mkdir(
            parents=True,
            exist_ok=True
        )

        self.plots_experiment = self.experiment_path / "plots" / "residuals"

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

    # ============================================
    # Prediction vs True
    # ============================================
    def plot_predictions(
        self,
        gridsize=70
    ):

        # ----------------------------------------
        # Convert tensors -> numpy
        # ----------------------------------------
        targets = (
            self.targets
            .detach()
            .cpu()
            .numpy()
        )

        preds = (
            self.preds
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------
        # Figure
        # ----------------------------------------
        fig, ax = plt.subplots(
            figsize=(8,7)
        )

        ax.set_box_aspect(1)
        
        # ----------------------------------------
        # Hexbin density
        # ----------------------------------------
        hb = ax.hexbin(

            targets,

            preds,

            gridsize=gridsize,

            bins="log",

            cmap="viridis",

            mincnt=1
        )

        # ----------------------------------------
        # Perfect prediction line
        # ----------------------------------------
        min_v = min(
            targets.min(),
            preds.min()
        )

        max_v = max(
            targets.max(),
            preds.max()
        )

        ax.plot(

            [min_v, max_v],

            [min_v, max_v],

            "--",

            color="red",

            linewidth=2,

            alpha=0.8
        )

        # ----------------------------------------
        # Metrics
        # ----------------------------------------
        corr = np.corrcoef(
            targets,
            preds
        )[0,1]

        bias = np.mean(
            preds - targets
        )

        resolution = np.std(
            preds - targets
        )

        text = (

            f"Correlation = {corr:.4f}\n"

            f"Bias = {bias:.4f}\n"

            f"Resolution = {resolution:.4f}"
        )

        ax.text(

            0.03,

            0.97,

            text,

            transform=ax.transAxes,

            fontsize=10,

            verticalalignment="top",

            bbox=dict(
                boxstyle="round",
                alpha=0.15
            )
        )

        # ----------------------------------------
        # Labels
        # ----------------------------------------
        ax.set_title(
            "Prediction vs True"
        )

        ax.set_xlabel(
            "True Energy"
        )

        ax.set_ylabel(
            "Predicted Energy"
        )

        # ----------------------------------------
        # Optional log scales
        # ----------------------------------------
        # ax.set_xscale("log")
        # ax.set_yscale("log")

        # ----------------------------------------
        # Colorbar
        # ----------------------------------------
        cbar = fig.colorbar(
            hb,
            ax=ax
        )

        cbar.set_label(
            "Log Counts"
        )

        # ----------------------------------------
        # Grid
        # ----------------------------------------
        ax.grid(
            alpha=0.2
        )

        # ----------------------------------------
        # Layout
        # ----------------------------------------
        plt.tight_layout()

        # ----------------------------------------
        # Save
        # ----------------------------------------
        plt.savefig(

            self.plots_path /

            "pred_vs_true.png",

            dpi=300,

            bbox_inches="tight"
        )

        plt.close()

    # ============================================
    # Residual Distribution
    # ============================================
    def plot_residuals(self): 
        
        residuals = (self.preds - self.targets)/torch.abs(self.targets)
        
        plt.figure(figsize=(8,5)) 
        
        sns.histplot( residuals, bins=50, kde=True ) 
        
        plt.title("Residual") 
        plt.xlabel("Residual") 
        plt.tight_layout() 
        plt.savefig( self.plots_path / "residuals.png" ) 
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

        # ============================================
        # Prediction vs True (Hexbin)
        # ============================================

        hb = axes[1].hexbin(

            self.targets.detach().cpu().numpy(),

            self.preds.detach().cpu().numpy(),

            gridsize=70,

            bins="log",

            cmap="viridis",

            mincnt=1
        )

        # ----------------------------------------
        # Perfect prediction line
        # ----------------------------------------
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

            "--",

            color="red",

            linewidth=2
        )

        # ----------------------------------------
        # Labels
        # ----------------------------------------
        axes[1].set_title(
            "Prediction vs True"
        )

        axes[1].set_xlabel(
            "True"
        )

        axes[1].set_ylabel(
            "Predicted"
        )

        # ----------------------------------------
        # Optional log scale
        # ----------------------------------------
        # axes[1].set_xscale("log")
        # axes[1].set_yscale("log")

        # ----------------------------------------
        # Colorbar
        # ----------------------------------------
        cbar = plt.colorbar(

            hb,

            ax=axes[1]
        )

        cbar.set_label(
            "Log Counts"
        )

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

    # ============================================
    # Residual vs True Energy
    # ============================================
    def residual_vs_ytrue(
        self, plot_name,
        bins=100
    ):

        # ----------------------------------------
        # Residuals
        # ----------------------------------------
        residuals = self.preds - self.targets

        residuals = (
             (self.preds - self.targets)
             / (torch.abs(self.targets) + 1e-8)
        )

        # ----------------------------------------
        # Figure
        # ----------------------------------------
        fig, ax = plt.subplots(
            figsize=(8,6)
        )

        # ----------------------------------------
        # Scatter / density
        # ----------------------------------------
        hb = ax.hexbin(

            self.targets,

            residuals,

            gridsize=70,

            bins="log",

            cmap="viridis",

            mincnt=1
        )

        # ----------------------------------------
        # Zero residual line
        # ----------------------------------------
        ax.axhline(

            0,

            color="red",

            linestyle="--",

            linewidth=2
        )

        # ----------------------------------------
        # Labels
        # ----------------------------------------
        ax.set_title(
            "Residuals vs True Energy"
        )

        ax.set_xlabel(
            "True Energy"
        )

        ax.set_ylabel(
            "Residual (Pred - True)"
        )

        # ----------------------------------------
        # Colorbar
        # ----------------------------------------
        cbar = fig.colorbar(
            hb,
            ax=ax
        )

        cbar.set_label(
            "Counts"
        )

        # ========================================
        # Right histogram
        # ========================================
        divider = make_axes_locatable(ax)

        ax_hist = divider.append_axes(

            "right",

            size="20%",

            pad=0.15,

            sharey=ax
        )

        # ----------------------------------------
        # Horizontal histogram
        # ----------------------------------------
        ax_hist.hist(

            residuals,

            bins=50,

            orientation="horizontal",

            log=True
        )

        # ----------------------------------------
        # Histogram style
        # ----------------------------------------
        ax_hist.set_xlabel(
            "Counts"
        )

        ax_hist.grid(alpha=0.3)

        plt.setp(
            ax_hist.get_yticklabels(),
            visible=False
        )

        # ----------------------------------------
        # Layout
        # ----------------------------------------
        plt.tight_layout()

        # ----------------------------------------
        # Save
        # ----------------------------------------

        plt.savefig(
            self.plots_path / "residual_vs_ytrue.png",
            dpi=300
        )

        plt.savefig(
            self.experiment_path / "plots" / "residuals" / f"{plot_name}.png",
            dpi=300
        )

        plt.close()

    def plot_all(self, plot_name):

        self.plot_loss()

        self.plot_predictions()

        self.plot_residuals()

        self.general_plot(plot_name)

        self.residual_vs_ytrue(plot_name,100)

