import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# =====================================================
# Experiment plots
# =====================================================

class ExpPlots:

    def __init__(self, path:str=None):

        self.path = Path(path)

        self.summary_path = (
            self.path / "summary.csv"
        )

        self.plot_path = (
            self.path / "general_plots"
        )

        self.plot_path.mkdir(
            parents=True,
            exist_ok=True
        )

        self.df = pd.read_csv(
            self.summary_path
        )

    # =================================================
    # Heatmaps grouped by seed
    # =================================================
    def heat_map_plot(
        self,
        metric="rmse"
    ):

        # ---------------------------------
        # Fixed configuration groups
        # ---------------------------------
        group_cols = [

            "seed",

            "batch_size",

            "activation",

            "optimizer",

            "loss",

            "dropout",

            "batchnorm"
        ]

        # ---------------------------------
        # Iterate groups
        # ---------------------------------
        for keys, group_df in self.df.groupby(group_cols):

            (
                seed,
                batch_size,
                activation,
                optimizer,
                loss,
                dropout,
                batchnorm
            ) = keys

            # ---------------------------------
            # Pivot table
            # ---------------------------------
            pivot = group_df.pivot_table(

                values=metric,

                index="layers",

                columns="width",

                aggfunc="mean"
            )

            # ---------------------------------
            # Skip empty pivots
            # ---------------------------------
            if pivot.empty:
                continue

            # ---------------------------------
            # Plot
            # ---------------------------------
            plt.figure(figsize=(8,6))

            sns.heatmap(

                pivot,

                annot=True,

                fmt=".4f",

                cmap="viridis"
            )

            # ---------------------------------
            # Title
            # ---------------------------------
            plt.title(

                f"{metric.upper()} Heatmap\n"

                f"Seed={seed} | "

                f"Batch={batch_size} | "

                f"Act={activation} | "

                f"Loss={loss} | "

                f"Opt={optimizer} | "

                f"Dropout={dropout} | "

                f"BN={batchnorm}"
            )

            # ---------------------------------
            # Labels
            # ---------------------------------
            plt.xlabel("Width")

            plt.ylabel("Hidden Layers")

            plt.tight_layout()

            # ---------------------------------
            # File name
            # ---------------------------------
            filename = (

                f"heatmap_"

                f"S{seed}_"

                f"B{batch_size}_"

                f"{activation}_"

                f"{optimizer}_"

                f"{loss}_"

                f"D{dropout}_"

                f"BN{batchnorm}.png"
            )

            # ---------------------------------
            # Save
            # ---------------------------------
            plt.savefig(

                self.plot_path / filename
            )

            plt.close()

    # =================================================
    # Horizontal comparison bar plot
    # =================================================
    def compare_models_barplot(
        self,
        metric="mse",
        top_k=20
    ):

        # ------------------------------------------------
        # Agrupar ignorando seed
        # ------------------------------------------------
        group_cols = [

            "layers",
            "width",
            "activation",
            "loss",
            "optimizer",
            "dropout",
            "batchnorm",
            "batch_size"
        ]

        grouped = (

            self.df

            .groupby(group_cols)[metric]

            .agg(["mean", "std"])

            .reset_index()
        )

        # ------------------------------------------------
        # Ordenar
        # ------------------------------------------------
        ascending = False if metric == "r2" else True

        grouped = (

            grouped

            .sort_values(
                "mean",
                ascending=ascending
            )

            .head(top_k)
        )

        # ------------------------------------------------
        # Labels
        # ------------------------------------------------
        labels = []

        for _, row in grouped.iterrows():

            label = (

                f"L{row['layers']} | "

                f"W{row['width']} | "

                f"{row['activation']} | "

                f"{row['optimizer']} | "

                f"B{row['batch_size']} "
            )

            labels.append(label)

        # ------------------------------------------------
        # Plot
        # ------------------------------------------------
        plt.figure(figsize=(14,8))

        plt.barh(

            labels,

            grouped["mean"],

            xerr=grouped["std"]
        )

        plt.xlabel(f"{metric.upper()} (mean ± std)")

        plt.ylabel("Models")

        plt.title(

            f"Top {top_k} Models Comparison"
        )

        # Mejor modelo arriba
        plt.gca().invert_yaxis()

        plt.tight_layout()

        plt.savefig(

            self.plot_path /

            f"compare_models_{metric}.png"
        )

        plt.close()

    # =================================================
    # Multi-metric comparison
    # =================================================
    def metrics_grid_plot(self):

        metrics = [

            "mse",
            "mae",
            "rmse",
            "r2",
            "relative_error",
            "bias",
            "resolution",
            "p68",
            "p95"
        ]

        fig, ax = plt.subplots(
            3,
            3,
            figsize=(18,14)
        )

        ax = ax.flatten()

        # ------------------------------------------------
        # Agrupar modelos ignorando seed
        # ------------------------------------------------
        group_cols = [

            "layers",
            "width",
            "activation",
            "loss",
            "optimizer",
            "dropout",
            "batchnorm",
            "batch_size"
        ]

        for i, metric in enumerate(metrics):

            grouped = (

                self.df

                .groupby(group_cols)[metric]

                .agg(["mean", "std"])

                .reset_index()
            )

            if metric == "bias":

                grouped["mean"] = grouped["mean"].abs()

            # --------------------------------------------
            # R2 se maximiza
            # --------------------------------------------
            ascending = False if metric == "r2" else True

            best_models = (

                grouped

                .sort_values(
                    "mean",
                    ascending=ascending
                )

                .head(10)
            )

            # --------------------------------------------
            # Labels
            # --------------------------------------------
            labels = [

                (
                    f"L{r['layers']} | "
                    f"W{r['width']} | "
                    f"{r['activation']} | "
                    f"B{r['batch_size']} "
                )

                for _, r in best_models.iterrows()
            ]

            # --------------------------------------------
            # Plot
            # --------------------------------------------
            ax[i].barh(

                labels,

                best_models["mean"],

                xerr=best_models["std"]
            )

            ax[i].invert_yaxis()

            ax[i].set_title(
                metric.upper()
            )

            ax[i].set_xlabel(
                "mean ± std"
            )

        plt.suptitle(
            "Model Comparison Across Metrics",
            fontsize=18
        )

        plt.tight_layout()

        plt.savefig(

            self.plot_path /

            "metrics_grid.png"
        )

        plt.close()

    # =================================================
    # Correlation heatmap
    # =================================================
    def correlation_heatmap(self):

        metrics = [

            "mse",
            "mae",
            "rmse",
            "r2",
            "relative_error",
            "bias",
            "resolution",
            "p68",
            "p95",
            "correlation"
        ]

        corr = (
            self.df[metrics]
            .corr()
        )

        plt.figure(figsize=(10,8))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title(
            "Metrics Correlation"
        )

        plt.tight_layout()

        plt.savefig(
            self.plot_path /
            "metrics_correlation.png"
        )

        plt.close()

    # =================================================
    # Run all
    # =================================================
    def plot_all(self):

        self.heat_map_plot()

        self.compare_models_barplot()

        self.metrics_grid_plot()

        self.correlation_heatmap()


if __name__ == "__main__":
    path = "/home/icantu24/Documents/ARCHES/experiments/exp_H_2_8_N_8_512_mse_relu_4S_stats_log1p"
    ExpPlots(path).plot_all()