"""Turn one or more search results*.csv into a ranked report + plots.

    python search/report.py --search-dir /path/to/experiments/search_<id>

Reads every `results*.csv` in --search-dir (so it transparently merges
per-shard files from a SLURM job array, see scripts/slurm_search.sh),
and writes:

    report.md          -- best overall, best per architecture/input_type,
                           top-20 table, failure summary
    plots/architecture_comparison.png
    plots/metric_vs_neurons.png
    plots/metric_vs_layers.png
    plots/best_model_pred_vs_true.png
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metric_direction import METRIC_DIRECTION


def load_results(search_dir: Path) -> pd.DataFrame:
    # Match run_search.py's outputs specifically (results.csv, or
    # results_shard_<i>.csv from a sharded/SLURM-array run) -- NOT the
    # broader "results*.csv", which would also match this script's own
    # `ranked_results.csv` output and double-count everything on a second
    # invocation.
    files = sorted(glob.glob(str(search_dir / "results.csv")) +
                    glob.glob(str(search_dir / "results_shard_*.csv")))
    if not files:
        raise FileNotFoundError(f"no results*.csv found in {search_dir}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"[report] loaded {len(df)} run(s) from {len(files)} file(s)")
    return df


def rank(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    direction = METRIC_DIRECTION.get(metric, "min")
    if direction == "min":
        ok["_score"] = ok[metric]
        ascending = True
    elif direction == "max":
        ok["_score"] = ok[metric]
        ascending = False
    elif direction == "abs_min":
        ok["_score"] = ok[metric].abs()
        ascending = True
    elif direction == "closest_to_1":
        ok["_score"] = (ok[metric] - 1.0).abs()
        ascending = True
    else:
        raise ValueError(f"unknown direction for metric {metric}")
    return ok.sort_values("_score", ascending=ascending)


def make_plots(df_ok: pd.DataFrame, metric: str, out_dir: Path, search_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) best score per architecture x input_type
    best_per_group = (
        df_ok.sort_values("_score").groupby(["architecture", "input_type"], as_index=False).first()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = best_per_group["architecture"] + " / " + best_per_group["input_type"]
    ax.bar(labels, best_per_group[metric])
    ax.set_ylabel(metric)
    ax.set_title(f"Best {metric} by architecture / input_type")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "architecture_comparison.png", dpi=150)
    plt.close(fig)

    # 2) metric vs neurons (width), colored by architecture
    fig, ax = plt.subplots(figsize=(6, 4))
    for arch, g in df_ok.groupby("architecture"):
        agg = g.groupby("neurons")[metric].mean().sort_index()
        ax.plot(agg.index, agg.values, marker="o", label=arch)
    ax.set_xlabel("neurons / base_channels (width)")
    ax.set_ylabel(f"mean {metric}")
    ax.set_title(f"{metric} vs. width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metric_vs_neurons.png", dpi=150)
    plt.close(fig)

    # 3) metric vs layers (depth)
    fig, ax = plt.subplots(figsize=(6, 4))
    for arch, g in df_ok.groupby("architecture"):
        agg = g.groupby("layers")[metric].mean().sort_index()
        ax.plot(agg.index, agg.values, marker="o", label=arch)
    ax.set_xlabel("layers / blocks (depth)")
    ax.set_ylabel(f"mean {metric}")
    ax.set_title(f"{metric} vs. depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metric_vs_layers.png", dpi=150)
    plt.close(fig)

    # 4) pred vs true for the single best run, if predictions were saved
    best_row = df_ok.sort_values("_score").iloc[0]
    pred_file = search_dir / "runs" / best_row["run_name"] / "predictions.npz"
    if pred_file.exists():
        data = np.load(pred_file)
        preds, targets = data["preds"], data["targets"]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(targets, preds, s=6, alpha=0.4)
        lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.set_title(f"Best run: {best_row['run_name']}")
        fig.tight_layout()
        fig.savefig(out_dir / "best_model_pred_vs_true.png", dpi=150)
        plt.close(fig)


def _fmt_cell(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def df_to_md(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> markdown table, avoids depending on `tabulate`
    (which pandas.DataFrame.to_markdown() otherwise requires)."""
    if df.empty:
        return "_(no rows)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt_cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report_md(df: pd.DataFrame, df_ok: pd.DataFrame, metric: str, out_path: Path):
    lines = ["# Architecture / hyperparameter search report", ""]
    n_failed = (df["status"] == "failed").sum()
    n_pruned = (df["status"] == "pruned").sum() if "status" in df.columns else 0
    summary = f"- total runs: {len(df)} ({len(df_ok)} ok, {n_failed} failed"
    if n_pruned:
        summary += f", {n_pruned} pruned by Optuna"
    summary += ")"
    lines.append(summary)
    lines.append(f"- ranking metric: `{metric}` "
                 f"({METRIC_DIRECTION.get(metric, 'min')} is better)")
    lines.append("")

    lines.append("## Best overall")
    lines.append("")
    best = df_ok.sort_values("_score").iloc[0]
    cols = ["run_name", "architecture", "input_type", "layers", "neurons", "activation",
            "dropout", "batchnorm", "loss", "optimizer", "lr", "batch_size", "seed",
            metric, "rmse", "r2", "resolution", "n_params", "train_time_sec"]
    seen = set()
    cols = [c for c in cols if c in best.index and not (c in seen or seen.add(c))]
    for c in cols:
        lines.append(f"- **{c}**: {best[c]}")
    lines.append("")

    lines.append("## Best per architecture")
    lines.append("")
    best_per_arch = df_ok.sort_values("_score").groupby("architecture", as_index=False).first()
    lines.append(df_to_md(best_per_arch[cols]))
    lines.append("")

    lines.append("## Best per input_type")
    lines.append("")
    best_per_input = df_ok.sort_values("_score").groupby("input_type", as_index=False).first()
    lines.append(df_to_md(best_per_input[cols]))
    lines.append("")

    lines.append("## Top 20 runs")
    lines.append("")
    lines.append(df_to_md(df_ok.sort_values("_score").head(20)[cols]))
    lines.append("")

    if n_failed:
        lines.append("## Failures")
        lines.append("")
        fail_cols = ["run_name", "architecture", "input_type", "error"]
        fail_cols = [c for c in fail_cols if c in df.columns]
        lines.append(df_to_md(df[df["status"] == "failed"][fail_cols]))
        lines.append("")

    lines.append("## Plots")
    lines.append("")
    for p in ["architecture_comparison.png", "metric_vs_neurons.png",
              "metric_vs_layers.png", "best_model_pred_vs_true.png"]:
        lines.append(f"![{p}](plots/{p})")
    lines.append("")

    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--metric", default="resolution",
                         help="Metric to rank by (default: resolution). "
                              "One of: " + ", ".join(METRIC_DIRECTION))
    args = parser.parse_args()

    search_dir = Path(args.search_dir)
    df = load_results(search_dir)
    df_ok = rank(df, args.metric)

    if df_ok.empty:
        print("[report] no successful runs to report on.")
        return

    out_dir = search_dir
    make_plots(df_ok, args.metric, out_dir / "plots", search_dir)
    write_report_md(df, df_ok, args.metric, out_dir / "report.md")
    df_ok.drop(columns="_score").to_csv(out_dir / "ranked_results.csv", index=False)

    print(f"[report] wrote {out_dir / 'report.md'}, {out_dir / 'plots'}/*.png, "
          f"{out_dir / 'ranked_results.csv'}")
    best = df_ok.iloc[0]
    print(f"[report] best: {best['run_name']} ({args.metric}={best[args.metric]:.4g})")


if __name__ == "__main__":
    main()
