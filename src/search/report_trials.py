"""Rank Optuna trials aggregated over seeds (trials*.csv written by run_search.py).

    python search/report_trials.py --search-dir <search_dir> [--metric resolution] [--top 10]

Ranks by mean VALIDATION metric over seeds and shows mean +/- std (validation
and test). Test columns are for reporting only -- never select on them.
"""
import argparse
import glob
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--search-dir", required=True)
ap.add_argument("--metric", default="resolution")
ap.add_argument("--top", type=int, default=10)
a = ap.parse_args()

files = glob.glob(f"{a.search_dir}/trials*.csv")
if not files:
    raise SystemExit("no trials*.csv found (run an Optuna search first)")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
asc = a.metric not in ("r2", "correlation")
df = df.sort_values(f"val_{a.metric}_mean", ascending=asc)
cols = ["trial", "input_type", "layers", "neurons", "activation", "loss", "optimizer", "lr",
        "batch_size", "n_seeds", f"val_{a.metric}_mean", f"val_{a.metric}_std",
        f"{a.metric}_mean", f"{a.metric}_std", "val_r2_mean", "n_params_mean"]
print(df[[c for c in cols if c in df.columns]].head(a.top).to_string(index=False))
print("\nColumns *_mean/*_std without 'val_' are TEST (report only).")