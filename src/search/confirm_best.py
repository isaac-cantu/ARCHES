"""Confirm the winning configuration of an Optuna search on several random splits.

    python search/confirm_best.py --search-dir <search_dir> \
        [--trial N] [--metric resolution] [--split-seeds 1 2 3 4 5] [--seeds 1 2 3]

Why: the search compares every configuration on ONE train/val/test split
(`seed_split`). The reported test number of the winner then depends on that
single partition. This script re-trains the chosen configuration for every
combination of (split seed x model seed), each with its own fresh
train/val/test partition, and reports mean +/- std of the TEST metrics.

  * std across model seeds  -> initialization noise
  * std across split seeds  -> data-partition noise (the honest error bar)

The configuration is chosen by mean VALIDATION metric from trials*.csv
(or forced with --trial). Run this ONCE, after the search: it is the only
place the test set should be looked at to report a final number.

Output: <search_dir>/confirm/confirm.csv (one row per training) and
<search_dir>/confirm/confirm_summary.csv.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.run_search import DataCache, run_one  # noqa: E402
from search.space import RunSpec  # noqa: E402

TEST_METRICS = ["resolution", "rmse", "mae", "bias", "r2", "p68", "p95"]


def pick_trial(trials: pd.DataFrame, metric: str, trial: int | None) -> pd.Series:
    if trial is not None:
        sel = trials[trials["trial"] == trial]
        if sel.empty:
            raise SystemExit(f"trial {trial} not found in trials*.csv")
        return sel.iloc[0]
    col = f"val_{metric}_mean"
    asc = metric not in ("r2", "correlation")
    return trials.sort_values(col, ascending=asc).iloc[0]


def spec_from_row(row: pd.Series, run_id: int, seed: int) -> RunSpec:
    def get(key, default):
        v = row.get(key, default)
        return default if pd.isna(v) else v

    sched_type = get("scheduler_type", None)
    return RunSpec(
        run_id=run_id,
        architecture=str(get("architecture", "mlp")),
        input_type=str(row["input_type"]),
        layers=int(row["layers"]),
        neurons=int(row["neurons"]),
        activation=str(row["activation"]),
        dropout=float(get("dropout", 0.0)),
        batchnorm=bool(get("batchnorm", False)),
        loss=str(row["loss"]),
        optimizer=str(row["optimizer"]),
        lr=float(row["lr"]),
        scheduler_enabled=bool(get("scheduler_enabled", False)),
        scheduler_type=None if sched_type is None else str(sched_type),
        batch_size=int(row["batch_size"]),
        seed=int(seed),
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """mean/std over everything, plus std of the per-split means."""
    ok = df[df["status"] == "ok"]
    out = []
    for m in TEST_METRICS:
        if m not in ok.columns:
            continue
        per_split = ok.groupby("split_seed")[m].mean()
        out.append({
            "metric": m,
            "mean": ok[m].mean(),
            "std_all": ok[m].std(ddof=1) if len(ok) > 1 else float("nan"),
            "std_across_splits": per_split.std(ddof=1) if len(per_split) > 1 else float("nan"),
            "n_trainings": len(ok),
            "n_splits": per_split.shape[0],
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-dir", required=True)
    ap.add_argument("--trial", type=int, default=None, help="trial number (default: best by validation)")
    ap.add_argument("--metric", default="resolution", help="validation metric used to pick the winner")
    ap.add_argument("--split-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="model seeds (default: the seeds in the config)")
    a = ap.parse_args()

    search_dir = Path(a.search_dir)
    files = glob.glob(str(search_dir / "trials*.csv"))
    if not files:
        raise SystemExit("no trials*.csv found (run an Optuna search first)")
    trials = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    row = pick_trial(trials, a.metric, a.trial)

    with open(search_dir / "config_resolved.yaml") as f:
        full_cfg = yaml.safe_load(f)
    search_cfg = full_cfg["search"]
    arch = str(row.get("architecture", "mlp"))
    seeds = a.seeds or [int(s) for s in
                        (full_cfg["architectures"][arch].get("seeds", [1]))]

    out_dir = search_dir / "confirm"
    out_dir.mkdir(exist_ok=True)
    search_cfg["_search_dir"] = str(out_dir)
    use_mlflow = False

    print(f"[confirm] trial {int(row['trial'])}: {arch}/{row['input_type']} L{int(row['layers'])} "
          f"N{int(row['neurons'])} {row['activation']} loss={row['loss']} lr={row['lr']} "
          f"bs={int(row['batch_size'])}")
    print(f"[confirm] split seeds={a.split_seeds} model seeds={seeds} "
          f"-> {len(a.split_seeds) * len(seeds)} trainings")

    rows = []
    run_id = 900_000
    for ss in a.split_seeds:
        cfg = dict(search_cfg)
        cfg["seed_split"] = ss
        cache = DataCache(cfg)          # fresh split per split seed
        for sd in seeds:
            spec = spec_from_row(row, run_id, sd)
            run_id += 1
            r = run_one(spec, cache, cfg, use_mlflow)
            r["split_seed"] = ss
            r["trial"] = int(row["trial"])
            rows.append(r)
            pd.DataFrame(rows).to_csv(out_dir / "confirm.csv", index=False)
            print(f"[confirm] split={ss} seed={sd} status={r['status']} "
                  f"resolution={r.get('resolution', float('nan')):.4g}")

    df = pd.DataFrame(rows)
    summ = summarize(df)
    summ.to_csv(out_dir / "confirm_summary.csv", index=False)
    print("\nTEST metrics of the selected configuration (real units, e.g. GeV):")
    print(summ.to_string(index=False))
    print("\nstd_across_splits = partition noise; std_all = splits + initialization.")


if __name__ == "__main__":
    main()