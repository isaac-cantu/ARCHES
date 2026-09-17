"""Run a full architecture / hyperparameter search and write raw results.

    python search/run_search.py --config pipeline/config_search_example.yaml

Trains every (architecture, input_type, layers, neurons, activation,
dropout, batchnorm, loss, optimizer, lr, scheduler, batch_size, seed)
combination described in the search YAML, evaluates each on a held-out
test split, and appends one row per run to `results.csv` in the search's
output directory (flushed after every run, so a job that gets killed
partway through a long cluster run does not lose earlier results).

Use `search/report.py` afterwards to turn `results.csv` into a ranked
table + plots (see docs/SEARCH.md).

For SLURM job-array parallelism, pass --shard-index/--shard-count (see
scripts/slurm_search.sh); each shard writes its own
`results_shard_<i>.csv`, which report.py merges automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> src/

from data.load_db import DAT_file, particles_csv, shower_csv
from data.preprocessing import data_processed
from data.split_data import split_data
from pipeline.set_seed import set_seed
from experiments.experiment_runner import ExperimentRunner
from search.space import ARCH_COMPATIBLE_INPUT_TYPES, IMPLEMENTED_ARCHITECTURES, build_run_specs, shard
from search.metric_direction import optuna_direction, transform_for_direction


def load_data(search_cfg: dict):
    if search_cfg.get("dat_file"):
        print(f"[search] reading raw CORSIKA file via corsario: {search_cfg['dat_file']}")
        return DAT_file(search_cfg["dat_file"])
    path = search_cfg["data_path"]
    print(f"[search] reading particles.csv / shower.csv from: {path}")
    return particles_csv(path), shower_csv(path)


def load_prebuilt_features(feature_dir: str, input_type: str, architecture: str):
    """Load a compact (X, y) dataset built ahead of time by
    data/build_feature_dataset.py -- see docs/SEARCH.md#large-files. Used
    instead of computing features on the fly when `search.feature_dir` is
    set, so a multi-hour scan of a huge raw file only has to happen once,
    not once per run_search.py invocation.

    build_feature_dataset.py always stores "grid" flattened (matching
    data_processed's default); for architecture="cnn" we reshape it back
    to the channel-first (N, 6, n, n) tensor Conv2d needs. This is an
    exact reshape/permute of the same underlying values (verified in
    tests/test_search.py), not an approximation -- no need to store both
    layouts on disk.
    """
    import h5py

    path = Path(feature_dir) / f"{input_type}.h5"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run data/build_feature_dataset.py --input-types {input_type} "
            f"--out-dir {feature_dir} first (see docs/SEARCH.md#large-files)")
    with h5py.File(path, "r") as f:
        X = torch.from_numpy(f["X"][:]).float()
        y = torch.from_numpy(f["y"][:]).float()

    if architecture == "cnn":
        if input_type != "grid":
            raise ValueError("architecture=cnn requires input_type=grid")
        n_showers = X.shape[0]
        grid_n = round((X.shape[1] / 6) ** 0.5)
        if grid_n * grid_n * 6 != X.shape[1]:
            raise ValueError(f"could not infer a square grid_n from feature width {X.shape[1]}")
        X = X.reshape(n_showers, grid_n, grid_n, 6).permute(0, 3, 1, 2).contiguous()
    return X, y


def flatten_for(architecture: str) -> bool:
    return architecture != "cnn"


class DataCache:
    """Caches feature extraction and train/val/test splits so repeated
    hyperparameter combinations that share (input_type, architecture-shape)
    don't redo scatter_add/PCA/etc. or re-split the data."""

    def __init__(self, search_cfg: dict, df_particles=None, df_shower=None):
        self.df_particles = df_particles
        self.df_shower = df_shower
        self.feature_dir = search_cfg.get("feature_dir")
        self.split_cfg = search_cfg["split"]
        self.seed_split = search_cfg.get("seed_split", 1)
        self._features: dict = {}
        self._splits: dict = {}

    def get_splits(self, input_type: str, architecture: str):
        flatten = flatten_for(architecture)
        key = (input_type, flatten)
        if key not in self._splits:
            if key not in self._features:
                if self.feature_dir:
                    print(f"[search] loading prebuilt features: input_type={input_type} "
                          f"architecture={architecture} from {self.feature_dir}")
                    X, y = load_prebuilt_features(self.feature_dir, input_type, architecture)
                else:
                    print(f"[search] computing features: input_type={input_type} flatten={flatten}")
                    if input_type == "grid":
                        X, y = data_processed(input_type=input_type, df_particles=self.df_particles,
                                               df_shower=self.df_shower, flatten=flatten)
                    else:
                        X, y = data_processed(input_type=input_type, df_particles=self.df_particles,
                                               df_shower=self.df_shower)
                self._features[key] = (X, y)
            X, y = self._features[key]
            # Fixed seed for the split itself (independent of each run's
            # model-init seed) so every hyperparameter combination for a
            # given input_type is compared on the exact same test set --
            # mirrors pipeline/run_pipeline.py's set_seed(1) before split_data.
            set_seed(self.seed_split)
            self._splits[key] = split_data(X, y, train=self.split_cfg["train"], val=self.split_cfg["validation"])
        return self._splits[key]


def build_model_data(spec, search_cfg: dict, X_shape, run_dir: Path):
    architecture = spec.architecture
    data_block = {
        "type": spec.input_type,
        "output": search_cfg["output"],
        "scale": search_cfg["scale"],
    }
    if architecture == "cnn":
        data_block["in_channels"] = X_shape[1]
    else:
        data_block["input_dim"] = X_shape[1] if len(X_shape) == 2 else int(np.prod(X_shape[1:]))

    model_data = {
        "experiment": {
            "id": search_cfg["id"],
            "name": search_cfg["name"],
            "description": search_cfg.get("description", ""),
            "path": search_cfg["_search_dir"],
        },
        "model": {
            "id": spec.run_id,
            "type": architecture,
            "hidden_layers": spec.layers,
            "neurons": spec.neurons,
            "activation": spec.activation,
            "path": str(run_dir),
        },
        "training": {
            "batch_size": spec.batch_size,
            "epochs": search_cfg["epochs"],
            "early_stopping": search_cfg["early_stopping"]["enabled"],
            "patience": search_cfg["early_stopping"]["patience"],
            "dropout": spec.dropout,
            "loss": spec.loss,
            "batchnorm": spec.batchnorm,
            "optimizer": spec.optimizer,
            "lr": spec.lr,
            "scheduler_enabled": spec.scheduler_enabled,
            "scheduler_type": spec.scheduler_type,
            "seed": spec.seed,
        },
        "data": data_block,
    }
    return model_data


def run_one(spec, data_cache: DataCache, search_cfg: dict, use_mlflow: bool,
            epoch_callback=None) -> dict:
    train_ds, val_ds, test_ds = data_cache.get_splits(spec.input_type, spec.architecture)
    X_shape = tuple(train_ds.tensors[0].shape)

    set_seed(spec.seed)
    train_loader = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=spec.batch_size)
    test_loader = DataLoader(test_ds, batch_size=spec.batch_size)

    run_name = f"run_{spec.run_id:04d}_{spec.architecture}_{spec.input_type}"
    run_dir = Path(search_cfg["_search_dir"]) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_data = build_model_data(spec, search_cfg, X_shape, run_dir)
    training_data = {"train": train_loader, "val": val_loader, "test": test_loader}

    row = spec.as_dict()
    row["run_name"] = run_name
    t0 = time.time()
    try:
        runner = ExperimentRunner(model_data, training_data, use_mlflow=use_mlflow,
                                   epoch_callback=epoch_callback)
        runner.run()
        row["status"] = "ok"
        row["train_time_sec"] = time.time() - t0
        row["n_params"] = sum(p.numel() for p in runner.model.parameters())
        row["best_epoch"] = runner.train_model.get_history()["best_epoch"]["epoch"]
        for k, v in runner.train_model.test_metrics.items():
            row[k] = float(v.item()) if hasattr(v, "item") else float(v)

        preds, targets = runner.y_pred.numpy(), runner.y_true.numpy()
        np.savez(run_dir / "predictions.npz", preds=preds, targets=targets)
    except Exception as exc:
        # Optuna signals "stop this trial early, it's not promising" by
        # raising TrialPruned from the epoch_callback (see
        # run_optuna_search()). That is not a failure -- it must be
        # reported to Optuna as a pruned trial, not swallowed here, or the
        # pruner loses the very information it needs to keep pruning well.
        try:
            import optuna
            is_pruned = isinstance(exc, optuna.TrialPruned)
        except ImportError:
            is_pruned = False

        row["status"] = "pruned" if is_pruned else "failed"
        row["train_time_sec"] = time.time() - t0
        row["error"] = f"{type(exc).__name__}: {exc}"
        if not is_pruned:
            (run_dir / "error.txt").write_text(traceback.format_exc())
            print(f"[search] {run_name} FAILED: {row['error']}")
        if is_pruned:
            raise  # let Optuna's study.optimize() see it and mark the trial pruned
    return row


def run_optuna_search(full_cfg: dict, search_cfg: dict, data_cache: DataCache,
                       use_mlflow: bool, results_path: Path, worker_id: int = 0):
    """Bayesian optimization (TPE) + median pruning over the SAME discrete
    choices as the grid/random strategies -- no new config schema, just a
    smarter way to walk the same search space. See docs/SEARCH.md#optuna.

    `worker_id` lets multiple processes (e.g. a SLURM array, see
    scripts/slurm_search_optuna.sh) run trials against one shared Optuna
    study concurrently: pass a shared `search.optuna.storage` URL (a
    sqlite/postgres URL) and a distinct worker_id per process so run
    numbering and results.csv files don't collide.
    """
    import optuna
    from search.space import RunSpec

    optuna_cfg = search_cfg.get("optuna", {})
    n_trials = optuna_cfg.get("n_trials", 50)
    objective_metric = optuna_cfg.get("objective_metric", "resolution")
    pruning_cfg = optuna_cfg.get("pruning", {})
    pruning_enabled = pruning_cfg.get("enabled", True)
    n_warmup_steps = pruning_cfg.get("n_warmup_steps", 5)
    sampler_seed = optuna_cfg.get("sampler_seed", 0)
    storage = optuna_cfg.get("storage")  # e.g. "sqlite:///<search_dir>/optuna.db"
    study_name = optuna_cfg.get("study_name", f"search_{search_cfg['id']}")

    architectures_cfg = full_cfg["architectures"]
    enabled_architectures = [a for a in architectures_cfg if a in IMPLEMENTED_ARCHITECTURES]
    skipped = [a for a in architectures_cfg if a not in IMPLEMENTED_ARCHITECTURES]
    if skipped:
        print(f"[search] skipping architecture(s) not yet implemented: {skipped}")
    if not enabled_architectures:
        raise ValueError("no implemented architecture found in the search config")

    global_batch_sizes = search_cfg.get("batch_size", [64])

    def as_list(v):
        return v if isinstance(v, list) else [v]

    rows = []
    # Offset run numbering by worker so concurrent workers sharing one
    # Optuna study never write to the same runs/run_<n>_.../ directory.
    run_id_base = worker_id * 1_000_000

    def flush():
        pd.DataFrame(rows).to_csv(results_path, index=False)

    def objective(trial: "optuna.Trial"):
        architecture = trial.suggest_categorical("architecture", enabled_architectures)
        arch_cfg = architectures_cfg[architecture]
        compatible_inputs = [t for t in as_list(arch_cfg["input_types"])
                              if t in ARCH_COMPATIBLE_INPUT_TYPES[architecture]]
        input_type = trial.suggest_categorical(f"{architecture}_input_type", compatible_inputs)
        layers = trial.suggest_categorical(f"{architecture}_layers", [int(v) for v in as_list(arch_cfg.get("layers", [2]))])
        neurons = trial.suggest_categorical(f"{architecture}_neurons", [int(v) for v in as_list(arch_cfg.get("neurons", [32]))])
        activation = trial.suggest_categorical(f"{architecture}_activation", as_list(arch_cfg.get("activation", ["relu"])))
        dropout = trial.suggest_categorical(f"{architecture}_dropout", [float(v) for v in as_list(arch_cfg.get("dropout", [0.0]))])
        batchnorm = trial.suggest_categorical(f"{architecture}_batchnorm", [bool(v) for v in as_list(arch_cfg.get("batchnorm", [False]))])
        loss = trial.suggest_categorical(f"{architecture}_loss", as_list(arch_cfg.get("loss", ["mse"])))
        optimizer_name = trial.suggest_categorical(f"{architecture}_optimizer", as_list(arch_cfg.get("optimizer", ["adam"])))
        lr = trial.suggest_categorical(f"{architecture}_lr", [float(v) for v in as_list(arch_cfg.get("lr", [1e-3]))])
        batch_size = trial.suggest_categorical(f"{architecture}_batch_size",
                                                [int(v) for v in as_list(arch_cfg.get("batch_size", global_batch_sizes))])
        seed = trial.suggest_categorical(f"{architecture}_seed", [int(v) for v in as_list(arch_cfg.get("seeds", [1]))])
        scheduler_cfg = arch_cfg.get("scheduler", {"enabled": False, "type": None})

        spec = RunSpec(
            run_id=run_id_base + trial.number, architecture=architecture, input_type=input_type,
            layers=layers, neurons=neurons, activation=activation, dropout=dropout,
            batchnorm=batchnorm, loss=loss, optimizer=optimizer_name, lr=lr,
            scheduler_enabled=bool(scheduler_cfg.get("enabled", False)),
            scheduler_type=scheduler_cfg.get("type"), batch_size=batch_size, seed=seed,
        )

        print(f"[search] [optuna trial {trial.number}] {architecture}/{input_type} "
              f"L{layers} N{neurons} {activation} loss={loss} seed={seed} bs={batch_size}")

        def epoch_callback(epoch, val_loss):
            if not pruning_enabled:
                return
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            row = run_one(spec, data_cache, search_cfg, use_mlflow, epoch_callback=epoch_callback)
        except optuna.TrialPruned:
            rows.append({**spec.as_dict(), "run_name": f"run_{spec.run_id:07d}_{architecture}_{input_type}",
                         "status": "pruned"})
            flush()
            raise
        rows.append(row)
        flush()

        if row["status"] != "ok":
            # A real failure (not a deliberate prune) still shouldn't crash
            # the whole study -- tell Optuna to skip it rather than feed it
            # a fabricated objective value.
            raise optuna.TrialPruned()
        return transform_for_direction(objective_metric, row[objective_metric])

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps) if pruning_enabled else optuna.pruners.NopPruner()
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(
        direction=optuna_direction(objective_metric), sampler=sampler, pruner=pruner,
        storage=storage, study_name=study_name, load_if_exists=bool(storage),
    )

    study.optimize(objective, n_trials=n_trials)

    n_ok = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    n_pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    print(f"[search] optuna study done: {len(study.trials)} trial(s) this worker "
          f"({n_ok} completed, {n_pruned} pruned)")
    if study.best_trial is not None:
        print(f"[search] best so far: value={study.best_value:.4g} params={study.best_params}")
    print(f"[search] wrote {len(rows)} row(s) -> {results_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a search YAML (see docs/SEARCH.md)")
    parser.add_argument("--shard-index", type=int, default=0,
                         help="Grid/random: which shard of the plan to run. "
                              "Optuna: worker id (for shared-storage parallel studies, see docs/SEARCH.md#optuna).")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-runs", type=int, default=None, help="Truncate the plan (smoke-testing)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit without training")
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    search_cfg = full_cfg["search"]
    search_dir = Path(search_cfg["path"]) / f"search_{search_cfg['id']}"
    search_dir.mkdir(parents=True, exist_ok=True)
    search_cfg["_search_dir"] = str(search_dir)

    use_mlflow = bool(search_cfg.get("use_mlflow", False))
    suffix = "" if args.shard_count <= 1 else f"_shard_{args.shard_index}"
    results_path = search_dir / f"results{suffix}.csv"
    strategy = search_cfg.get("strategy", "grid")

    if strategy == "optuna":
        n_trials = full_cfg["search"].get("optuna", {}).get("n_trials", 50)
        archs = [a for a in full_cfg["architectures"] if a in IMPLEMENTED_ARCHITECTURES]
        print(f"[search] optuna strategy: {n_trials} trial(s) this worker over architectures {archs}")
        if args.dry_run:
            return
    else:
        specs = build_run_specs(full_cfg)
        specs = shard(specs, args.shard_index, args.shard_count)
        if args.max_runs:
            specs = specs[: args.max_runs]

        by_arch = {}
        for s in specs:
            by_arch.setdefault(s.architecture, 0)
            by_arch[s.architecture] += 1
        print(f"[search] {len(specs)} run(s) planned (shard {args.shard_index}/{args.shard_count}): {by_arch}")

        if args.dry_run:
            return

    with open(search_dir / "config_resolved.yaml", "w") as f:
        yaml.dump(full_cfg, f, default_flow_style=False)

    if search_cfg.get("feature_dir"):
        print(f"[search] using prebuilt features from: {search_cfg['feature_dir']} "
              f"(skipping raw data load -- see docs/SEARCH.md#large-files)")
        data_cache = DataCache(search_cfg)
    else:
        df_particles, df_shower = load_data(search_cfg)
        data_cache = DataCache(search_cfg, df_particles, df_shower)

    if strategy == "optuna":
        run_optuna_search(full_cfg, search_cfg, data_cache, use_mlflow, results_path,
                           worker_id=args.shard_index)
        print(f"[search] next: python search/report.py --search-dir {search_dir}")
        return

    rows = []
    for i, spec in enumerate(specs):
        print(f"[search] ({i+1}/{len(specs)}) {spec.architecture}/{spec.input_type} "
              f"L{spec.layers} N{spec.neurons} {spec.activation} loss={spec.loss} "
              f"seed={spec.seed} bs={spec.batch_size}")
        row = run_one(spec, data_cache, search_cfg, use_mlflow)
        rows.append(row)
        pd.DataFrame(rows).to_csv(results_path, index=False)  # flush every run

    print(f"[search] done. wrote {len(rows)} row(s) -> {results_path}")
    print(f"[search] next: python search/report.py --search-dir {search_dir}")


if __name__ == "__main__":
    main()
