"""Tests for search/run_search.py's strategy: optuna (TPE + pruning).

Requires `pip install -e ".[optuna,corsario]"`. Run with:
    cd src && pytest ../tests/test_search_optuna.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import pandas as pd
import pytest
import yaml

pytest.importorskip("optuna")
pytest.importorskip("corsario")

from test_build_feature_dataset import _write_synthetic_dat  # noqa: E402

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))


def _build_features(tmp_path, n_showers=200):
    dat_path = tmp_path / "DAT_TEST"
    _write_synthetic_dat(dat_path, n_showers=n_showers)
    feature_dir = tmp_path / "features"
    subprocess.run(
        [sys.executable, str(SRC_DIR / "data" / "build_feature_dataset.py"),
         "--dat-file", str(dat_path), "--out-dir", str(feature_dir),
         "--input-types", "stats", "grid", "--batch-size", "50"],
        check=True, cwd=SRC_DIR,
    )
    return feature_dir


def _base_config(search_dir, feature_dir, n_trials=6, epochs=6):
    return {
        "search": {
            "id": "optuna_test",
            "name": "optuna test",
            "path": str(search_dir),
            "feature_dir": str(feature_dir),
            "output": ["total_energy"],
            "scale": "log1p",
            "strategy": "optuna",
            "seed_split": 1,
            "split": {"train": 0.7, "validation": 0.15},
            "epochs": epochs,
            "early_stopping": {"enabled": False, "patience": 10},
            "batch_size": [32],
            "use_mlflow": False,
            "optuna": {
                "n_trials": n_trials,
                "objective_metric": "resolution",
                "pruning": {"enabled": True, "n_warmup_steps": 1},
                "sampler_seed": 0,
            },
        },
        "architectures": {
            "mlp": {
                "input_types": ["stats", "grid"],
                "layers": [1, 2, 3],
                "neurons": [8, 16, 32],
                "activation": ["relu", "tanh"],
                "dropout": [0.0],
                "batchnorm": [False],
                "loss": ["mse"],
                "optimizer": ["adam"],
                "lr": [0.01, 0.001],
                "scheduler": {"enabled": False},
                "seeds": [1],
            },
            # deliberately unimplemented -- must be skipped, not crash:
            "kan": {"input_types": ["stats"], "layers": [2], "neurons": [16]},
        },
    }


def _run(config_path, extra_args=()):
    return subprocess.run(
        [sys.executable, str(SRC_DIR / "search" / "run_search.py"),
         "--config", str(config_path), *extra_args],
        check=True, cwd=SRC_DIR, capture_output=True, text=True,
    )


def test_optuna_search_runs_and_reports(tmp_path):
    feature_dir = _build_features(tmp_path)
    search_dir = tmp_path / "experiments"
    cfg = _base_config(search_dir, feature_dir)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))

    result = _run(cfg_path)
    assert "skipping architecture(s) not yet implemented: ['kan']" in result.stdout

    results_csv = search_dir / "search_optuna_test" / "results.csv"
    assert results_csv.exists()
    df = pd.read_csv(results_csv)
    assert len(df) == cfg["search"]["optuna"]["n_trials"]
    assert set(df["status"]).issubset({"ok", "pruned", "failed"})
    assert (df["status"] == "failed").sum() == 0

    # report.py must handle a mix of ok/pruned rows without error.
    subprocess.run(
        [sys.executable, str(SRC_DIR / "search" / "report.py"),
         "--search-dir", str(search_dir / "search_optuna_test")],
        check=True, cwd=SRC_DIR,
    )
    report_text = (search_dir / "search_optuna_test" / "report.md").read_text()
    assert "total runs" in report_text


def test_optuna_pruning_actually_prunes(tmp_path):
    """With an aggressive pruner and enough trials, at least one trial
    should be pruned before completing all epochs -- otherwise the
    pruning integration isn't doing anything."""
    feature_dir = _build_features(tmp_path)
    search_dir = tmp_path / "experiments"
    cfg = _base_config(search_dir, feature_dir, n_trials=8, epochs=10)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))

    _run(cfg_path)

    df = pd.read_csv(search_dir / "search_optuna_test" / "results.csv")
    assert (df["status"] == "pruned").sum() >= 1


def test_optuna_shared_storage_parallel_workers(tmp_path):
    """Two sequential 'workers' sharing one Optuna study (as a SLURM array
    would run concurrently) must not collide on output files or run
    directories, and the second worker's study must already contain the
    first worker's trials."""
    feature_dir = _build_features(tmp_path)
    search_dir = tmp_path / "experiments"
    cfg = _base_config(search_dir, feature_dir, n_trials=2, epochs=4)
    cfg["search"]["optuna"]["storage"] = f"sqlite:///{tmp_path / 'study.db'}"
    cfg["search"]["optuna"]["study_name"] = "shared"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))

    _run(cfg_path, extra_args=["--shard-index", "0", "--shard-count", "2"])
    _run(cfg_path, extra_args=["--shard-index", "1", "--shard-count", "2"])

    run_dir = search_dir / "search_optuna_test"
    shard0 = pd.read_csv(run_dir / "results_shard_0.csv")
    shard1 = pd.read_csv(run_dir / "results_shard_1.csv")
    assert len(shard0) == 2
    assert len(shard1) == 2
    assert set(shard0["run_name"]).isdisjoint(set(shard1["run_name"]))

    import optuna
    study = optuna.load_study(study_name="shared", storage=cfg["search"]["optuna"]["storage"])
    assert len(study.trials) == 4  # both workers' trials landed in the same study


def _multiseed_cfg(search_dir, feature_dir, seeds, n_trials=3):
    cfg = _base_config(search_dir, feature_dir, n_trials=n_trials, epochs=4)
    cfg["search"]["optuna"]["pruning"] = {"enabled": False}
    arch = cfg["architectures"]["mlp"]
    arch.update({"input_types": ["stats"], "layers": [1], "neurons": [8, 16],
                 "activation": ["relu"], "lr": [0.01], "seeds": seeds})
    cfg["architectures"].pop("kan")
    return cfg


def test_multiseed_trials_aggregate_over_seeds(tmp_path):
    """Every trial trains all seeds of the SAME config; trials.csv has mean/std
    and the objective is the mean VALIDATION metric (not test)."""
    feature_dir = _build_features(tmp_path)
    search_dir = tmp_path / "experiments"
    cfg = _multiseed_cfg(search_dir, feature_dir, seeds=[1, 2, 3])
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    _run(cfg_path)

    run_dir = search_dir / "search_optuna_test"
    res = pd.read_csv(run_dir / "results.csv")
    trials = pd.read_csv(run_dir / "trials.csv")

    # one row per (trial, seed); all seeds of a trial share the hyperparameters
    per_trial = res.groupby("trial")
    assert (per_trial["seed"].nunique() == 3).all()
    assert (per_trial[["layers", "neurons", "lr"]].nunique() == 1).all().all()

    # aggregate columns exist and match the per-seed rows
    for _, t in trials.iterrows():
        rows = res[res["trial"] == t["trial"]]
        assert t["n_seeds"] == 3
        assert t["val_resolution_mean"] == pytest.approx(rows["val_resolution"].mean(), rel=1e-5)
        assert t["val_resolution_std"] == pytest.approx(rows["val_resolution"].std(ddof=1), rel=1e-4)
        assert "resolution_mean" in trials.columns  # test, report only

    # study objective == mean validation resolution (selection never uses test)
    assert (trials["val_resolution_mean"] > 0).all()


def test_duplicate_combination_is_not_retrained(tmp_path):
    """Search space with a single combination: after the first trial every
    other trial must reuse the cached result (no extra training rows)."""
    feature_dir = _build_features(tmp_path)
    search_dir = tmp_path / "experiments"
    cfg = _multiseed_cfg(search_dir, feature_dir, seeds=[1, 2], n_trials=4)
    cfg["architectures"]["mlp"]["neurons"] = [8]
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    result = _run(cfg_path)

    res = pd.read_csv(search_dir / "search_optuna_test" / "results.csv")
    assert len(res) == 2                      # 1 combination x 2 seeds, trained once
    assert result.stdout.count("duplicate combination") == 3


def test_report_trials_hides_test_by_default(tmp_path):
    feature_dir = _build_features(tmp_path)
    search_dir = tmp_path / "experiments"
    cfg = _multiseed_cfg(search_dir, feature_dir, seeds=[1, 2])
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    _run(cfg_path)
    run_dir = search_dir / "search_optuna_test"

    def report(*extra):
        return subprocess.run(
            [sys.executable, str(SRC_DIR / "search" / "report_trials.py"),
             "--search-dir", str(run_dir), *extra],
            check=True, cwd=SRC_DIR, capture_output=True, text=True).stdout

    default = report()
    assert "val_resolution_mean" in default and "val_bias_mean" in default
    header = default.splitlines()[0]
    assert " resolution_mean" not in header           # test column hidden
    assert " resolution_mean" in report("--show-test").splitlines()[0]


def test_confirm_best_runs_over_split_seeds(tmp_path):
    feature_dir = _build_features(tmp_path)
    search_dir = tmp_path / "experiments"
    cfg = _multiseed_cfg(search_dir, feature_dir, seeds=[1, 2], n_trials=2)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    _run(cfg_path)
    run_dir = search_dir / "search_optuna_test"

    subprocess.run(
        [sys.executable, str(SRC_DIR / "search" / "confirm_best.py"),
         "--search-dir", str(run_dir), "--split-seeds", "1", "2", "--seeds", "1", "2"],
        check=True, cwd=SRC_DIR)

    conf = pd.read_csv(run_dir / "confirm" / "confirm.csv")
    assert len(conf) == 4 and (conf["status"] == "ok").all()
    assert set(conf["split_seed"]) == {1, 2}
    assert conf["trial"].nunique() == 1
    summ = pd.read_csv(run_dir / "confirm" / "confirm_summary.csv")
    assert {"resolution", "rmse", "bias"}.issubset(set(summ["metric"]))
    assert (summ["n_trainings"] == 4).all()