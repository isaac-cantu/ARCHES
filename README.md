# ARCHES

A systematic study of neural-network architectures applied to
CORSIKA-simulated cosmic-ray air showers.

Raw CORSIKA `.DAT` files are read with **[corsario](https://github.com/isaac-cantu/corsario)**
(a companion C++/pybind11 library — see that repository for the reader
itself), turned into per-particle / per-shower feature tables, and fed into
a configurable training pipeline (currently: MLP regression of shower
observables such as primary energy) with experiment tracking via MLflow.

## Pipeline overview

```
CORSIKA .DAT
     |  corsario (C++/pybind11) -- see github.com/isaac-cantu/corsario
     v
particles.csv / shower.csv   (or read directly: data.load_db.DAT_file)
     |  data.load_db.{particles_csv, shower_csv}
     v
DataFrames (df_particles, df_shower)
     |  data.preprocessing.data_processed(input_type=...)
     |    +-- "stats"  -> data.features.stats   (16 hand/PCA-derived features per shower)
     |    +-- "grid"   -> data.features.grid    (n x n spatial grid of per-cell stats)
     |    +-- "circle" -> data.features.circle  (concentric-ring stats)
     v
X, y (torch tensors, one row per shower)
     |  data.split_data.split_data  (train/val/test split + per-feature normalization)
     v
DataLoaders
     |  experiments.experiment_runner.ExperimentRunner
     |    +-- models.mlp.ShowerMLP
     |    +-- trainers.trainer.Trainer  (training loop, early stopping, scheduler,
     |    |                              physics-aware output scaling, metrics)
     |    +-- visualization.{plots,exp_plots}  (per-model and per-experiment plots)
     v
experiments/exp_<id>/model_<n>/seed_<s>/{best_model.pth, history.json, metrics.json, plots/}
                     /summary.csv   (one row per model, across the whole experiment)
```

`pipeline/run_pipeline.py` drives all of this from a single YAML config
(grid search over layers/neurons/activation/seed/batch size — see
`pipeline/config_example.yaml` for the schema that is actually wired up;
`pipeline/config_yaml.yaml` is an older draft with a different, **not**
currently supported schema — see the comment at the top of that file).

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# to read raw .DAT files directly instead of pre-converted CSVs:
pip install -e ".[corsario]"
```

See [`docs/PIPELINE.md`](docs/PIPELINE.md) for the config schema and what
each `input_type` produces, and [`docs/INSTALL_SERVER.md`](docs/INSTALL_SERVER.md)
for cluster/Docker deployment.

## Running an experiment

```bash
cd src
python pipeline/run_pipeline.py --configs pipeline/config_example.yaml
```

This will, for every combination in `search_space` x `training.batch_size`:
train an MLP, evaluate it on the held-out test split, save the model,
training history, test metrics, and plots under
`<experiment.path>/exp_<experiment.id>/model_<n>/seed_<s>/`, and append a row
to `<experiment.path>/exp_<experiment.id>/summary.csv`.

## Comparing architectures / hyperparameters (the full search)

For "which architecture and settings work best" — not just one MLP config
— use the search system instead:

```bash
cd src
python search/run_search.py --config pipeline/config_search_example.yaml
python search/report.py --search-dir <search.path>/search_<id>
```

This trains every (architecture x input_type x layers x neurons x
activation x dropout x loss x optimizer x seed x ...) combination described
in the YAML, and produces a ranked `report.md` + comparison plots (best per
architecture, metric vs. width/depth, predicted-vs-true for the best run).
Designed to be handed to a SLURM job array (`scripts/slurm_search.sh`) for
a large search, or run locally for a quick one. See
[`docs/SEARCH.md`](docs/SEARCH.md) for the full config schema.

## Preparing data

```python
from data.load_db import DAT_file, particles_csv, shower_csv

# Option A: read a raw CORSIKA file directly (requires: pip install -e ".[corsario]")
df_particles, df_shower = DAT_file("/path/to/DAT000001")

# Option B: read pre-converted CSVs (e.g. produced once via `corsario-convert`)
df_particles = particles_csv("/path/to/processed/example")   # expects particles.csv there
df_shower    = shower_csv("/path/to/processed/example")      # expects shower.csv there
```

## Current status / known gaps

This is an active research pipeline; the honest state as of this review:

- **Implemented and tested**: the full `.DAT` -> features -> MLP -> metrics ->
  plots path, for `input_type` in `{stats, grid, circle}`, with MLflow
  tracking, early stopping, and (as of this review) a working LR scheduler.
- **New**: `search/run_search.py` + `search/report.py` — a
  multi-architecture, multi-hyperparameter search with an auto-generated
  ranked report, and a minimal `ShowerCNN` (`models/cnn.py`) as a second
  real architecture to compare against the MLP. See
  [`docs/SEARCH.md`](docs/SEARCH.md).
- **Partially implemented**: `models/cnn.py` now has a minimal working
  `ShowerCNN` (conv blocks + global average pool + linear head) added as
  part of the search system, trainable on the `grid` input representation.
  `models/kan.py` is still empty — `model.type: kan` (or `pinn`) is
  accepted by `pipeline/run_pipeline.py`'s config but silently does nothing
  in `ExperimentRunner.build_model()`, then fails later with a confusing
  `AttributeError` rather than a clear error at config-validation time;
  `search/run_search.py` guards against this specific failure mode by
  checking `search.space.IMPLEMENTED_ARCHITECTURES` up front and skipping
  with a clear message instead. `notebooks/05_cnn_model.ipynb`,
  `06_rnn_model.ipynb`, `07_lstm_model.ipynb` remain placeholders (RNN/LSTM
  would need a sequential, variable-length particle-level input pipeline
  that doesn't exist yet — a bigger change than the fixed-size feature
  vectors `stats`/`grid`/`circle` produce today).
- **Single-output only in practice**: `data_config["output"]` is a list
  (e.g. `["total_energy"]`) but every feature generator currently hard-codes
  `total_energy` as the regression target regardless of what's listed
  there. Multi-target regression (e.g. energy + arrival direction together)
  would need `data/features/*.py` and `models/mlp.py`'s `output_dim` wiring
  extended to actually use `data_config["output"]`.
- `experiment_management/{experiment_registry,optuna_manager,pruning,search_spaces}.py`
  and `evaluation/report.py` are empty stubs; the grid search in
  `run_pipeline.py` is a plain nested loop (no Optuna/pruning yet), which
  is fine for the current search space size but will not scale to a larger
  hyperparameter search.
- See [`CHANGELOG.md`](CHANGELOG.md) for what changed in this review pass.
