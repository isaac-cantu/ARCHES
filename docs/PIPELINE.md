# Pipeline reference

## Config schema (`pipeline/config_example.yaml`)

```yaml
experiment:
  id: <str>                # used in exp_<id> folder name
  name: <str>               # MLflow experiment name
  description: <str>
  path: <str>                # where exp_<id>/ is created

data:
  path: <str>                 # directory containing particles.csv + shower.csv
                              # (data.load_db.particles_csv/shower_csv append
                              # "/particles.csv" and "/shower.csv" to this)
  input_type: stats|grid|circle
  scale: log1p|log|asinh|none  # applied to the target inside Trainer, NOT
                                # inside the feature generators (see below)
  output: [total_energy]       # currently informational only -- see gap below

model:
  type: mlp                    # cnn/kan/pinn accepted but not implemented yet
  dropout: <float>
  batchnorm: <bool>

training:
  epochs: <int>
  batch_size: [<int>, ...]     # swept
  optimizer: {type: adam|adamw|sgd|rmsprop, lr: <float>}
  early_stopping: {enabled: <bool>, patience: <int>}
  split: {train: <frac>, validation: <frac>, test: <frac>}   # test is
                                # implicit (1 - train - val) in split_data()
  scheduler: {enabled: <bool>, type: step|plateau|cosine}

search_space:
  layers: [<int>, ...]         # swept
  neurons: [<int>, ...]        # swept
  seed: [<int>, ...]           # swept
  activation: [relu|leaky_relu|elu|gelu|sigmoid|tanh, ...]   # swept

loss:
  type: mse|mae|pafl|...       # see losses/losses.py for the full list

metrics:
  - energy_res
  - mse
  - R2
```

Run with:

```bash
python pipeline/run_pipeline.py --configs pipeline/config_example.yaml [-v] [-g] [-o] [-m]
```

(`-v`/`-g`/`-o`/`-m` are parsed by `pipeline/utils.read_parser` for
verbose/GPU/Optuna/MLflow toggles; only `-g` (device) and MLflow are
currently consumed by the rest of the pipeline — Optuna is not wired up
yet, see the README's "known gaps" section.)

## What each `input_type` produces

All three read `df_particles` (columns: `shower, particle_type, no_particle,
hadr_gen, no_obs, mass, energy, particle_id, px, py, pz, x, y, t, weight` —
see the `corsario` CSV schema) and `df_shower` (columns: `event_no,
particle_id, total_energy, altitude, no_target, z, px, py, pz, zenith,
azimuth`), and return `(X, y)` as `torch.Tensor`, one row per shower,
`y = df_shower["total_energy"]` **unscaled** (the `data.scale` config value
is applied later, inside `Trainer.forward_step`, not here — do not also
apply a log/asinh transform inside a feature generator, or the target will
be double-transformed).

- **`stats`** (`data/features/stats.py`): 16 features per shower — spatial
  mean (x, y), 2D PCA of the (x, y) point cloud (eigenvalues + angle),
  time mean/std/skew, energy sum/mean/var/max (log1p'd), x-t and y-t
  covariance, radial mean/variance.
- **`grid`** (`data/features/grid.py`): bins particles into an `n x n`
  spatial grid (default `n=8`) and computes 6 per-cell statistics (count,
  energy sum/mean/std, time mean/std), flattened to `6*n*n` features.
- **`circle`** (`data/features/circle.py`): splits particles into 4
  concentric rings (by radial std from the shower centroid) and computes
  per-ring statistics, plus the global centroid and 2D covariance.

## Output directory layout

```
<experiment.path>/exp_<id>/
├── config.yaml                 # full resolved config for this experiment
├── summary.csv                 # one row per trained model (Trainer.save_model_csv)
├── plots/                      # DataPlots.plot_all() -- dataset-level plots
└── model_<n>/
    ├── config.yaml             # architecture/training config shared across seeds
    └── seed_<s>/
        ├── config.yaml
        ├── best_model.pth
        ├── history.json        # per-epoch train/val metrics + scheduler LR
        ├── metrics.json        # test-set metrics (Metrics class)
        └── plots/               # TrainPlots.plot_all() -- pred-vs-true, residuals, ...
```

## Known gap: single-output target

`data_config["output"]` is read and threaded through as `model_data["data"]["output"]`
(used for `output_dim=len(...)` in `ExperimentRunner.build_model`), but the
feature generators (`stats_generator`/`grid_generator`/`circle_generator`)
always return `df_shower["total_energy"]` as `y`, regardless of what
`output` lists. Today this is harmless because every example config uses
`output: [total_energy]`, so `output_dim == 1` matches the actual target
shape — but changing `output` in the config will silently *not* change what
the model is trained to predict. If/when multi-target regression (e.g.
energy + zenith together) is needed, the feature generators need an
`output_columns` parameter that selects/stacks the requested `df_shower`
columns instead of hard-coding `total_energy`.

## Large files (millions of showers)

`particles_csv()`/`shower_csv()`/`DAT_file()` all load the full particle
table into memory. For a file with millions of showers, use
`data/build_feature_dataset.py` to build compact per-shower features once
(streaming the file via `corsario.iter_shower_batches`, never holding more
than one batch in memory), then point `search/run_search.py` at the result
via `search.feature_dir`. See `docs/SEARCH.md#large-files` for the full
workflow, memory/runtime estimation, and why `grid_generator` specifically
needs a fixed (not per-batch) spatial range.
