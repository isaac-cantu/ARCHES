# Changelog (review pass)

## Added
- **`strategy: optuna`** in `search/run_search.py`: Bayesian optimization
  (TPE) + early pruning (`optuna.pruners.MedianPruner`) as an alternative
  to exhaustive `grid`/blind `random` search, for when the combinatorial
  grid gets too large to run in full. Same `architectures:` config schema
  as grid/random (lists of choices) -- Optuna samples from the same
  discrete space rather than needing a new range-based schema. Supports
  multiple worker processes collaborating on one shared study via
  `optuna.storage` (sqlite/postgres), see `scripts/slurm_search_optuna.sh`
  and `pipeline/config_search_optuna_example.yaml`. Required a small,
  backward-compatible addition to `trainers/trainer.py`'s `train()`: an
  optional `epoch_callback(epoch, val_loss)` hook, used to let Optuna
  observe progress and prune mid-training. See
  `docs/SEARCH.md#optuna` and `tests/test_search_optuna.py`.
- **Large-file support (millions of showers)**: `data/build_feature_dataset.py`
  streams a huge `.DAT` file via `corsario.iter_shower_batches()` (never
  holding more than one batch's particles in memory) and writes compact,
  resumable per-shower feature datasets (`stats.h5`/`grid.h5`/`circle.h5`).
  `search/run_search.py` gained `search.feature_dir` as an alternative to
  `dat_file`/`data_path`, loading these prebuilt files directly instead of
  recomputing features from raw particles on every invocation -- the
  multi-hour/day file scan happens once, not once per search. CNN training
  works from the same prebuilt `grid.h5` (reshaped back to channel-first,
  an exact transform of the same values, not a recomputation). See
  `docs/SEARCH.md#large-files` and `scripts/slurm_build_features.sh`.
- **`search/run_search.py` + `search/report.py`**: multi-architecture,
  multi-hyperparameter search driver + ranked report generator. Given a
  YAML describing a grid (or random sample) over architecture, input
  representation, depth, width, activation, dropout, batchnorm, loss,
  optimizer, learning rate, scheduler, and seed, trains + evaluates every
  combination, writes one row per run to `results.csv` (flushed after every
  run, so a killed job doesn't lose earlier results), and generates
  `report.md` + comparison plots (best per architecture/input_type, metric
  vs. width, metric vs. depth, predicted-vs-true for the best run).
  Supports SLURM job-array sharding (`--shard-index`/`--shard-count`,
  `scripts/slurm_search.sh`) with automatic merging in `report.py`. See
  `docs/SEARCH.md` and `pipeline/config_search_example.yaml`.
- `models/cnn.py`: implemented `ShowerCNN` (was an empty stub) so the
  search has a second real architecture -- a few Conv2d blocks + global
  average pooling + linear head over the `grid` input representation.
- `experiments/experiment_runner.py`: `build_model()` now handles
  `model.type == "cnn"`; the LR scheduler (`training.scheduler`) is now
  actually built and stepped (previously accepted but never used, see
  Fixed below).
- `data/load_db.py`: `DAT_file(path)` reads a raw CORSIKA `.DAT` file
  directly via the `corsario` package, returning `(df_particles,
  df_shower)` with the same columns as `particles_csv()`/`shower_csv()` --
  no separate CSV-conversion step needed. Optional dependency, see the new
  `corsario` extra in `pyproject.toml`.
- `docs/PIPELINE.md` (config schema, output layout, known gaps),
  `docs/INSTALL_SERVER.md`, `docs/SEARCH.md`.
- `docker/Dockerfile`, `scripts/slurm_train.sh`, `scripts/slurm_search.sh`,
  `scripts/slurm_build_features.sh`.
- `src/pipeline/config_search_mlp_first.yaml`: a ready-to-run, MLP-only,
  modest-grid config (all three input types) for a first validation run
  on the server before committing to the full search grid.
- `tests/test_build_feature_dataset.py`: covers the grid-range fix and the
  build/resume correctness guarantees below, for all three input types
  (`stats`/`grid`/`circle`).
- A warning comment at the top of `pipeline/config_yaml.yaml`: it uses a
  different schema than the one `run_pipeline.py`/`pipeline/utils.py`
  actually read (compare to `config_example.yaml`) and will raise
  `KeyError` if used as-is. Left in place (looks like a draft for a future
  config revision) rather than deleted.

## Fixed
- **`data/features/grid.py`: batch-dependent spatial binning.**
  `grid_generator()` computed its spatial bin edges from
  `df_particles["x"/"y"].min()/.max()` -- fine for a single call over an
  entire dataset, but wrong the moment it's called once per batch while
  streaming a huge file: each batch would define spatially different grid
  cells (batch A's cell `(0,0)` would cover different physical ground than
  batch B's), silently producing an inconsistent "grid" feature across the
  dataset with no error. Found while building the large-file pipeline
  above, before it was used at scale. Fixed by adding optional
  `x_range`/`y_range` parameters (default `None`, i.e. unchanged behavior
  for a single monolithic call); `data/build_feature_dataset.py` computes
  a fixed range once (from a sample or the full file, or an explicit
  physical range) and passes it to every batch. `stats_generator` and
  `circle_generator` were audited for the same class of bug and don't have
  it -- every quantity they compute is already per-shower (via
  `scatter_add` keyed on shower index), not batch-dependent.
- `data/conversion.py`: `convert()` called `torch.from_numpy(np_data,
  dtype=torch.float32)`, which raises `TypeError` (`from_numpy` takes no
  `dtype` argument) on every call. Currently unused by `run_pipeline.py`,
  but fixed so it works if wired back in.
- `pipeline/run_pipeline.py` and `pipeline/utils.py`: 4 f-strings reused the
  same quote character inside `{}` (e.g. `f"...{d["id"]}..."`), which is
  only valid syntax on Python >= 3.12 (PEP 701). Since `pyproject.toml`
  declares `requires-python = ">=3.10"`, these modules would fail to import
  at all on 3.10/3.11 with a `SyntaxError`. Rewritten to use single quotes
  internally, so the code now actually runs on the Python versions the
  package claims to support.
- `pipeline/run_pipeline.py`: replaced the hard-coded
  `sys.path.append(r"/home/icantu24/Documents/ARCHES/src")` with a path
  computed from `__file__`, so the pipeline runs on any machine/checkout,
  not just the original developer's.
- `trainers/trainer.py`: `Trainer.train()` accepted a `scheduler` argument
  but never called `.step()` on it, so `training.scheduler.enabled`/`type`
  in the config had no effect regardless of value. Now steps the scheduler
  every epoch (with the `ReduceLROnPlateau` special case of passing
  `val_loss`).
- `experiments/experiment_runner.py`: a diverged/degenerate run (exploding
  loss, dead units -- something a broad hyperparameter search *will*
  occasionally produce) can crash `plot_results()` deep inside matplotlib's
  histogram binning on NaN/zero-variance residuals (reproduced during
  testing of `search/run_search.py`). That exception used to propagate out
  of `run()` and discard the already-computed, already-saved model and test
  metrics along with it. Plotting failures are now caught and logged
  instead of losing the run's results.
- `pyproject.toml`: fixed the placeholder `Repository` URL
  (`github.com/tu_usuario/ARCHES` -> the real repository).
- `scripts/slurm_build_features.sh`: an earlier version of this script
  passed `--input-types stats grid`, omitting `circle` -- fixed to build
  all three (also now covered by `tests/test_build_feature_dataset.py`,
  which previously only exercised `stats`/`grid` through the batched
  build+resume path).

## Verified end-to-end
- `data/load_db.py`, `data/preprocessing.py`, `data/features/{stats,grid,circle}.py`,
  `data/split_data.py`, `experiments/experiment_runner.py`: ran the full
  path from a synthetic CORSIKA `.DAT` file through `corsario` -> feature
  generation (all three `input_type`s, both MLP and CNN) -> train/val/test
  split -> `ExperimentRunner.run()` (model, optimizer, scheduler, trainer,
  metrics), with no errors and no NaNs on realistic (varying energy/particle
  count) synthetic data.
- `search/run_search.py` + `search/report.py`: ran a 9-run search (MLP x
  {stats, grid} x depth x width, plus 1 CNN run) end to end twice -- once
  reading a raw `.DAT` file, once reading a prebuilt `feature_dir` -- both
  producing the same best configuration; also verified sharded execution
  (`--shard-index`/`--shard-count`, simulating a 3-task SLURM array) with
  `report.py` correctly merging the shards, and that `report.py` is
  idempotent (re-running it does not double-count results).
- `data/build_feature_dataset.py`: verified against a synthetic file that
  (a) batched feature construction matches a single monolithic call
  exactly for all three input types, and (b) interrupting after a few
  batches and resuming produces a byte-identical final dataset to an
  uninterrupted run (`tests/test_build_feature_dataset.py`).

## Known gaps (not changed -- flagged for a design decision, not a bug fix)
- `models/kan.py` is still empty; `model.type: kan` (or `pinn`) is accepted
  by `pipeline/run_pipeline.py`'s config but silently does nothing in
  `ExperimentRunner.build_model()`. `search/run_search.py` guards against
  this (see `search/space.py: IMPLEMENTED_ARCHITECTURES`) by skipping
  unimplemented architectures with a clear message; `run_pipeline.py`
  itself does not have this guard.
- `data_config["output"]` is threaded through as `output_dim` but every
  feature generator hard-codes `total_energy` as the regression target
  regardless of what `output` lists -- see `docs/PIPELINE.md`.
- `experiment_management/{experiment_registry,optuna_manager,pruning,search_spaces}.py`
  and `evaluation/report.py` are empty stubs.
- `data/build_feature_dataset.py`'s resume works by re-parsing (but not
  re-computing features for) already-checkpointed batches from the start
  of the file, since the CORSIKA format has no index to seek into
  directly. For most files this overhead is small relative to feature
  computation + training time; if it becomes a bottleneck on a very large
  file with many resume cycles, a companion "shower index" (event number ->
  byte offset), built once, would let corsario seek directly instead.
