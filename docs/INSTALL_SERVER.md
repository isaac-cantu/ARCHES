# Server / HPC installation and training

## Quick install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,corsario]"
```

`torch`/`torchvision`/`torchaudio` are pulled in as regular dependencies
(CPU wheels by default on most platforms via PyPI). For a CUDA build on a
GPU node, install torch from the appropriate index **before** `pip install
-e .` so pip does not later try to replace it with a CPU-only wheel, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev,corsario]"
```

## MLflow tracking

`experiment_management/mlflow_tracker.py` calls `mlflow.set_experiment(...)`
without an explicit tracking URI, so it defaults to a local `./mlruns`
directory wherever the pipeline is launched from. For a persistent,
queryable store across runs (the repository already ships an `mlflow.db`,
suggesting this is the intended setup), set:

```bash
export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
```

before running `run_pipeline.py`, and browse results with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To run a batch of experiments without MLflow at all (e.g. quick local
smoke tests), instantiate `ExperimentRunner(model_data, training_data,
use_mlflow=False)` directly instead of going through `run_pipeline.py`
(which currently always uses the default `use_mlflow=True`).

## Container

`docker/Dockerfile` builds an image with ARCHES and its dependencies
(including `corsario`) installed:

```bash
docker build -t arches -f docker/Dockerfile .
docker run --rm -v /data:/data -v $(pwd)/experiments:/experiments arches \
    python pipeline/run_pipeline.py --configs pipeline/config_example.yaml
```

Mount your CORSIKA/processed data directory and an experiments output
directory as volumes; update `data.path` / `experiment.path` in the config
(or override them via environment/CLI before launch) to point at the
mounted paths rather than the original developer's home directory.

## SLURM

`scripts/slurm_train.sh` submits one `run_pipeline.py` invocation per
config file — this is the natural unit of parallelism here since
`run_pipeline.py` already sweeps `search_space` (layers/neurons/seed/
activation) and `batch_size` internally in a single process. Submitting one
job per **config file** (e.g. one per `input_type`, or one per loss
function you want to compare) lets independent sweeps run concurrently
without touching the pipeline's internal loop:

```bash
sbatch scripts/slurm_train.sh pipeline/config_example.yaml
```

For a large hyperparameter search where the internal nested loop becomes
the bottleneck (many hours in one job), the cleanest next step is to first
address the "no Optuna/pruning yet" gap noted in the README, and dispatch
one job per hyperparameter *point* rather than growing the SLURM script
around the current nested-loop structure.

## Millions of events

If you're working with CORSIKA output at the scale of millions of showers
(one or a few large `.DAT` files), do not point `run_pipeline.py` or
`search/run_search.py` directly at them — see
`docs/SEARCH.md#large-files` first. In short:

1. `sbatch scripts/slurm_build_features.sh` once (a multi-hour job that
   streams the raw file and writes compact per-shower features; resumable
   if it hits the time limit — just resubmit). Budget **~64-128 GB RAM**
   for this step even though the *output* is small (a few GB) — see the
   memory math in `corsario`'s `docs/LARGE_FILES.md` for why the raw
   particle table itself does not fit at this scale, and how streaming
   avoids needing it to.
2. `sbatch scripts/slurm_search.sh` (or `scripts/slurm_train.sh`) against
   the resulting `feature_dir` — these read a few-GB HDF5 file, not the
   raw `.DAT`, so they stay fast and can be rerun freely while you iterate
   on the hyperparameter grid.
