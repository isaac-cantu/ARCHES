#!/bin/bash
# Submit one run_pipeline.py sweep (a whole config file's search_space) as a
# SLURM job. run_pipeline.py already loops over layers/neurons/seed/
# activation/batch_size internally, so the natural way to parallelize
# further on a cluster is one job per CONFIG FILE (e.g. one per input_type,
# or one per loss function you're comparing) -- see docs/INSTALL_SERVER.md.
#
# Usage:
#   sbatch scripts/slurm_train.sh pipeline/config_example.yaml
#
#SBATCH --job-name=arches_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/arches_train_%j.out
# For a GPU node, uncomment and adjust:
# #SBATCH --gres=gpu:1

set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/slurm_train.sh <path/to/config.yaml>}"
VENV_ACTIVATE="/path/to/.venv/bin/activate"   # <-- created via: pip install -e ".[dev,corsario]"

mkdir -p logs
source "$VENV_ACTIVATE"

export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"

cd src
echo "[$SLURM_JOB_ID] running $CONFIG"
python pipeline/run_pipeline.py --configs "$CONFIG"
echo "[$SLURM_JOB_ID] done"
