#!/bin/bash
# Run the architecture/hyperparameter search (search/run_search.py) as a
# SLURM job array, sharding the full combination grid across N_SHARDS
# tasks that train in parallel, then merge into one report.
#
# Usage:
#   1. Edit CONFIG / VENV_ACTIVATE / #SBATCH --array below.
#      To size the array: python search/run_search.py --config <cfg> --dry-run
#      prints the total number of planned runs; pick N_SHARDS so that
#      (total_runs / N_SHARDS) finishes comfortably within --time.
#   2. sbatch scripts/slurm_search.sh
#   3. Once all array tasks finish:
#        python search/report.py --search-dir <search.path>/search_<id>
#
#SBATCH --job-name=arches_search
#SBATCH --array=0-7                # <-- N_SHARDS - 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/arches_search_%A_%a.out
# For GPU nodes, uncomment:
# #SBATCH --gres=gpu:1

set -euo pipefail

CONFIG="pipeline/config_search_example.yaml"
VENV_ACTIVATE="/path/to/.venv/bin/activate"   # <-- created via: pip install -e ".[dev,corsario]"
# SLURM_ARRAY_TASK_COUNT requires a fairly recent SLURM; compute it from
# TASK_MIN/TASK_MAX instead for portability to older clusters.
N_SHARDS="$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))"
SHARD_INDEX="$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))"

mkdir -p logs
source "$VENV_ACTIVATE"
cd src

echo "[$SLURM_ARRAY_TASK_ID] shard $SHARD_INDEX / $N_SHARDS"
python search/run_search.py \
    --config "$CONFIG" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$N_SHARDS"
echo "[$SLURM_ARRAY_TASK_ID] done"
