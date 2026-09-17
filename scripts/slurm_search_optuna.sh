#!/bin/bash
# Run an Optuna (TPE + pruning) hyperparameter search as a SLURM job array,
# with all array tasks contributing trials to ONE shared study (via a
# shared SQLite file) instead of each exploring independently -- so the
# sampler benefits from every trial any worker has finished, not just its
# own. See docs/SEARCH.md#optuna.
#
# Usage:
#   1. Set search.optuna.storage in your config to a shared sqlite path,
#      e.g. "sqlite:////scratch/$USER/arches/optuna.db" (note: 4 slashes
#      for an absolute path with the sqlite:/// scheme).
#   2. Edit CONFIG / VENV_ACTIVATE / #SBATCH --array (N_WORKERS - 1) below.
#      Total trials run = N_WORKERS * (search.optuna.n_trials in the config)
#      -- e.g. 8 workers x 25 trials each = 200 trials total.
#   3. sbatch scripts/slurm_search_optuna.sh
#   4. python search/report.py --search-dir <search.path>/search_<id>
#      (merges every results_shard_*.csv automatically)
#
#SBATCH --job-name=arches_search_optuna
#SBATCH --array=0-7                # <-- N_WORKERS - 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/arches_search_optuna_%A_%a.out

set -euo pipefail

CONFIG="pipeline/config_search_optuna_example.yaml"
VENV_ACTIVATE="/path/to/.venv/bin/activate"   # <-- created via: pip install -e ".[dev,corsario,optuna]"
N_WORKERS="$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))"
WORKER_ID="$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))"

mkdir -p logs
source "$VENV_ACTIVATE"
cd src

echo "[$SLURM_ARRAY_TASK_ID] optuna worker $WORKER_ID / $N_WORKERS (shared study)"
python search/run_search.py \
    --config "$CONFIG" \
    --shard-index "$WORKER_ID" \
    --shard-count "$N_WORKERS"
echo "[$SLURM_ARRAY_TASK_ID] done"
