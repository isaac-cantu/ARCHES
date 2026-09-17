#!/bin/bash
# Build compact per-shower feature datasets from a huge CORSIKA .DAT file
# (millions of showers) -- a one-time, potentially multi-hour job, kept
# separate from slurm_search.sh (which reruns cheaply against the result).
#
# Resumable: if this job hits the time limit or gets killed, resubmitting
# the exact same command picks up where it left off (the HDF5 output's row
# count is the checkpoint) -- see docs/SEARCH.md#large-files. So it's safe
# (and often necessary for very large files) to set --time below to your
# cluster's max and just resubmit if it doesn't finish in one go.
#
# Usage:
#   1. Edit DAT_FILE / OUT_DIR / VENV_ACTIVATE / --time below.
#   2. sbatch scripts/slurm_build_features.sh
#   3. If it times out before finishing: sbatch scripts/slurm_build_features.sh
#      again (same command) to resume.
#
#SBATCH --job-name=arches_build_features
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/arches_build_features_%j.out
# If you have a GPU node available, uncomment both lines below --
# scatter_add-based feature computation benefits from it on large files:
# #SBATCH --gres=gpu:1
# DEVICE=cuda

DAT_FILE="/path/to/DAT_15M"
OUT_DIR="/path/to/features/run1"
VENV_ACTIVATE="/path/to/.venv/bin/activate"   # <-- created via: pip install -e ".[dev,corsario]"
DEVICE="${DEVICE:-cpu}"

set -euo pipefail
mkdir -p logs
source "$VENV_ACTIVATE"
cd src

echo "[$SLURM_JOB_ID] building features from $DAT_FILE -> $OUT_DIR (device=$DEVICE)"
python data/build_feature_dataset.py \
    --dat-file "$DAT_FILE" \
    --out-dir "$OUT_DIR" \
    --input-types stats grid circle \
    --batch-size 50000 \
    --device "$DEVICE" \
    --tolerant
echo "[$SLURM_JOB_ID] done (or hit the time limit -- resubmit to resume)"
