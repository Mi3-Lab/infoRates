#!/usr/bin/env bash
#SBATCH --job-name=accv26-videomamba-all
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

export ACCV_JOB_ID="${SLURM_JOB_ID}"
export WANDB_MODE="${WANDB_MODE:-online}"
export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
# Override which datasets to run (default: all 7)
export DATASETS="${DATASETS:-ssv2 ucf101 hmdb51 driveact diving48 autsl epic_kitchens}"

echo "=== VideoMamba all datasets — job ${SLURM_JOB_ID} ==="
echo "Node: $(hostname) | $(date)"
nvidia-smi | head -10 || true

bash scripts/accv2026/run_h200_dataset_all_videomamba.sh

echo "=== VideoMamba all datasets COMPLETE — $(date) ==="
