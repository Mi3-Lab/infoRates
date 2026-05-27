#!/usr/bin/env bash
#SBATCH --job-name=accv26-vmamba-%x
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluations/accv2026/logs/videomamba-%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/videomamba-%x-%j.err
# Submit with: sbatch --job-name=<dataset> --export=ALL,DATASET=<dataset> this_file.sbatch
set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

echo "=== VideoMamba: ${DATASET} — job ${SLURM_JOB_ID} ==="
echo "Node: $(hostname) | $(date)"
nvidia-smi | head -6 || true

export ACCV_JOB_ID="${SLURM_JOB_ID}"
export WANDB_MODE="${WANDB_MODE:-online}"
export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export HF_HOME="${HF_HOME:-/scratch/wesleyferreiramaia/hf_unified}"

bash scripts/accv2026/run_h200_multidata_videomamba.sh

echo "=== VideoMamba: ${DATASET} DONE — $(date) ==="
