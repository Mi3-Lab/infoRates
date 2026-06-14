#!/usr/bin/env bash
#SBATCH --job-name=res-sweep
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE=disabled  # too many sweep runs for W&B
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export PYTHONUNBUFFERED=1

# Required: MODEL, DATASET, TRAIN_RES (px)
MODEL="${MODEL:?MODEL must be set}"
DATASET="${DATASET:?DATASET must be set}"
TRAIN_RES="${TRAIN_RES:?TRAIN_RES must be set}"

# VideoMamba needs .venv_mamba
if [[ "$MODEL" == "videomamba" ]]; then
    source .venv_mamba/bin/activate
else
    source .venv/bin/activate
fi

echo "=== Resolution Coverage×Stride Sweep — ${MODEL} / ${DATASET} @ ${TRAIN_RES}px — job ${SLURM_JOB_ID} ==="
nvidia-smi | head -6 || true

python scripts/accv2026/sweep_coverage_stride.py \
    --model      "${MODEL}" \
    --dataset    "${DATASET}" \
    --train-res  "${TRAIN_RES}" \
    --batch-size 64

echo "=== DONE — $(date) ==="
