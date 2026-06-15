#!/usr/bin/env bash
#SBATCH --job-name=native48
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=evaluations/accv2026/logs/native48-%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/native48-%x-%j.err
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE=disabled
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export PYTHONUNBUFFERED=1

MODEL="${MODEL:?MODEL must be set}"
DATASET="${DATASET:?DATASET must be set}"

if [[ "$MODEL" == "videomamba" ]]; then
    source .venv_mamba/bin/activate
else
    source .venv/bin/activate
fi

echo "=== Native @ 48px sweep — ${MODEL} / ${DATASET} — job ${SLURM_JOB_ID} ==="
nvidia-smi | head -6 || true

python scripts/accv2026/sweep_coverage_stride.py \
    --model      "${MODEL}" \
    --dataset    "${DATASET}" \
    --input-size 48 \
    --batch-size 64

echo "=== DONE — $(date) ==="
