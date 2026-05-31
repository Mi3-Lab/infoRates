#!/usr/bin/env bash
#SBATCH --job-name=spatial-res-sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE=online
export WANDB_PROJECT=inforates-accv2026

MODEL="${MODEL:?MODEL must be set}"
DATASET="${DATASET:?DATASET must be set}"

if [[ "$MODEL" == "videomamba" ]]; then
    source .venv_mamba/bin/activate
    export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
else
    source .venv/bin/activate
fi

echo "=== E6 Spatial Resolution Sweep — ${MODEL} / ${DATASET} — job ${SLURM_JOB_ID} ==="
nvidia-smi | head -6 || true

python scripts/accv2026/sweep_spatial_resolution.py \
    --model   "${MODEL}" \
    --dataset "${DATASET}" \
    --batch-size 32

echo "=== DONE ==="
