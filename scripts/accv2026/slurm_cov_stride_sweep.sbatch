#!/usr/bin/env bash
#SBATCH --job-name=cov-stride-sweep
#SBATCH --partition=gpu
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
export WANDB_MODE=online
export WANDB_PROJECT=inforates-accv2026

# MODEL and DATASET passed via environment (sbatch --export)
MODEL="${MODEL:?MODEL must be set}"
DATASET="${DATASET:?DATASET must be set}"

# VideoMamba needs .venv_mamba, everything else uses .venv
if [[ "$MODEL" == "videomamba" ]]; then
    source .venv_mamba/bin/activate
    export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
else
    source .venv/bin/activate
fi

echo "=== Coverage×Stride Sweep — ${MODEL} / ${DATASET} — job ${SLURM_JOB_ID} ==="
nvidia-smi | head -6 || true

python scripts/accv2026/sweep_coverage_stride.py \
    --model   "${MODEL}" \
    --dataset "${DATASET}" \
    --batch-size 32

echo "=== DONE ==="
