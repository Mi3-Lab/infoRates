#!/usr/bin/env bash
#SBATCH --job-name=trainres-sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

# Coverage×Stride sweep on resolution-retrained checkpoints.
# Required env: MODEL, DATASET, TRAIN_RES
set -euo pipefail
# CWD must be /scratch so relative paths in manifests (data/UCF101_data/...) resolve correctly
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export HF_HOME="${HF_HOME:-/scratch/wesleyferreiramaia/hf_unified}"
export WANDB_MODE=offline

MODEL="${MODEL:?MODEL must be set}"
DATASET="${DATASET:?DATASET must be set}"
TRAIN_RES="${TRAIN_RES:?TRAIN_RES must be set}"

if [[ "$MODEL" == "videomamba" ]]; then
    source /data/wesleyferreiramaia/infoRates/.venv_mamba/bin/activate
else
    source /data/wesleyferreiramaia/infoRates/.venv/bin/activate
fi

echo "=== TrainRes Sweep — ${MODEL} / ${DATASET} @ train_res=${TRAIN_RES}px — job ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true

python /data/wesleyferreiramaia/infoRates/scripts/accv2026/sweep_coverage_stride.py \
    --model    "${MODEL}" \
    --dataset  "${DATASET}" \
    --train-res "${TRAIN_RES}" \
    --batch-size 32

echo "=== DONE — $(date) ==="
