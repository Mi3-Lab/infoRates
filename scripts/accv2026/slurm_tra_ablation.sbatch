#!/usr/bin/env bash
#SBATCH --job-name=tra-ablation
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err
set -uo pipefail

REPO=/data/wesleyferreiramaia/infoRates
cd /scratch/wesleyferreiramaia/infoRates

source "${REPO}/.venv/bin/activate"
export PYTHONPATH="${REPO}/src"
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export HF_HOME="${HF_HOME:-/scratch/wesleyferreiramaia/infoRates/hf_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT=inforates-accv2026
export WANDB_DIR=/scratch/wesleyferreiramaia/infoRates/wandb

MODEL="${MODEL:?MODEL must be set}"
DATASET="${DATASET:?DATASET must be set}"
ARM="${ARM:?ARM must be set (paper|fixed)}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EXTRA="${EXTRA:-}"

echo "=== TRA ablation — job ${SLURM_JOB_ID} ==="
echo "model=${MODEL} dataset=${DATASET} arm=${ARM} epochs=${EPOCHS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python "${REPO}/scripts/accv2026/train_with_tra.py" \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --arm "${ARM}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers 6 \
    ${EXTRA}

echo "=== DONE $(date) ==="
