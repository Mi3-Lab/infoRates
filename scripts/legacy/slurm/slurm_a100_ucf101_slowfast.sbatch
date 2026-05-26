#!/usr/bin/env bash
#SBATCH --job-name=accv26-a100-slowfast-ucf101
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

export DATASET=ucf101
export WANDB_MODE="${WANDB_MODE:-online}"
export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export NPROC_PER_NODE=1
export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"

srun --ntasks=1 --cpus-per-task=6 --gpus=1 \
  bash scripts/accv2026/run_a100_multidata_slowfast.sh
