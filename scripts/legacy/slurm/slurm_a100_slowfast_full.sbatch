#!/usr/bin/env bash
#SBATCH --job-name=accv26-a100-slowfast
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=1-06:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

export WANDB_MODE="${WANDB_MODE:-online}"
export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export NPROC_PER_NODE=2
export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"

srun --ntasks=1 --cpus-per-task=16 --gpus=2 \
  bash scripts/accv2026/run_a100_ssv2_slowfast_full.sh
