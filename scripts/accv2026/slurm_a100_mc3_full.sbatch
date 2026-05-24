#!/usr/bin/env bash
#SBATCH --job-name=accv26-a100-mc3-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"
export WANDB_MODE="${WANDB_MODE:-online}"
export EPOCHS="${EPOCHS:-5}"
export BATCH_SIZE="${BATCH_SIZE:-24}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

srun --ntasks=1 --cpus-per-task=16 --gpus=2 \
  bash scripts/accv2026/run_a100_ssv2_mc3_full.sh
