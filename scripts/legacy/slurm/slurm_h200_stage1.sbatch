#!/usr/bin/env bash
#SBATCH --job-name=accv26-h200-s1
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --time=08:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"
export WANDB_MODE="${WANDB_MODE:-online}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

srun --ntasks=1 --cpus-per-task=12 --gpus=1 \
  bash scripts/accv2026/run_h200_ssv2_stage1.sh

