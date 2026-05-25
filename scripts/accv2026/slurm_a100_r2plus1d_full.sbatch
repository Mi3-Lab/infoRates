#!/usr/bin/env bash
#SBATCH --job-name=accv26-a100-r2plus1d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

export WANDB_MODE="${WANDB_MODE:-online}"
export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export NPROC_PER_NODE=2
export RESUME_FROM="${RESUME_FROM:-fine_tuned_models/accv2026_r2plus1d_18_ssv2_full_e10_a100_epoch3}"
export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"

srun --ntasks=1 --cpus-per-task=16 --gpus=2 \
  bash scripts/accv2026/run_a100_ssv2_r2plus1d_full.sh
