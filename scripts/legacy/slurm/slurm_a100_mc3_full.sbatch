#!/usr/bin/env bash
#SBATCH --job-name=accv26-a100-mc3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
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
export NPROC_PER_NODE=1
export RESUME_FROM="${RESUME_FROM:-fine_tuned_models/accv2026_mc3_18_ssv2_full_e10_a100_epoch5}"
export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"

srun --ntasks=1 --cpus-per-task=10 --gpus=1 \
  bash scripts/accv2026/run_a100_ssv2_mc3_full.sh
