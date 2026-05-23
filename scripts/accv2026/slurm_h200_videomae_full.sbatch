#!/usr/bin/env bash
#SBATCH --job-name=accv26-h200-videomae-full
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export EPOCHS="${EPOCHS:-5}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-24}"
export NUM_WORKERS="${NUM_WORKERS:-12}"

srun --ntasks=1 --cpus-per-task=16 --gpus=1 \
  bash scripts/accv2026/run_h200_ssv2_videomae_full.sh
