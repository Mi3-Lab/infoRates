#!/usr/bin/env bash
#SBATCH --job-name=accv26-h200-videomae-hmdb51
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

export DATASET=hmdb51
export WANDB_MODE="${WANDB_MODE:-online}"
export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"

srun --ntasks=1 --cpus-per-task=8 --gpus=1 \
  bash scripts/accv2026/run_h200_multidata_videomae.sh
