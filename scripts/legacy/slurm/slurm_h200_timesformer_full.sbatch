#!/usr/bin/env bash
#SBATCH --job-name=accv26-h200-timesformer
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
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
export RESUME_FROM="${RESUME_FROM:-fine_tuned_models/accv2026_timesformer_ssv2_full_e10_h200_epoch1}"
export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"

srun --ntasks=1 --cpus-per-task=16 --gpus=1 \
  bash scripts/accv2026/run_h200_ssv2_timesformer_full.sh
