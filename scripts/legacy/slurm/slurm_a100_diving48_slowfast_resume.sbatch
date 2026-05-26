#!/usr/bin/env bash
#SBATCH --job-name=accv26-a100-slowfast-diving48-resume
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

export DATASET=diving48
export WANDB_MODE="${WANDB_MODE:-online}"
export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"
# Resume from best epoch saved before timeout
export RESUME_FROM=fine_tuned_models/accv2026_slowfast_r50_diving48_full_e10_a100_epoch7

srun --ntasks=1 --cpus-per-task=8 --gpus=1 \
  bash scripts/accv2026/run_a100_multidata_slowfast.sh
