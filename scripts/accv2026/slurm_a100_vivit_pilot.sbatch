#!/usr/bin/env bash
#SBATCH --job-name=accv26-a100-vivit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export NUM_WORKERS="${NUM_WORKERS:-6}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-5000}"
export MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-1000}"

srun --ntasks=1 --cpus-per-task=12 --gpus=2 \
  bash scripts/accv2026/run_a100_ssv2_vivit_pilot_ddp.sh
