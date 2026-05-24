#!/usr/bin/env bash
#SBATCH --job-name=accv26-a100-vmae-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-12}"

srun --ntasks=1 --cpus-per-task=12 --gpus=1 \
  bash scripts/accv2026/run_a100_eval_videomae_10k.sh
