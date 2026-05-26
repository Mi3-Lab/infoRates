#!/usr/bin/env bash
#SBATCH --job-name=accv26-a100-mc3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err

set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"
export WANDB_MODE="${WANDB_MODE:-online}"
export ACCV_JOB_ID="${SLURM_JOB_ID:-manual}"
export TRAIN_GPU=0
export EVAL_GPU=0
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"

bash scripts/accv2026/run_a100_ssv2_mc3_pilot.sh
