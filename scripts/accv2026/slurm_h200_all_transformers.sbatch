#!/usr/bin/env bash
#SBATCH --job-name=accv26-h200-all-transformers
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=24
#SBATCH --time=1-06:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

# TimeSformer + VideoMAE em paralelo (GPU 0 e 1), depois ViViT (GPU 0).
# Wall time: ~5h paralelo + ~10h ViViT = 15h. 30h de margem.

set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

export WANDB_MODE="${WANDB_MODE:-online}"
export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export VIVIT_BATCH_SIZE="${VIVIT_BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-12}"
export WANDB_PROJECT="${WANDB_PROJECT:-inforates-accv2026}"

srun --ntasks=1 --cpus-per-task=24 --gpus=2 \
  bash scripts/accv2026/run_h200_ssv2_all_transformers.sh
