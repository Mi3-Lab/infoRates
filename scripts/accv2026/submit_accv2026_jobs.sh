#!/usr/bin/env bash
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python scripts/accv2026/check_wandb_login.py

mkdir -p evaluations/accv2026/logs

echo "[submit] W&B is ready. Submitting ACCV 2026 jobs."

echo "[submit] H200 stage 1"
sbatch scripts/accv2026/slurm_h200_stage1.sbatch

echo "[submit] A100 VideoMAE pilot"
sbatch scripts/accv2026/slurm_a100_videomae_pilot.sbatch

echo "[submit] Current queue"
squeue -u "$USER"

