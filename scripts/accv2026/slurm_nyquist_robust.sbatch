#!/bin/bash
#SBATCH --job-name=nyquist_robust
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=evaluations/accv2026/logs/nyquist_robust_%j.out
#SBATCH --error=evaluations/accv2026/logs/nyquist_robust_%j.err

set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
source .venv/bin/activate

mkdir -p evaluations/accv2026/logs

echo "=== Nyquist Robust Tests A+B+C — all 7 datasets ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python3 scripts/accv2026/e3_nyquist_robust_all_datasets.py

echo "Done: $(date)"
