#!/bin/bash
#SBATCH --job-name=sw_all_models
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=evaluations/accv2026/logs/sw_all_models_%j.out
#SBATCH --error=evaluations/accv2026/logs/sw_all_models_%j.err

set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
source .venv/bin/activate

mkdir -p evaluations/accv2026/logs

echo "=== Sliding Window — todos os 7 modelos × 7 datasets ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python3 -u scripts/accv2026/e3_sliding_window_all_models.py

echo "Done: $(date)"
