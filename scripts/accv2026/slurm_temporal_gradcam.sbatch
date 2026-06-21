#!/bin/bash
#SBATCH --job-name=gradcam_all
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=evaluations/accv2026/logs/gradcam_all_%j.out
#SBATCH --error=evaluations/accv2026/logs/gradcam_all_%j.err

set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
source .venv/bin/activate

mkdir -p evaluations/accv2026/logs

echo "=== Temporal GradCAM — all 7 models × all 7 datasets ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python3 scripts/accv2026/e3_temporal_gradcam_all_models.py

echo "Done: $(date)"
