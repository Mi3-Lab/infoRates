#!/bin/bash
#SBATCH --job-name=sw_vmamba
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=evaluations/accv2026/logs/sw_vmamba_%j.out
#SBATCH --error=evaluations/accv2026/logs/sw_vmamba_%j.err

set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
source .venv_mamba/bin/activate

mkdir -p evaluations/accv2026/logs

echo "=== Sliding Window — VideoMamba × 7 datasets ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python3 -u scripts/accv2026/e3_sw_videomamba.py

echo "Done: $(date)"
