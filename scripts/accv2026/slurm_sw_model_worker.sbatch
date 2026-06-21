#!/bin/bash
#SBATCH --job-name=sw_%x
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=evaluations/accv2026/logs/sw_worker_%x_%j.out
#SBATCH --error=evaluations/accv2026/logs/sw_worker_%x_%j.err

set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
source .venv/bin/activate

mkdir -p evaluations/accv2026/logs

MODEL="${MODEL_KEY:-}"
if [ -z "$MODEL" ]; then
  echo "Erro: MODEL_KEY não definida"; exit 1
fi

echo "=== Sliding Window Worker: $MODEL × 7 datasets ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"

python3 -u scripts/accv2026/e3_sw_model_worker.py "$MODEL"

echo "Done: $(date)"
