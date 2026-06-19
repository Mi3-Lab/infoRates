#!/usr/bin/env bash
#SBATCH --job-name=accv26-preprocess-autsl
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err

set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONPATH=src

echo "======================================================="
echo "AUTSL preprocessing — $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================="

python scripts/accv2026/preprocess_autsl.py \
    --data-root data/AUTSL_data

echo "======================================================="
echo "Done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Train: $(wc -l < data/AUTSL_data/splits/train.csv 2>/dev/null || echo '?') clips"
echo "  Val:   $(wc -l < data/AUTSL_data/splits/val.csv   2>/dev/null || echo '?') clips"
echo "======================================================="
