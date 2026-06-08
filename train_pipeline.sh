#!/bin/bash
set -e
cd /mnt/datasets/infoRates
source .venv/bin/activate

echo "[$(date)] FLAME + UCFCrime Training Pipeline Started"

echo "[$(date)] Phase 1/2: FLAME Training (8 models, ~1.5h)"
python scripts/accv2026/train_all_models.py --dataset flame --epochs 10

echo "[$(date)] Phase 2/2: UCFCrime Training (8 models, ~1.5h)"
python scripts/accv2026/train_all_models.py --dataset ufc_crime --epochs 10

echo "[$(date)] Training Pipeline Complete!"
echo "Checkpoints saved to: fine_tuned_models/"
