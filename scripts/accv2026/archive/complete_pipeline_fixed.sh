#!/bin/bash
# Complete pipeline with pin_memory fix
set -euo pipefail
cd /mnt/datasets/infoRates
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

LOG_DIR="evaluations/accv2026/logs"
mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════"
echo "PIPELINE COMPLETO - pin_memory DESABILITADO"
echo "════════════════════════════════════════════════════"

# SlowFast
echo "[$(date)] ▶ SlowFast UCFCrime"
python scripts/accv2026/train_slowfast.py \
  --dataset ufc_crime --epochs 10 --batch-size 24 --num-workers 6 \
  --wandb-tags accv2026 dataset_expansion ufc_crime \
  > "$LOG_DIR/slowfast_ufc_crime_fix.log" 2>&1 && echo "[$(date)] ✓" || echo "[$(date)] ✗"

echo "[$(date)] ▶ SlowFast FineGym"
python scripts/accv2026/train_slowfast.py \
  --dataset finegym --epochs 10 --batch-size 24 --num-workers 6 \
  --wandb-tags accv2026 dataset_expansion finegym \
  > "$LOG_DIR/slowfast_finegym_fix.log" 2>&1 && echo "[$(date)] ✓" || echo "[$(date)] ✗"

# Transformers
for MODEL in timesformer videomae vivit; do
  for DATASET in ufc_crime finegym; do
    echo "[$(date)] ▶ ${MODEL} × ${DATASET}"
    python scripts/accv2026/train_transformers.py \
      --dataset "$DATASET" --model "$MODEL" \
      --epochs 10 --batch-size 24 --num-workers 4 \
      --wandb-tags accv2026 dataset_expansion "$DATASET" \
      > "$LOG_DIR/transformer_${MODEL}_${DATASET}_fix.log" 2>&1 && echo "[$(date)] ✓" || echo "[$(date)] ✗"
  done
done

echo ""
echo "════════════════════════════════════════════════════"
echo "[$(date)] PIPELINE COMPLETO!"
echo "════════════════════════════════════════════════════"
