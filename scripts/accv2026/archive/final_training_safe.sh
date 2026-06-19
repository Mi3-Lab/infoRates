#!/bin/bash
# Final training with SAFE settings (reduced batch size to avoid OOM)
set -euo pipefail
cd /mnt/datasets/infoRates
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

LOG_DIR="evaluations/accv2026/logs"
mkdir -p "$LOG_DIR"

run_job() {
    local label="$1"; shift
    local logfile="$LOG_DIR/${label}.log"
    echo "[$(date)] ▶ $label"
    if python "$@" >"$logfile" 2>&1; then
        echo "[$(date)] ✓ $label"
    else
        echo "[$(date)] ✗ $label FAILED — last 30 lines:"
        tail -30 "$logfile"
    fi
    echo ""
}

# ─────────────────────────────────────────────────────────
# SlowFast: REDUCED batch size (16 instead of 32 to avoid OOM)
# ─────────────────────────────────────────────────────────
for DATASET in ufc_crime finegym; do
    run_job "slowfast_${DATASET}" \
        scripts/accv2026/train_slowfast.py \
        --dataset "$DATASET" \
        --epochs 10 --batch-size 16 --num-workers 6 \
        --wandb-tags accv2026 dataset_expansion "$DATASET"
done

# ─────────────────────────────────────────────────────────
# Transformers: TimeSformer, VideoMAE, ViViT on UCFCrime + FineGym
# Safe batch size 24
# ─────────────────────────────────────────────────────────
for MODEL in timesformer videomae vivit; do
    for DATASET in ufc_crime finegym; do
        run_job "transformer_${MODEL}_${DATASET}" \
            scripts/accv2026/train_transformers.py \
            --dataset "$DATASET" --model "$MODEL" \
            --epochs 10 --batch-size 24 --num-workers 6 \
            --wandb-tags accv2026 dataset_expansion "$DATASET"
    done
done

echo "[$(date)] ══════════════════════════════════════════════"
echo "[$(date)] ALL TRAINING COMPLETE"
echo "[$(date)] ══════════════════════════════════════════════"
