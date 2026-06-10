#!/bin/bash
# Run all remaining ACCV 2026 training jobs after VideoMamba UCFCrime finishes.
# Usage:
#   bash scripts/accv2026/run_remaining_training.sh           # run immediately
#   bash scripts/accv2026/run_remaining_training.sh <PID>     # wait for PID first

set -euo pipefail
cd /mnt/datasets/infoRates
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

LOG_DIR="evaluations/accv2026/logs"
mkdir -p "$LOG_DIR"

# Wait for previous job if PID given
WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
    echo "[$(date)] Waiting for PID $WAIT_PID to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
    echo "[$(date)] PID $WAIT_PID finished. Starting remaining training."
fi

run_job() {
    local label="$1"; shift
    local logfile="$LOG_DIR/${label}.log"
    echo ""
    echo "[$(date)] ▶ $label"
    echo "  cmd: $*"
    if python "$@" >"$logfile" 2>&1; then
        echo "[$(date)] ✓ $label"
    else
        echo "[$(date)] ✗ $label FAILED — see $logfile"
    fi
}

# ─────────────────────────────────────────────────────────
# VideoMamba — FineGym only (UCFCrime already running)
# ─────────────────────────────────────────────────────────
run_job "videomamba_finegym" \
    scripts/accv2026/train_videomamba.py \
    --dataset finegym --epochs 15 --batch-size 8 --lr 5e-6 \
    --wandb-tags accv2026 dataset_expansion finegym

# ─────────────────────────────────────────────────────────
# 3D CNNs — FineGym only (FLAME + UCFCrime already done)
# ─────────────────────────────────────────────────────────
for MODEL in r3d_18 mc3_18 r2plus1d_18; do
    run_job "torchvision_${MODEL}_finegym" \
        scripts/accv2026/train_torchvision.py \
        --dataset finegym --model "$MODEL" \
        --epochs 10 --batch-size 64 --num-frames 16 --num-workers 12 \
        --wandb-tags accv2026 dataset_expansion finegym
done

# ─────────────────────────────────────────────────────────
# SlowFast — UCFCrime + FineGym (FLAME deferred)
# ─────────────────────────────────────────────────────────
for DATASET in ufc_crime finegym; do
    run_job "slowfast_${DATASET}" \
        scripts/accv2026/train_slowfast.py \
        --dataset "$DATASET" \
        --epochs 10 --batch-size 32 --num-workers 12 \
        --wandb-tags accv2026 dataset_expansion "$DATASET"
done

# ─────────────────────────────────────────────────────────
# Transformers: TimeSformer, VideoMAE, ViViT — UCFCrime + FineGym (FLAME deferred)
# ─────────────────────────────────────────────────────────
for MODEL in timesformer videomae vivit; do
    for DATASET in ufc_crime finegym; do
        run_job "transformer_${MODEL}_${DATASET}" \
            scripts/accv2026/train_transformers.py \
            --dataset "$DATASET" --model "$MODEL" \
            --epochs 10 --batch-size 32 --num-workers 12 \
            --wandb-tags accv2026 dataset_expansion "$DATASET"
    done
done

echo ""
echo "[$(date)] ═══════════════════════════════════════════════"
echo "[$(date)] ALL REMAINING TRAINING COMPLETE"
echo "[$(date)] ═══════════════════════════════════════════════"
echo "Checkpoints: fine_tuned_models/"
echo "Logs:        $LOG_DIR/"
