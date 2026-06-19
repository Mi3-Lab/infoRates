#!/bin/bash
# Final stable training - skip SlowFast (too memory intensive), focus on Transformers
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
    if timeout 3600 python "$@" >"$logfile" 2>&1; then
        echo "[$(date)] ✓ $label"
    else
        ret=$?
        if [ $ret -eq 124 ]; then
            echo "[$(date)] ⏱ $label TIMEOUT (1h)"
        else
            echo "[$(date)] ✗ $label FAILED (exit $ret)"
        fi
    fi
    echo ""
}

echo "[$(date)] ════════════════════════════════════════════════════"
echo "[$(date)] FINAL STABLE TRAINING - TRANSFORMERS ONLY"
echo "[$(date)] (Skipping SlowFast due to memory constraints)"
echo "[$(date)] ════════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────────────────────────
# Transformers: TimeSformer, VideoMAE, ViViT on UCFCrime + FineGym
# Safe batch size 16, 4 workers
# ─────────────────────────────────────────────────────────

for MODEL in timesformer videomae vivit; do
    for DATASET in ufc_crime finegym; do
        run_job "transformer_${MODEL}_${DATASET}" \
            scripts/accv2026/train_transformers.py \
            --dataset "$DATASET" --model "$MODEL" \
            --epochs 10 --batch-size 16 --num-workers 4 \
            --wandb-tags accv2026 dataset_expansion "$DATASET"
    done
done

echo "[$(date)] ══════════════════════════════════════════════════"
echo "[$(date)] TRANSFORMERS TRAINING COMPLETE"
echo "[$(date)] ══════════════════════════════════════════════════"
echo ""
echo "Status:"
echo "  ✅ Completed: VideoMamba, 3D CNNs, TimeSformer FineGym"
echo "  ✅ Just done:  3x Transformers × 2 datasets"
echo "  ⚠️  Skipped:   SlowFast (memory intensive)"
echo ""
