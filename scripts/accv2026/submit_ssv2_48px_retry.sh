#!/usr/bin/env bash
set -uo pipefail
SBATCH=/data/wesleyferreiramaia/infoRates/scripts/accv2026/slurm_h200_resolution_retrain.sbatch
LOG_DIR=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs
CKPT=/scratch/wesleyferreiramaia/infoRates/fine_tuned_models/accv2026_r3d_18_ssv2_48px_e10_h200

if [[ -f "${CKPT}/config.json" ]]; then
    echo "[$(date +%T)] r3d_18/ssv2@48px already done — skip"
    exit 0
fi

while true; do
    TOTAL=$(squeue -u wesleyferreiramaia -h 2>/dev/null | wc -l)
    GPU_COUNT=$(squeue -u wesleyferreiramaia -h -p gpu 2>/dev/null | wc -l)
    if [[ $TOTAL -lt 8 && $GPU_COUNT -lt 4 ]]; then
        OUT=$(sbatch --partition=gpu \
            --export=ALL,MODEL=r3d_18,DATASET=ssv2,INPUT_SIZE=48,VERSION_SUFFIX= \
            --output="$LOG_DIR/retrain-48px-r3d_18-ssv2-retry-%j.out" \
            --error="$LOG_DIR/retrain-48px-r3d_18-ssv2-retry-%j.err" \
            "$SBATCH" 2>&1)
        if echo "$OUT" | grep -q "Submitted batch job"; then
            echo "[$(date +%T)] $OUT — r3d_18/ssv2@48px retry [gpu]"
            exit 0
        fi
        echo "[$(date +%T)] Submit failed: $OUT — retrying in 30s"
    fi
    sleep 30
done
