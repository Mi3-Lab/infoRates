#!/usr/bin/env bash
# Retrain videomamba/diving48@224px v3 using diving48_full as init
set -uo pipefail

SBATCH=/data/wesleyferreiramaia/infoRates/scripts/accv2026/slurm_h200_resolution_retrain.sbatch
LOG_DIR=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs
SCRATCH=/scratch/wesleyferreiramaia/infoRates

INIT="${SCRATCH}/fine_tuned_models/accv2026_videomamba_diving48_full_e10_h200"

if [[ ! -f "${INIT}/accv_meta.json" ]]; then
    echo "ERROR: init checkpoint not found: ${INIT}"
    exit 1
fi

OUT=$(sbatch --partition=cenvalarc.gpu \
    --export=ALL,MODEL=videomamba,DATASET=diving48,INPUT_SIZE=224,VERSION_SUFFIX=_v3,VMAMBA_PRETRAINED_FROM_OVERRIDE="${INIT}" \
    --output="${LOG_DIR}/vmamba-v3-diving48-224px-%j.out" \
    --error="${LOG_DIR}/vmamba-v3-diving48-224px-%j.err" \
    "${SBATCH}" 2>&1)

echo "$OUT"
if echo "$OUT" | grep -q "Submitted batch job"; then
    echo "[OK] videomamba/diving48@224px v3 submitted — init: diving48_full (43.6%)"
fi
