#!/usr/bin/env bash
set -uo pipefail
SBATCH=/data/wesleyferreiramaia/infoRates/scripts/accv2026/slurm_h200_resolution_retrain.sbatch
LOG_DIR=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs
UCF_160_V3=/scratch/wesleyferreiramaia/infoRates/fine_tuned_models/accv2026_videomamba_ucf101_160px_e10_v3_h200
CKPT=/scratch/wesleyferreiramaia/infoRates/fine_tuned_models/accv2026_videomamba_ucf101_224px_e10_v3_h200

if [[ -f "${CKPT}/accv_meta.json" ]]; then
    echo "[$(date +%T)] Already done — skip"
    exit 0
fi

while true; do
    COUNT=$(squeue -u wesleyferreiramaia -h -p cenvalarc.gpu 2>/dev/null | wc -l)
    if [[ $COUNT -lt 4 ]]; then
        OUT=$(sbatch --partition=cenvalarc.gpu \
            --constraint=H200 \
            --export=ALL,MODEL=videomamba,DATASET=ucf101,INPUT_SIZE=224,VERSION_SUFFIX=_v3,VMAMBA_PRETRAINED_FROM_OVERRIDE="${UCF_160_V3}" \
            --output="${LOG_DIR}/vmamba-v3-ucf101-224px-%j.out" \
            --error="${LOG_DIR}/vmamba-v3-ucf101-224px-%j.err" \
            "${SBATCH}" 2>&1)
        if echo "$OUT" | grep -q "Submitted batch job"; then
            echo "[$(date +%T)] $OUT — ucf101@224px v3 [cenvalarc.gpu H200]"
            exit 0
        fi
        echo "[$(date +%T)] Submit failed: $OUT — retrying in 30s"
    fi
    sleep 30
done
