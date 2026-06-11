#!/usr/bin/env bash
# Re-train VideoMamba at all resolutions with corrected hyperparameters:
#   lr=2e-5, warmup=2 epochs, cosine scheduler, batch=64/48/32 per resolution
# Saves to accv2026_videomamba_{DS}_{RES}px_e10_v2_h200 (VERSION_SUFFIX=_v2)
set -uo pipefail

SBATCH=/data/wesleyferreiramaia/infoRates/scripts/accv2026/slurm_h200_resolution_retrain.sbatch
LOG_DIR=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs
SCRATCH=/scratch/wesleyferreiramaia/infoRates/fine_tuned_models
MAX_JOBS=4
USER=wesleyferreiramaia

DATASETS="autsl diving48 ssv2 hmdb51 driveact epic_kitchens ucf101"
RESOLUTIONS="96 112 160 224"

running() { squeue -u "$USER" -h -r | wc -l; }

echo "[$(date +%T)] VideoMamba retrain v2 — warmup=2, cosine, correct batch sizes"

for RES in $RESOLUTIONS; do
    for DS in $DATASETS; do
        CKPT="${SCRATCH}/accv2026_videomamba_${DS}_${RES}px_e10_v2_h200"
        if [[ -f "${CKPT}/accv_meta.json" ]]; then
            echo "[SKIP] videomamba/${DS}@${RES}px v2 already done"
            continue
        fi
        while [[ $(running) -ge $MAX_JOBS ]]; do sleep 30; done
        JOB=$(sbatch --partition=cenvalarc.gpu \
            --export=ALL,MODEL=videomamba,DATASET="${DS}",INPUT_SIZE="${RES}",VERSION_SUFFIX=_v2 \
            --output="${LOG_DIR}/vmamba-v2-${DS}-${RES}px-%j.out" \
            --error="${LOG_DIR}/vmamba-v2-${DS}-${RES}px-%j.err" \
            "${SBATCH}" 2>&1)
        echo "[$(date +%T)] $JOB — videomamba/${DS}@${RES}px v2"
        sleep 2
    done
done
echo "[$(date +%T)] All VideoMamba v2 jobs submitted"
