#!/usr/bin/env bash
# Re-train VideoMamba at all resolutions on H200 (guaranteed).
# Uses constant lr=2e-5 (no warmup/cosine) — matches the setup that achieved 20.17% on AUTSL.
# Saves to accv2026_videomamba_{DS}_{RES}px_e10_v2_h200 (VERSION_SUFFIX=_v2)
set -uo pipefail

SBATCH=/data/wesleyferreiramaia/infoRates/scripts/accv2026/slurm_h200_resolution_retrain.sbatch
LOG_DIR=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs
SCRATCH=/scratch/wesleyferreiramaia/infoRates/fine_tuned_models
MAX_CENV=4     # cenvalarc.gpu partition QOS limit is 4 concurrent jobs
USER=wesleyferreiramaia

DATASETS="diving48 ssv2 hmdb51 driveact epic_kitchens ucf101"  # autsl handled by submit_autsl_pretrained.sh
RESOLUTIONS="96 112 160 224"

# Count jobs in cenvalarc.gpu partition only
cenv_count() { squeue -u "$USER" -h -p cenvalarc.gpu 2>/dev/null | wc -l; }

# Submit with retry on QOS error
submit_job() {
    local ds="$1" res="$2"
    local attempts=0
    while true; do
        # Wait for a free cenvalarc slot
        while [[ $(cenv_count) -ge $MAX_CENV ]]; do sleep 30; done
        OUT=$(sbatch --partition=cenvalarc.gpu \
            --constraint=H200 \
            --export=ALL,MODEL=videomamba,DATASET="${ds}",INPUT_SIZE="${res}",VERSION_SUFFIX=_v2 \
            --output="${LOG_DIR}/vmamba-v2-${ds}-${res}px-%j.out" \
            --error="${LOG_DIR}/vmamba-v2-${ds}-${res}px-%j.err" \
            "${SBATCH}" 2>&1)
        if echo "$OUT" | grep -q "Submitted batch job"; then
            echo "[$(date +%T)] $OUT — videomamba/${ds}@${res}px v2"
            return 0
        fi
        ((attempts++))
        if [[ $attempts -ge 10 ]]; then
            echo "[$(date +%T)] FAILED after 10 attempts: ${ds}@${res}px — $OUT"
            return 1
        fi
        echo "[$(date +%T)] QOS retry ${attempts}/10 for ${ds}@${res}px — sleeping 60s"
        sleep 60
    done
}

echo "[$(date +%T)] VideoMamba retrain v2 — constant lr=2e-5, H200 forced (MAX_CENV=$MAX_CENV)"

for RES in $RESOLUTIONS; do
    for DS in $DATASETS; do
        CKPT="${SCRATCH}/accv2026_videomamba_${DS}_${RES}px_e10_v2_h200"
        if [[ -f "${CKPT}/accv_meta.json" ]]; then
            echo "[SKIP] videomamba/${DS}@${RES}px v2 already done"
            continue
        fi
        submit_job "$DS" "$RES" || true
        sleep 2
    done
done
echo "[$(date +%T)] All VideoMamba v2 jobs submitted"
