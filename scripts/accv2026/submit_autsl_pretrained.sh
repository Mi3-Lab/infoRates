#!/usr/bin/env bash
# Submit VideoMamba AUTSL v2 @ all resolutions using --pretrained-from 224px checkpoint.
# Resolutions 96/112/160px init from AUTSL 224px v1 (BiMamba already adapted to sign language).
# Resolution 224px trains from K400 pretrained (constant lr=2e-5, same as original 20.17% run).
set -uo pipefail

SBATCH=/data/wesleyferreiramaia/infoRates/scripts/accv2026/slurm_h200_resolution_retrain.sbatch
LOG_DIR=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs
SCRATCH=/scratch/wesleyferreiramaia/infoRates/fine_tuned_models
MAX_CENV=4
USER=wesleyferreiramaia

cenv_count() { squeue -u "$USER" -h -p cenvalarc.gpu 2>/dev/null | wc -l; }

submit_job() {
    local res="$1"
    local attempts=0
    while true; do
        while [[ $(cenv_count) -ge $MAX_CENV ]]; do sleep 30; done
        OUT=$(sbatch --partition=cenvalarc.gpu \
            --constraint=H200 \
            --export=ALL,MODEL=videomamba,DATASET=autsl,INPUT_SIZE="${res}",VERSION_SUFFIX=_v2 \
            --output="${LOG_DIR}/vmamba-v2-autsl-${res}px-%j.out" \
            --error="${LOG_DIR}/vmamba-v2-autsl-${res}px-%j.err" \
            "${SBATCH}" 2>&1)
        if echo "$OUT" | grep -q "Submitted batch job"; then
            echo "[$(date +%T)] $OUT — videomamba/autsl@${res}px v2"
            return 0
        fi
        ((attempts++))
        if [[ $attempts -ge 10 ]]; then
            echo "[$(date +%T)] FAILED after 10 attempts: autsl@${res}px — $OUT"
            return 1
        fi
        echo "[$(date +%T)] QOS retry ${attempts}/10 for autsl@${res}px — sleeping 60s"
        sleep 60
    done
}

echo "[$(date +%T)] VideoMamba AUTSL v2 — pretrained-from 224px for sub-224 resolutions"

for RES in 96 112 160 224; do
    CKPT="${SCRATCH}/accv2026_videomamba_autsl_${RES}px_e10_v2_h200"
    if [[ -f "${CKPT}/accv_meta.json" ]]; then
        echo "[SKIP] autsl@${RES}px v2 already done"
        continue
    fi
    submit_job "$RES"
    sleep 2
done

echo "[$(date +%T)] All AUTSL v2 jobs submitted"
