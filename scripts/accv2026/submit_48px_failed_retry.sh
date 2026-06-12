#!/usr/bin/env bash
# Re-submit 48px jobs that failed due to disk quota (WANDB_DIR now fixed to /data)
# Targets: timesformer×7, vivit×7, videomae×7, videomamba×7
#          slowfast×5 incomplete (diving48/hmdb51/epic_kitchens/ssv2/ucf101)
set -uo pipefail

SBATCH=/data/wesleyferreiramaia/infoRates/scripts/accv2026/slurm_h200_resolution_retrain.sbatch
LOG_DIR=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs
SCRATCH=/scratch/wesleyferreiramaia/infoRates/fine_tuned_models
USER=wesleyferreiramaia
RES=48

MAX_CENV=4
MAX_GPU=4

DATASETS="autsl diving48 driveact epic_kitchens hmdb51 ssv2 ucf101"

# Models with ALL 7 datasets failed
TRANSFORMER_MODELS="timesformer vivit videomae videomamba"
# SlowFast: only incomplete ones (autsl/driveact/48px already done)
SLOWFAST_PENDING="diving48 hmdb51 epic_kitchens ssv2 ucf101"

cenv_count() { squeue -u "$USER" -h -p cenvalarc.gpu 2>/dev/null | wc -l; }
gpu_count()  { squeue -u "$USER" -h -p gpu          2>/dev/null | wc -l; }

submit_job() {
    local model="$1" ds="$2"
    while true; do
        PARTITION=""
        if [[ $(cenv_count) -lt $MAX_CENV ]]; then
            PARTITION="cenvalarc.gpu"
        elif [[ $(gpu_count) -lt $MAX_GPU ]]; then
            PARTITION="gpu"
        else
            sleep 30
            continue
        fi

        OUT=$(sbatch --partition="${PARTITION}" \
            --export=ALL,MODEL="${model}",DATASET="${ds}",INPUT_SIZE="${RES}",VERSION_SUFFIX= \
            --output="${LOG_DIR}/retrain-48px-${model}-${ds}-%j.out" \
            --error="${LOG_DIR}/retrain-48px-${model}-${ds}-%j.err" \
            "${SBATCH}" 2>&1)
        if echo "$OUT" | grep -q "Submitted batch job"; then
            echo "[$(date +%T)] $OUT — ${model}/${ds}@${RES}px [${PARTITION}]"
            return 0
        fi
        sleep 30
    done
}

echo "[$(date +%T)] Re-submitting 48px failed jobs (WANDB_DIR fix applied)"

# Transformer models — all 7 datasets
for MODEL in $TRANSFORMER_MODELS; do
    for DS in $DATASETS; do
        CKPT="${SCRATCH}/accv2026_${MODEL}_${DS}_${RES}px_e10_h200"
        SKIP_MARKER="${CKPT}/config.json"
        [[ "$MODEL" == "videomamba" ]] && SKIP_MARKER="${CKPT}/accv_meta.json"
        if [[ -f "${SKIP_MARKER}" ]]; then
            echo "[SKIP] ${MODEL}/${DS}@${RES}px already done"
            continue
        fi
        submit_job "$MODEL" "$DS" || true
        sleep 1
    done
done

# SlowFast — only incomplete datasets
for DS in $SLOWFAST_PENDING; do
    CKPT="${SCRATCH}/accv2026_slowfast_r50_${DS}_${RES}px_e10_h200"
    if [[ -f "${CKPT}/config.json" ]]; then
        echo "[SKIP] slowfast_r50/${DS}@${RES}px already done"
        continue
    fi
    submit_job "slowfast_r50" "$DS" || true
    sleep 1
done

echo "[$(date +%T)] All retry jobs submitted."
