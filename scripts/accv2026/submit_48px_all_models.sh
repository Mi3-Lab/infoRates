#!/usr/bin/env bash
# Submit 48px training for ALL models × ALL datasets.
# Resolution 48px closes the Nyquist curve: 48→96→112→160→224px.
# Uses BOTH cenvalarc.gpu (L40S/H200) and gpu (A100) for max parallelism.
# Checkpoint path: accv2026_{MODEL}_{DATASET}_48px_e10_h200
set -uo pipefail

SBATCH=/data/wesleyferreiramaia/infoRates/scripts/accv2026/slurm_h200_resolution_retrain.sbatch
LOG_DIR=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs
SCRATCH=/scratch/wesleyferreiramaia/infoRates/fine_tuned_models
USER=wesleyferreiramaia
RES=48

# QOS limits per partition
MAX_CENV=4   # cenvalarc.gpu
MAX_GPU=4    # gpu (A100)

MODELS="r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba"
DATASETS="autsl diving48 driveact epic_kitchens hmdb51 ssv2 ucf101"

cenv_count() { squeue -u "$USER" -h -p cenvalarc.gpu 2>/dev/null | wc -l; }
gpu_count()  { squeue -u "$USER" -h -p gpu          2>/dev/null | wc -l; }

submit_job() {
    local model="$1" ds="$2"
    local attempts=0

    while true; do
        # Pick the partition with an open slot (prefer cenvalarc.gpu for H200/L40S)
        PARTITION=""
        EXTRA_FLAGS=""
        if [[ $(cenv_count) -lt $MAX_CENV ]]; then
            PARTITION="cenvalarc.gpu"
            # No --constraint=H200 at 48px — L40S/H200 both fine
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
        ((attempts++))
        if [[ $attempts -ge 10 ]]; then
            echo "[$(date +%T)] FAILED after 10 attempts: ${model}/${ds} — $OUT"
            return 1
        fi
        echo "[$(date +%T)] QOS retry ${attempts}/10 for ${model}/${ds} — sleeping 60s"
        sleep 60
    done
}

echo "[$(date +%T)] 48px training — all models × all datasets"
echo "  Partitions: cenvalarc.gpu (MAX=$MAX_CENV) + gpu/A100 (MAX=$MAX_GPU)"
echo "  Max concurrent: $((MAX_CENV + MAX_GPU)) jobs"
echo "  Checkpoint path: accv2026_{MODEL}_{DS}_48px_e10_h200"

total=0; skipped=0; submitted=0
for MODEL in $MODELS; do
    for DS in $DATASETS; do
        ((total++))
        CKPT="${SCRATCH}/accv2026_${MODEL}_${DS}_${RES}px_e10_h200"
        SKIP_MARKER="${CKPT}/config.json"
        [[ "$MODEL" == "videomamba" ]] && SKIP_MARKER="${CKPT}/accv_meta.json"
        if [[ -f "${SKIP_MARKER}" ]]; then
            echo "[SKIP] ${MODEL}/${DS}@${RES}px already done"
            ((skipped++))
            continue
        fi
        submit_job "$MODEL" "$DS" || true
        ((submitted++))
        sleep 1
    done
done

echo "[$(date +%T)] Done. Total: $total | Skipped: $skipped | Submitted: $submitted"
