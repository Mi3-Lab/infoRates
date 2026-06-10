#!/usr/bin/env bash
# Daemon: Retrain all 8 models on Ego4D at 4 resolutions (96/112/160/224px).
# Only starts submitting once the ego4d_manifest.csv exists (preprocessing done).
# Uses same lock dir as other daemons to avoid duplicates.
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX_H200=4
MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
DATASET=ego4d
RESOLUTIONS=(96 112 160 224)

CKPT_BASE="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"
MANIFEST="/scratch/wesleyferreiramaia/infoRates/data/Ego4D_data/splits/ego4d_manifest.csv"
LOCK_DIR="/data/wesleyferreiramaia/infoRates/evaluations/accv2026/locks"
mkdir -p "$LOCK_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] EGO4D-RETRAIN | $*"; }

n_h200() {
    squeue -u wesleyferreiramaia --noheader --format="%P" 2>/dev/null \
        | grep -c "^cenvalarc.gpu$" || true
}

TOTAL=$(( ${#MODELS[@]} * ${#RESOLUTIONS[@]} ))

is_done() {
    local model=$1 res=$2
    [[ -f "${CKPT_BASE}/accv2026_${model}_${DATASET}_${res}px_e10_h200/config.json" ]]
}

done_count() {
    local n=0
    for m in "${MODELS[@]}"; do for res in "${RESOLUTIONS[@]}"; do
        is_done "$m" "$res" && ((n++)) || true
    done; done
    echo $n
}

lock_name() { echo "${LOCK_DIR}/${1}@${DATASET}@${2}.lock"; }

try_submit() {
    local model=$1 res=$2
    local lock
    lock=$(lock_name "$model" "$res")

    is_done "$model" "$res" && return 0
    [[ -f "$lock" ]] && return 0

    [[ $(n_h200) -ge $MAX_H200 ]] && return 1

    touch "$lock"

    local result
    if result=$(sbatch \
        --partition=cenvalarc.gpu \
        --export=MODEL=${model},DATASET=${DATASET},INPUT_SIZE=${res} \
        scripts/accv2026/slurm_h200_retrain_all.sbatch 2>&1); then
        log "[${model}/${DATASET}@${res}px] $result"
    else
        log "[${model}/${DATASET}@${res}px] SBATCH FAILED — $result"
        rm -f "$lock"
        return 1
    fi
    sleep 2
    return 0
}

log "=== EGO4D RETRAIN DAEMON ==="
log "Target: ${TOTAL} jobs (8 models × 4 resolutions: 96+112+160+224px)"
log "Waiting for preprocessing manifest: ${MANIFEST}"

# Wait for download + preprocessing to complete
while [[ ! -f "${MANIFEST}" ]]; do
    log "Manifest not ready yet — sleeping 10 min..."
    sleep 600
done
log "Manifest found — starting training submissions"

while true; do
    # Clean locks for done jobs
    for lockfile in "${LOCK_DIR}"/*.lock; do
        [[ -f "$lockfile" ]] || continue
        base=$(basename "$lockfile" .lock)
        IFS='@' read -r m ds res <<< "$base"
        [[ "$ds" == "$DATASET" ]] || continue
        is_done "$m" "$res" && rm -f "$lockfile" || true
    done

    done=$(done_count)
    log "Progress: ${done}/${TOTAL} done | h200_queue=$(n_h200)"
    [[ $done -ge $TOTAL ]] && { log "=== ALL EGO4D DONE ==="; exit 0; }

    submitted=0
    for res in "${RESOLUTIONS[@]}"; do
        for model in "${MODELS[@]}"; do
            if [[ $(n_h200) -lt $MAX_H200 ]]; then
                try_submit "$model" "$res" && ((submitted++)) || true
            fi
        done
    done

    [[ $submitted -eq 0 ]] && sleep 60 || sleep 5
done
