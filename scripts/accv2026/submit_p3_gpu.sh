#!/usr/bin/env bash
# P3 daemon for gpu (A100) — CNNs only, max 4 concurrent
# Runs until ALL checkpoints exist. Never exits early.
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX=4
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)
MODELS=(r3d_18 mc3_18 r2plus1d_18)
RESOLUTIONS=(96 160 224 336)
TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#RESOLUTIONS[@]} ))   # 84

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU | $*"; }
n_gpu(){ squeue -u wesleyferreiramaia -p gpu --noheader 2>/dev/null | wc -l; }

done_count() {
    local n=0
    for m in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            for r in "${RESOLUTIONS[@]}"; do
                [[ -f "fine_tuned_models/accv2026_${m}_${ds}_${r}px_e10_h200/config.json" ]] && ((n++)) || true
            done
        done
    done
    echo $n
}

try_submit() {
    local model=$1 dataset=$2 res=$3
    local ckpt="fine_tuned_models/accv2026_${model}_${dataset}_${res}px_e10_h200"
    [[ -f "${ckpt}/config.json" ]] && return 0          # already done
    [[ $(n_gpu) -ge $MAX ]]        && return 1          # no slot
    local result
    result=$(sbatch --partition=gpu --gres=gpu:1 \
        --export=MODEL=${model},DATASET=${dataset},INPUT_SIZE=${res} \
        scripts/accv2026/slurm_h200_resolution_retrain.sbatch 2>&1)
    log "[${model}/${dataset}@${res}px] $result"
    sleep 2
    return 0
}

log "=== P3/gpu DAEMON started — will run until ${TOTAL} checkpoints exist ==="

while true; do
    done=$(done_count)
    log "Progress: ${done}/${TOTAL} done | queue=$(n_gpu)/${MAX}"

    [[ $done -ge $TOTAL ]] && { log "=== ALL ${TOTAL} GPU CHECKPOINTS DONE — exiting ==="; exit 0; }

    # Try to fill all available slots
    submitted=0
    for model in "${MODELS[@]}"; do
        for res in "${RESOLUTIONS[@]}"; do
            for ds in "${DATASETS[@]}"; do
                ckpt="fine_tuned_models/accv2026_${model}_${ds}_${res}px_e10_h200"
                [[ -f "${ckpt}/config.json" ]] && continue
                if [[ $(n_gpu) -lt $MAX ]]; then
                    try_submit "$model" "$ds" "$res" && ((submitted++)) || true
                fi
            done
        done
    done

    if [[ $submitted -eq 0 ]]; then
        sleep 30  # shorter sleep if no progress
    else
        sleep 2   # quick retry if we just submitted
    fi
done
