#!/usr/bin/env bash
# P3 daemon for cenvalarc.gpu — Transformers + SSMs + SlowFast, max 4 concurrent
# Runs until ALL checkpoints exist. Never exits early.
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX=4
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)
MODELS=(timesformer vivit videomae videomamba slowfast_r50)
RESOLUTIONS=(96 112 160 336)
TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#RESOLUTIONS[@]} ))   # 140

log()    { echo "[$(date '+%Y-%m-%d %H:%M:%S')] CENVALRC | $*"; }
n_cenva(){ squeue -u wesleyferreiramaia -p cenvalarc.gpu --noheader 2>/dev/null | wc -l; }

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
    [[ -f "${ckpt}/config.json" ]] && return 0
    [[ $(n_cenva) -ge $MAX ]]      && return 1
    local result
    result=$(sbatch --partition=cenvalarc.gpu --gres=gpu:1 \
        --export=MODEL=${model},DATASET=${dataset},INPUT_SIZE=${res} \
        scripts/accv2026/slurm_h200_resolution_retrain.sbatch 2>&1)
    log "[${model}/${dataset}@${res}px] $result"
    sleep 2
    return 0
}

log "=== P3/cenvalarc DAEMON started — will run until ${TOTAL} checkpoints exist ==="

while true; do
    done=$(done_count)
    log "Progress: ${done}/${TOTAL} done | queue=$(n_cenva)/${MAX}"

    [[ $done -ge $TOTAL ]] && { log "=== ALL ${TOTAL} CENVALRC CHECKPOINTS DONE — exiting ==="; exit 0; }

    # Priority order: timesformer first (most done), then vivit, videomae, videomamba, slowfast last
    submitted=0
    for model in timesformer vivit videomae videomamba slowfast_r50; do
        for res in 96 112 160 336; do
            for ds in "${DATASETS[@]}"; do
                ckpt="fine_tuned_models/accv2026_${model}_${ds}_${res}px_e10_h200"
                [[ -f "${ckpt}/config.json" ]] && continue
                if [[ $(n_cenva) -lt $MAX ]]; then
                    try_submit "$model" "$ds" "$res" && ((submitted++)) || true
                fi
            done
        done
    done

    [[ $submitted -eq 0 ]] && sleep 60 || sleep 5
done
