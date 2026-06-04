#!/bin/bash
# P3 daemon for 336px CLEANUP (H200 only)
# Recalculates R3D, MC3, SlowFast @ 336px with batch=64
# Runs until all 336px checkpoints exist
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX=2  # Conservative: only 2 concurrent 336px jobs (memory-heavy)
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)
MODELS=(r3d_18 mc3_18 slowfast_r50)
RES=336
TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))  # 21 jobs

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] 336px-H200 | $*"; }
n_h200(){ squeue -u wesleyferreiramaia -p cenvalarc.gpu --noheader 2>/dev/null | wc -l; }

done_count() {
    local n=0
    for m in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            [[ -f "fine_tuned_models/accv2026_${m}_${ds}_${RES}px_e10_h200/config.json" ]] && ((n++)) || true
        done
    done
    echo $n
}

try_submit() {
    local model=$1 dataset=$2
    local ckpt="fine_tuned_models/accv2026_${model}_${dataset}_${RES}px_e10_h200"
    [[ -f "${ckpt}/config.json" ]] && return 0          # already done
    [[ $(n_h200) -ge $MAX ]]        && return 1          # no slot
    local result
    result=$(sbatch --partition=cenvalarc.gpu --gres=gpu:1 \
        --export=MODEL=${model},DATASET=${dataset},INPUT_SIZE=${RES} \
        scripts/accv2026/slurm_h200_resolution_retrain.sbatch 2>&1)
    log "[${model}/${dataset}@${RES}px] $result"
    sleep 2
    return 0
}

log "=== P3/336px CLEANUP DAEMON started — will run until ${TOTAL} checkpoints exist ==="
log "Target: R3D-18, MC3-18, SlowFast @ 336px with batch=64 on H200 only"

while true; do
    done=$(done_count)
    log "Progress: ${done}/${TOTAL} done | queue=$(n_h200)/${MAX}"

    [[ $done -ge $TOTAL ]] && { log "=== ALL ${TOTAL} 336px CHECKPOINTS DONE — exiting ==="; exit 0; }

    # Try to fill available slots
    submitted=0
    for model in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            ckpt="fine_tuned_models/accv2026_${model}_${ds}_${RES}px_e10_h200"
            [[ -f "${ckpt}/config.json" ]] && continue
            if [[ $(n_h200) -lt $MAX ]]; then
                try_submit "$model" "$ds" && ((submitted++)) || true
            fi
        done
    done

    if [[ $submitted -eq 0 ]]; then
        sleep 60  # longer sleep if no progress (wait for jobs to finish)
    else
        sleep 5   # quick retry if we just submitted
    fi
done
