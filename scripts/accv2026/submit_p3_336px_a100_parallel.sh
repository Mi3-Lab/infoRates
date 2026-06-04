#!/bin/bash
# P3 daemon for 336px CLEANUP — A100 PARALLEL execution
# Submits to GPU partition (A100, 40GB) with batch=48
# Runs in parallel with H200 daemon for faster completion
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX=2  # Conservative: 2 concurrent jobs on A100
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)
MODELS=(r3d_18 mc3_18 slowfast_r50)
RES=336
TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))  # 21 jobs

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] 336px-A100 | $*"; }
n_a100(){ squeue -u wesleyferreiramaia -p gpu --noheader 2>/dev/null | wc -l; }

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
    [[ $(n_a100) -ge $MAX ]]        && return 1          # no slot
    local result
    result=$(sbatch --partition=gpu --gres=gpu:1 \
        --export=MODEL=${model},DATASET=${dataset},INPUT_SIZE=${RES} \
        scripts/accv2026/slurm_h200_resolution_retrain.sbatch 2>&1)
    log "[${model}/${dataset}@${RES}px] $result"
    sleep 2
    return 0
}

log "=== P3/336px CLEANUP DAEMON (A100) started ==="
log "Will submit to GPU partition with batch=48 (A100 40GB limits)"
log "Running in PARALLEL with H200 daemon for faster completion"

while true; do
    done=$(done_count)
    log "Progress: ${done}/${TOTAL} done | A100 queue=$(n_a100)/${MAX}"

    [[ $done -ge $TOTAL ]] && { log "=== ALL ${TOTAL} 336px CHECKPOINTS DONE ==="; exit 0; }

    # Try to fill available A100 slots
    submitted=0
    for model in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            ckpt="fine_tuned_models/accv2026_${model}_${ds}_${RES}px_e10_h200"
            [[ -f "${ckpt}/config.json" ]] && continue
            if [[ $(n_a100) -lt $MAX ]]; then
                try_submit "$model" "$ds" && ((submitted++)) || true
            fi
        done
    done

    if [[ $submitted -eq 0 ]]; then
        sleep 60  # wait for slot to free
    else
        sleep 5   # quick retry after submission
    fi
done
