#!/usr/bin/env bash
# P3 336px daemon — submits to BOTH L40s (cenvalarc.gpu) and A100 (gpu) in parallel
# Uses a lock file per model/dataset to prevent duplicates across partitions
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX_CENV=4   # max concurrent on cenvalarc.gpu
MAX_GPU=4    # max concurrent on gpu (A100)
MAX_TOTAL=8  # combined max

DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)
MODELS=(r3d_18 mc3_18 slowfast_r50)
RES=336
TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))  # 21

CKPT_BASE="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"
LOCK_DIR="/tmp/p3_336px_locks"
mkdir -p "$LOCK_DIR"

log()      { echo "[$(date '+%Y-%m-%d %H:%M:%S')] 336px | $*"; }
n_cenv()   { squeue -u wesleyferreiramaia -p cenvalarc.gpu --noheader 2>/dev/null | wc -l; }
n_gpu()    { squeue -u wesleyferreiramaia -p gpu --noheader 2>/dev/null | wc -l; }
n_total()  { squeue -u wesleyferreiramaia --noheader 2>/dev/null | grep "retrain-336px" | wc -l; }

done_count() {
    local n=0
    for m in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            [[ -f "${CKPT_BASE}/accv2026_${m}_${ds}_${RES}px_e10_h200/config.json" ]] && ((n++)) || true
        done
    done
    echo $n
}

try_submit() {
    local model=$1 dataset=$2
    local ckpt="${CKPT_BASE}/accv2026_${model}_${dataset}_${RES}px_e10_h200"
    local lock="${LOCK_DIR}/${model}_${dataset}.lock"

    [[ -f "${ckpt}/config.json" ]] && return 0   # already done
    [[ -f "$lock" ]] && return 0                  # already submitted by this daemon

    # Choose partition with most free slots
    local nc=$(n_cenv) ng=$(n_gpu)
    local partition=""
    if [[ $nc -lt $MAX_CENV && $nc -le $ng ]]; then
        partition="cenvalarc.gpu"
    elif [[ $ng -lt $MAX_GPU ]]; then
        partition="gpu"
    else
        return 1  # both full
    fi

    touch "$lock"
    local result
    result=$(sbatch --partition=$partition --gres=gpu:2 \
        --export=MODEL=${model},DATASET=${dataset} \
        scripts/accv2026/slurm_336px_2gpu.sbatch 2>&1)
    log "[${model}/${dataset}@${RES}px → $partition] $result"
    sleep 2
    return 0
}

log "=== 336px COMBINED DAEMON started — target: ${TOTAL} checkpoints ==="
log "Submits to both cenvalarc.gpu (L40s) and gpu (A100), max ${MAX_TOTAL} total"

while true; do
    done=$(done_count)
    total_q=$(n_total)
    log "Progress: ${done}/${TOTAL} done | running=$total_q"

    [[ $done -ge $TOTAL ]] && { log "=== ALL ${TOTAL} 336px DONE ==="; rm -f "${LOCK_DIR}"/*.lock; exit 0; }

    # Clean stale locks for completed checkpoints
    for lockfile in "${LOCK_DIR}"/*.lock; do
        [[ -f "$lockfile" ]] || continue
        base=$(basename "$lockfile" .lock)
        model="${base%%_*}"
        dataset="${base#*_}"
        ckpt="${CKPT_BASE}/accv2026_${model}_${dataset}_${RES}px_e10_h200"
        [[ -f "${ckpt}/config.json" ]] && rm -f "$lockfile"
    done

    submitted=0
    for model in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            if [[ $(n_total) -lt $MAX_TOTAL ]]; then
                try_submit "$model" "$ds" && ((submitted++)) || true
            fi
        done
    done

    if [[ $submitted -eq 0 ]]; then
        sleep 60
    else
        sleep 5
    fi
done
