#!/usr/bin/env bash
# Submit all missing trainres sweep dirs for transformers + VideoMamba.
# Respects per-partition QOS limits (4 jobs each on gpu and cenvalarc.gpu).
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

SBATCH="scripts/accv2026/slurm_trainres_sweep.sbatch"
SWEEP_ROOT="evaluations/accv2026/coverage_stride_sweep"
LOG="evaluations/accv2026/logs/daemon_missing_sweeps.log"

# All missing cells: MODEL DATASET RES PARTITION
# VideoMamba MUST use cenvalarc.gpu (H200 only); others can use gpu
declare -a JOBS=(
    "timesformer epic_kitchens 224 gpu"
    "videomae    driveact      224 gpu"
    "videomae    epic_kitchens 224 gpu"
    "videomae    hmdb51        224 gpu"
    "videomae    ssv2           96 gpu"
    "videomae    ssv2          112 gpu"
    "videomae    ssv2          160 gpu"
    "videomamba  epic_kitchens  96 cenvalarc.gpu"
    "videomamba  epic_kitchens 112 cenvalarc.gpu"
    "videomamba  epic_kitchens 160 cenvalarc.gpu"
    "videomamba  epic_kitchens 224 cenvalarc.gpu"
)

is_done() {
    local model=$1 ds=$2 res=$3
    local dir="${SWEEP_ROOT}/${model}_${ds}_trainres${res}"
    [ -f "$dir/sweep_summary.csv" ] && [ "$(wc -l < "$dir/sweep_summary.csv")" -ge 26 ]
}

slots_free() {
    local part=$1
    local limit=4
    local running
    running=$(squeue -u wesleyferreiramaia -p "$part" --noheader 2>/dev/null | wc -l)
    echo $(( limit - running ))
}

submit_one() {
    local model=$1 ds=$2 res=$3 part=$4
    local extra=""
    [[ "$model" == "videomamba" ]] && extra="--constraint=H200"
    local job_id
    job_id=$(sbatch \
        --job-name="sw-$(echo $model | cut -c1-4)-${ds}-${res}" \
        --partition="$part" \
        $extra \
        --export=ALL,MODEL="$model",DATASET="$ds",TRAIN_RES="$res" \
        "$SBATCH" 2>&1 | awk '{print $NF}')
    if [[ "$job_id" =~ ^[0-9]+$ ]]; then
        echo "[$(date '+%H:%M:%S')] Submitted ${model}/${ds}@${res}px → ${part} job ${job_id}" | tee -a "$LOG"
        return 0
    else
        echo "[$(date '+%H:%M:%S')] QOS limit: ${model}/${ds}@${res}px" | tee -a "$LOG"
        return 1
    fi
}

echo "[$(date '+%H:%M:%S')] Daemon starting — ${#JOBS[@]} missing sweep cells" | tee -a "$LOG"

remaining=("${JOBS[@]}")
while [ ${#remaining[@]} -gt 0 ]; do
    new_remaining=()
    for entry in "${remaining[@]}"; do
        read -r model ds res part <<< "$entry"

        if is_done "$model" "$ds" "$res"; then
            echo "[$(date '+%H:%M:%S')] DONE: ${model}/${ds}@${res}px" | tee -a "$LOG"
            continue
        fi

        jname="sw-$(echo $model | cut -c1-4)-${ds}-${res}"
        if squeue -u wesleyferreiramaia --format="%j" --noheader 2>/dev/null | grep -qF "$jname"; then
            new_remaining+=("$entry")
            continue
        fi

        if [ "$(slots_free $part)" -gt 0 ]; then
            if submit_one "$model" "$ds" "$res" "$part"; then
                new_remaining+=("$entry")
            else
                new_remaining+=("$entry")
            fi
        else
            new_remaining+=("$entry")
        fi
    done

    remaining=("${new_remaining[@]}")
    [ ${#remaining[@]} -eq 0 ] && break
    sleep 120
done

echo "[$(date '+%H:%M:%S')] All missing sweeps submitted and complete." | tee -a "$LOG"
