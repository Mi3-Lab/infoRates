#!/usr/bin/env bash
# Re-run VideoMAE trainres48 sweeps (corrected inference after stale data deletion)
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

SBATCH=scripts/accv2026/slurm_res_cov_stride.sbatch
SWEEP_ROOT=evaluations/accv2026/coverage_stride_sweep
MAX_CENV=4
MAX_GPU=4

datasets=(autsl diving48 ssv2 hmdb51 driveact epic_kitchens ucf101)
pending=("${datasets[@]}")

is_done() {
    local ds=$1
    local dir="${SWEEP_ROOT}/videomae_${ds}_trainres48"
    [ -f "$dir/sweep_summary.csv" ] && [ "$(wc -l < "$dir/sweep_summary.csv")" -ge 26 ]
}

running_count() {
    local part=$1
    squeue -u wesleyferreiramaia -p "$part" --noheader 2>/dev/null | wc -l
}

submit_one() {
    local ds=$1
    local part=$2
    local job_id
    job_id=$(sbatch --job-name="res48-videomae-${ds}" \
        --partition="$part" \
        --export=ALL,MODEL=videomae,DATASET="$ds",TRAIN_RES=48 \
        "$SBATCH" 2>&1 | awk '{print $NF}')
    echo "[$(date '+%H:%M:%S')] Submitted videomae/$ds@48px → $part job $job_id"
}

echo "[$(date '+%H:%M:%S')] Daemon starting — VideoMAE trainres48 rerun (7 datasets)"

while [ ${#pending[@]} -gt 0 ]; do
    new_pending=()
    for ds in "${pending[@]}"; do
        if is_done "$ds"; then
            echo "[$(date '+%H:%M:%S')] DONE: videomae/$ds@48px"
            continue
        fi
        # Already queued/running?
        if squeue -u wesleyferreiramaia --noheader 2>/dev/null | grep -q "videomae.*${ds}\|${ds}.*videomae"; then
            new_pending+=("$ds")
            continue
        fi
        # Try to submit
        c_run=$(running_count cenvalarc.gpu)
        g_run=$(running_count gpu)
        if [ "$c_run" -lt "$MAX_CENV" ]; then
            submit_one "$ds" "cenvalarc.gpu"
            new_pending+=("$ds")
        elif [ "$g_run" -lt "$MAX_GPU" ]; then
            submit_one "$ds" "gpu"
            new_pending+=("$ds")
        else
            new_pending+=("$ds")
        fi
    done
    pending=("${new_pending[@]}")
    [ ${#pending[@]} -eq 0 ] && break
    sleep 120
done

echo "[$(date '+%H:%M:%S')] All VideoMAE trainres48 sweeps submitted and complete."
