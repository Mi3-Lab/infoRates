#!/usr/bin/env bash
# Feeder: runs in background, watches for free GPU/H200 slots,
# submits missing jobs automatically until all are done.
# Usage: nohup bash scripts/accv2026/feeder_submit_jobs.sh &
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

POLL_SECONDS=120   # check every 2 minutes
LOG="evaluations/accv2026/logs/feeder_$(date +%Y%m%d_%H%M%S).log"
mkdir -p evaluations/accv2026/logs

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "Feeder started — polling every ${POLL_SECONDS}s"
log "Log: $LOG"

all_done() {
    local datasets=(hmdb51 diving48 epic_kitchens autsl driveact)
    for ds in "${datasets[@]}"; do
        for model in r3d_18 mc3_18 slowfast_r50; do
            [[ -f "evaluations/accv2026/fixed_budget/${model}_${ds}_full_e10_a100/temporal_metrics.csv" ]] || return 1
        done
        for model in timesformer vivit; do
            [[ -f "evaluations/accv2026/fixed_budget/${model}_${ds}_full_e10_h200/temporal_metrics.csv" ]] || return 1
        done
    done
    return 0
}

while true; do
    if all_done; then
        log "ALL JOBS COMPLETE — feeder exiting."
        break
    fi

    gpu_count=$(squeue -u wesleyferreiramaia -p gpu --format="%i" --noheader 2>/dev/null | wc -l || echo 4)
    h200_count=$(squeue -u wesleyferreiramaia -p cenvalarc.gpu --format="%i" --noheader 2>/dev/null | wc -l || echo 4)

    if [[ $gpu_count -lt 4 || $h200_count -lt 4 ]]; then
        log "Slots free — GPU=${gpu_count}/4  H200=${h200_count}/4 — submitting..."
        bash scripts/accv2026/submit_missing_jobs.sh >> "$LOG" 2>&1
    else
        log "All slots full (GPU=${gpu_count}/4  H200=${h200_count}/4) — waiting..."
    fi

    sleep $POLL_SECONDS
done
