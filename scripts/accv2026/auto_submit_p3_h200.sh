#!/usr/bin/env bash
# P3 Resolution Retraining — H200 only, runs in parallel with master (E1 on A100)
# Max 3 concurrent on cenvalarc.gpu (leaves 1 slot for E1/VideoMamba jobs)
#
# Usage: nohup bash scripts/accv2026/auto_submit_p3_h200.sh \
#              > evaluations/accv2026/logs/p3_h200_submit.log 2>&1 &

set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
MAX_H200=5   # max concurrent on cenvalarc.gpu
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

n_h200() {
    squeue -u wesleyferreiramaia -p cenvalarc.gpu --noheader 2>/dev/null | wc -l
}

wait_h200_slot() {
    local label=$1
    while true; do
        local n; n=$(n_h200)
        [[ $n -lt $MAX_H200 ]] && return
        log "  $n/$MAX_H200 H200 jobs — waiting ($label)..."
        sleep 30
    done
}

# model  target_res  (CNNs native=112 → new: 96,160,224,336 | Transformers native=224 → new: 96,112,160,336)
declare -a RETRAIN_JOBS=(
    "r3d_18        96"
    "r3d_18        160"
    "r3d_18        224"
    "r3d_18        336"
    "mc3_18        96"
    "mc3_18        160"
    "mc3_18        224"
    "mc3_18        336"
    "r2plus1d_18   96"
    "r2plus1d_18   160"
    "r2plus1d_18   224"
    "r2plus1d_18   336"
    "slowfast_r50  96"
    "slowfast_r50  112"
    "slowfast_r50  160"
    "slowfast_r50  336"
    "timesformer   96"
    "timesformer   112"
    "timesformer   160"
    "timesformer   336"
    "vivit         96"
    "vivit         112"
    "vivit         160"
    "vivit         336"
    "videomae      96"
    "videomae      112"
    "videomae      160"
    "videomae      336"
    "videomamba    96"
    "videomamba    112"
    "videomamba    160"
    "videomamba    336"
)

total_submitted=0
total_skipped=0

log "=== P3 H200 auto-submitter started ==="
log "${#RETRAIN_JOBS[@]} model×resolution × 7 datasets = $((${#RETRAIN_JOBS[@]} * 7)) jobs"
log "MAX_H200=$MAX_H200 concurrent | batch: CNN=256, Transformers=64, SlowFast=64"
log ""

for job_def in "${RETRAIN_JOBS[@]}"; do
    read -r model target_res <<< "$job_def"

    for dataset in "${DATASETS[@]}"; do
        ckpt="fine_tuned_models/accv2026_${model}_${dataset}_${target_res}px_e10_h200"

        if [[ -f "${ckpt}/config.json" ]]; then
            log "  SKIP $model/$dataset@${target_res}px (done)"
            total_skipped=$((total_skipped+1))
            continue
        fi

        wait_h200_slot "$model/$dataset@${target_res}px"

        result=$(sbatch \
            --export=MODEL=${model},DATASET=${dataset},INPUT_SIZE=${target_res} \
            scripts/accv2026/slurm_h200_resolution_retrain.sbatch 2>&1)
        log "  $model/$dataset@${target_res}px → $result"
        total_submitted=$((total_submitted+1))
        sleep 2
    done
done

log ""
log "=== P3 H200 submitter DONE: $total_submitted submitted, $total_skipped skipped ==="
log "W&B: https://wandb.ai/mi3lab/inforates-accv2026"
