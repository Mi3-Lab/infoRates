#!/usr/bin/env bash
# Auto-submitter: retrain all 8 models at their MISSING resolution on all 7 datasets
#
# Logic:
#   CNNs (native=112px)       → retrain at 224px
#   Transformers (native=224px) → retrain at 112px
#   VideoMamba (native=224px)  → retrain at 112px
#   SlowFast (native=224px)    → retrain at 112px
#
# Usage: nohup bash scripts/accv2026/auto_submit_resolution_retrain.sh \
#              > evaluations/accv2026/logs/auto_submit_resretrain.log 2>&1 &

set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
MAX_JOBS=8
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

wait_for_slot() {
    local label=$1
    while true; do
        local n; n=$(squeue -u wesleyferreiramaia --noheader 2>/dev/null | wc -l)
        [[ $n -lt $MAX_JOBS ]] && break
        log "  $n/$MAX_JOBS jobs — waiting for slot ($label)..."
        sleep 30
    done
}

# model  target_res  partition
declare -a JOBS=(
    "r3d_18        224 gpu"
    "mc3_18        224 gpu"
    "r2plus1d_18   224 gpu"
    "slowfast_r50  112 cenvalarc.gpu"
    "timesformer   112 cenvalarc.gpu"
    "vivit         112 cenvalarc.gpu"
    "videomae      112 cenvalarc.gpu"
    "videomamba    112 cenvalarc.gpu"
)

log "=== Resolution Retrain auto-submitter started ==="
log "8 models × 7 datasets = 56 jobs (max $MAX_JOBS concurrent)"
log ""

total=0
skipped=0

for job_def in "${JOBS[@]}"; do
    read -r model target_res partition <<< "$job_def"

    for dataset in "${DATASETS[@]}"; do
        ckpt="fine_tuned_models/accv2026_${model}_${dataset}_${target_res}px_e10_h200"

        if [[ -f "${ckpt}/config.json" ]]; then
            log "  SKIP $model/$dataset@${target_res}px (checkpoint exists)"
            skipped=$((skipped+1))
            continue
        fi

        wait_for_slot "$model/$dataset@${target_res}px"

        gres_arg=""
        [[ "$partition" == "cenvalarc.gpu" ]] && gres_arg="--gres=gpu:nvidia_h200_nvl:1"

        result=$(sbatch \
            --partition="$partition" \
            $gres_arg \
            --export=MODEL=${model},DATASET=${dataset},INPUT_SIZE=${target_res} \
            scripts/accv2026/slurm_h200_resolution_retrain.sbatch 2>&1)

        log "  $model/$dataset@${target_res}px → $result"
        total=$((total+1))
        sleep 2
    done
done

log ""
log "=== Done: $total submitted, $skipped skipped ==="
log "Monitor: squeue -u wesleyferreiramaia"
log "W&B: https://wandb.ai/mi3lab/inforates-accv2026"
