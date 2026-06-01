#!/usr/bin/env bash
# P3 submitter for cenvalarc.gpu (L40s/H200) — Transformers + SlowFast + VideoMamba
# Runs in parallel with submit_p3_gpu.sh
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
MAX=4
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] CENVALRC | $*"; }
n() { squeue -u wesleyferreiramaia -p cenvalarc.gpu --noheader 2>/dev/null | wc -l; }
wait_slot() {
    while true; do
        [[ $(n) -lt $MAX ]] && return
        log "$(($(n)))/$MAX — waiting ($1)..."
        sleep 20
    done
}
submit() {
    local label=$1 model=$2 dataset=$3 res=$4
    local ckpt="fine_tuned_models/accv2026_${model}_${dataset}_${res}px_e10_h200"
    [[ -f "${ckpt}/config.json" ]] && { log "SKIP $label"; return; }
    wait_slot "$label"
    local result
    result=$(sbatch --partition=cenvalarc.gpu --gres=gpu:1 \
        --export=MODEL=${model},DATASET=${dataset},INPUT_SIZE=${res} \
        scripts/accv2026/slurm_h200_resolution_retrain.sbatch 2>&1)
    log "[$label] $result"
    sleep 2
}

log "=== P3/cenvalarc submitter: SlowFast+Transformers+VideoMamba → L40s/H200 ==="

# SlowFast: native=224 → train at 96, 112, 160, 336
for res in 96 112 160 336; do
    for ds in "${DATASETS[@]}"; do
        submit "slowfast_r50/$ds@${res}px" "slowfast_r50" "$ds" "$res"
    done
done

# Transformers: native=224 → train at 96, 112, 160, 336
for model in timesformer vivit videomae; do
    for res in 96 112 160 336; do
        for ds in "${DATASETS[@]}"; do
            submit "$model/$ds@${res}px" "$model" "$ds" "$res"
        done
    done
done

# VideoMamba: native=224 → train at 96, 112, 160, 336
for res in 96 112 160 336; do
    for ds in "${DATASETS[@]}"; do
        submit "videomamba/$ds@${res}px" "videomamba" "$ds" "$res"
    done
done

log "=== P3/cenvalarc DONE ==="
