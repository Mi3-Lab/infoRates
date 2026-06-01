#!/usr/bin/env bash
# P3 submitter for gpu (A100) partition — CNNs only, max 4 concurrent
# Runs in parallel with submit_p3_cenvalarc.sh
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
MAX=4
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU | $*"; }
n() { squeue -u wesleyferreiramaia -p gpu --noheader 2>/dev/null | wc -l; }
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
    result=$(sbatch --partition=gpu --gres=gpu:1 \
        --export=MODEL=${model},DATASET=${dataset},INPUT_SIZE=${res} \
        scripts/accv2026/slurm_h200_resolution_retrain.sbatch 2>&1)
    log "[$label] $result"
    sleep 2
}

log "=== P3/gpu submitter: CNNs (r3d, mc3, r2plus1d) → A100 ==="

# CNNs: native=112px → train at 96, 160, 224, 336
for model in r3d_18 mc3_18 r2plus1d_18; do
    for res in 96 160 224 336; do
        for ds in "${DATASETS[@]}"; do
            submit "$model/$ds@${res}px" "$model" "$ds" "$res"
        done
    done
done

log "=== P3/gpu DONE ==="
