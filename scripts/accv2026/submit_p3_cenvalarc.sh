#!/usr/bin/env bash
# P3 submitter for cenvalarc.gpu — H200 NVL for Transformers/SSMs, L40s for SlowFast
# Transformers go on H200 (nvidia_h200_nvl) for ~3x speedup over L40s.
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
# submit_h200: request H200 specifically (Transformers / VideoMamba)
submit_h200() {
    local label=$1 model=$2 dataset=$3 res=$4
    local ckpt="fine_tuned_models/accv2026_${model}_${dataset}_${res}px_e10_h200"
    [[ -f "${ckpt}/config.json" ]] && { log "SKIP $label"; return; }
    wait_slot "$label"
    local result
    result=$(sbatch --partition=cenvalarc.gpu --gres=gpu:nvidia_h200_nvl:1 \
        --export=MODEL=${model},DATASET=${dataset},INPUT_SIZE=${res} \
        scripts/accv2026/slurm_h200_resolution_retrain.sbatch 2>&1)
    log "[H200|$label] $result"
    sleep 2
}
# submit_any: any GPU (L40s or H200) — for SlowFast (fast enough on any)
submit_any() {
    local label=$1 model=$2 dataset=$3 res=$4
    local ckpt="fine_tuned_models/accv2026_${model}_${dataset}_${res}px_e10_h200"
    [[ -f "${ckpt}/config.json" ]] && { log "SKIP $label"; return; }
    wait_slot "$label"
    local result
    result=$(sbatch --partition=cenvalarc.gpu --gres=gpu:1 \
        --export=MODEL=${model},DATASET=${dataset},INPUT_SIZE=${res} \
        scripts/accv2026/slurm_h200_resolution_retrain.sbatch 2>&1)
    log "[ANY|$label] $result"
    sleep 2
}

log "=== P3/cenvalarc: SlowFast→any GPU | Transformers+VMamba→H200 NVL ==="

# Transformers FIRST — qualquer GPU disponível (L40s agora, H200 quando liberar)
# Heaviest jobs first so training starts immediately on available L40s
for model in timesformer vivit videomae; do
    for res in 96 112 160 336; do
        for ds in "${DATASETS[@]}"; do
            submit_any "$model/$ds@${res}px" "$model" "$ds" "$res"
        done
    done
done

# VideoMamba
for res in 96 112 160 336; do
    for ds in "${DATASETS[@]}"; do
        submit_any "videomamba/$ds@${res}px" "videomamba" "$ds" "$res"
    done
done

# SlowFast LAST (fast enough on L40s, ~25min/job)
for res in 96 112 160 336; do
    for ds in "${DATASETS[@]}"; do
        submit_any "slowfast_r50/$ds@${res}px" "slowfast_r50" "$ds" "$res"
    done
done

log "=== P3/cenvalarc DONE ==="
