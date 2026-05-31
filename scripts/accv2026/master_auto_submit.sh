#!/usr/bin/env bash
# Master auto-submitter — handles E1, E6, and resolution retraining in priority order.
# Respects QOS limit (max 6 concurrent to stay safe under 7 hard limit).
#
# Usage: nohup bash scripts/accv2026/master_auto_submit.sh \
#              > evaluations/accv2026/logs/master_auto_submit.log 2>&1 &

set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
MAX_JOBS=4
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

n_running() { squeue -u wesleyferreiramaia --noheader 2>/dev/null | wc -l; }

wait_for_slot() {
    local label=$1
    while true; do
        local n; n=$(n_running)
        [[ $n -lt $MAX_JOBS ]] && return
        log "  $n/$MAX_JOBS jobs — waiting ($label)..."
        sleep 30
    done
}

submit() {
    local label=$1 partition=$2 gres=$3 export_str=$4 script=$5
    local result
    # gres="none" means no --gres flag (A100 partition picks GPU automatically)
    if [[ "$gres" == "none" || -z "$gres" ]]; then
        result=$(sbatch --partition="$partition" \
            --export="$export_str" \
            "$script" 2>&1)
    else
        result=$(sbatch --partition="$partition" \
            --gres="$gres" \
            --export="$export_str" \
            "$script" 2>&1)
    fi
    log "  [$label] $result"
    sleep 2
}

# ── PRIORITY 1: E1 remaining (coverage×stride sweep) ─────────────────────
log "=== PRIORITY 1: E1 Coverage×Stride — remaining ==="
for pair in \
    "r2plus1d_18 ucf101 gpu none" \
    "slowfast_r50 ucf101 gpu none" \
    "timesformer ucf101 gpu none" \
    "vivit ucf101 gpu none" \
    "videomae ucf101 gpu none" \
    "videomae diving48 gpu none" \
    "videomae autsl gpu none" \
    "videomae driveact gpu none" \
    "videomae epic_kitchens gpu none" \
    "videomamba ucf101 cenvalarc.gpu gpu:nvidia_h200_nvl:1" \
    "videomamba ssv2 cenvalarc.gpu gpu:nvidia_h200_nvl:1" \
    "videomamba hmdb51 cenvalarc.gpu gpu:nvidia_h200_nvl:1" \
    "videomamba diving48 cenvalarc.gpu gpu:nvidia_h200_nvl:1" \
    "videomamba autsl cenvalarc.gpu gpu:nvidia_h200_nvl:1" \
    "videomamba driveact cenvalarc.gpu gpu:nvidia_h200_nvl:1" \
    "videomamba epic_kitchens cenvalarc.gpu gpu:nvidia_h200_nvl:1"; do

    read -r model dataset partition gres <<< "$pair"
    dir="evaluations/accv2026/coverage_stride_sweep/${model}_${dataset}"
    n=$(find "$dir" -name "*_summary.csv" 2>/dev/null | wc -l)
    [[ $n -ge 25 ]] && { log "  SKIP E1 $model/$dataset ($n/25)"; continue; }

    wait_for_slot "E1 $model/$dataset"
    submit "E1 $model/$dataset" "$partition" "$gres" \
        "MODEL=${model},DATASET=${dataset}" \
        scripts/accv2026/slurm_cov_stride_sweep.sbatch
done

# ── PRIORITY 2: E6 remaining (spatial resolution eval on existing ckpts) ──
log ""
log "=== PRIORITY 2: E6 Spatial Resolution Sweep — remaining ==="
for pair in \
    "r2plus1d_18 ssv2 gpu none" \
    "slowfast_r50 ssv2 gpu none" \
    "vivit ssv2 cenvalarc.gpu gpu:nvidia_h200_nvl:1" \
    "videomae ssv2 cenvalarc.gpu gpu:nvidia_h200_nvl:1" \
    "videomamba ssv2 cenvalarc.gpu gpu:nvidia_h200_nvl:1"; do

    read -r model dataset partition gres <<< "$pair"
    dir="evaluations/accv2026/spatial_resolution_sweep/${model}_${dataset}"
    done_flag="${dir}/spatial_sweep_summary.csv"
    [[ -f "$done_flag" ]] && { log "  SKIP E6 $model/$dataset"; continue; }

    wait_for_slot "E6 $model/$dataset"
    submit "E6 $model/$dataset" "$partition" "$gres" \
        "MODEL=${model},DATASET=${dataset}" \
        scripts/accv2026/slurm_spatial_resolution_sweep.sbatch
done

# ── PRIORITY 3: Resolution retraining — 5-point grid 96/112/160/224/336px ──
# All 5 resolutions valid for ALL architectures (divisible by patch_size=16)
# Mirrors temporal E1 which has 5 strides → spatial aliasing curve needs 5 points too
# Each model trains at the 4 resolutions that are NOT its native:
#   CNNs  (native=112) → new: 96, 160, 224, 336
#   Transformers/SlowFast/VideoMamba (native=224) → new: 96, 112, 160, 336
log ""
log "=== PRIORITY 3: Resolution Retrain — 5-point grid (96/112/160/224/336px) ==="
log "    8 models × 4 new resolutions × 7 datasets = 224 new jobs"
#  model          target_px  partition             gres
declare -a RETRAIN_JOBS=(
    # CNNs (native=112) → train at 96, 160, 224, 336
    "r3d_18        96  gpu            none"
    "r3d_18        160 gpu            none"
    "r3d_18        224 gpu            none"
    "r3d_18        336 gpu            none"
    "mc3_18        96  gpu            none"
    "mc3_18        160 gpu            none"
    "mc3_18        224 gpu            none"
    "mc3_18        336 gpu            none"
    "r2plus1d_18   96  gpu            none"
    "r2plus1d_18   160 gpu            none"
    "r2plus1d_18   224 gpu            none"
    "r2plus1d_18   336 gpu            none"
    # SlowFast (native=224) → train at 96, 112, 160, 336
    "slowfast_r50  96  cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "slowfast_r50  112 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "slowfast_r50  160 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "slowfast_r50  336 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    # Transformers (native=224) → train at 96, 112, 160, 336
    "timesformer   96  cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "timesformer   112 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "timesformer   160 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "timesformer   336 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "vivit         96  cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "vivit         112 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "vivit         160 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "vivit         336 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "videomae      96  cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "videomae      112 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "videomae      160 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "videomae      336 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    # VideoMamba (native=224) → train at 96, 112, 160, 336
    "videomamba    96  cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "videomamba    112 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "videomamba    160 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
    "videomamba    336 cenvalarc.gpu  gpu:nvidia_h200_nvl:1"
)

for job_def in "${RETRAIN_JOBS[@]}"; do
    read -r model target_res partition gres <<< "$job_def"

    for dataset in "${DATASETS[@]}"; do
        ckpt="fine_tuned_models/accv2026_${model}_${dataset}_${target_res}px_e10_h200"
        [[ -f "${ckpt}/config.json" ]] && { log "  SKIP retrain $model/$dataset@${target_res}px"; continue; }

        wait_for_slot "retrain $model/$dataset@${target_res}px"
        submit "retrain $model/$dataset@${target_res}px" "$partition" "$gres" \
            "MODEL=${model},DATASET=${dataset},INPUT_SIZE=${target_res}" \
            scripts/accv2026/slurm_h200_resolution_retrain.sbatch
    done
done

log ""
log "=== Master auto-submitter COMPLETE ==="
log "W&B: https://wandb.ai/mi3lab/inforates-accv2026"
