#!/usr/bin/env bash
# Single unified submitter — handles all remaining work in priority order
# Respects the real QOS limit of 4 total concurrent jobs
#
# Priority:
#   1. E1 VideoMamba remaining (ucf101, ssv2 partials)
#   2. E6 missing resolutions (r3d/mc3 96px+336px, timesformer 96px)
#   3. P3 Resolution retraining (224 jobs, all models × resolutions)
#
# Usage: nohup bash scripts/accv2026/auto_submit_final.sh \
#              > evaluations/accv2026/logs/final_submit.log 2>&1 &

set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
MAX_JOBS=9   # QOS limit: total RUNNING + PENDING across all partitions
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
n_jobs() { squeue -u wesleyferreiramaia --noheader 2>/dev/null | wc -l; }

wait_slot() {
    local label=$1
    while true; do
        local n; n=$(n_jobs)
        [[ $n -lt $MAX_JOBS ]] && return
        log "  $n/$MAX_JOBS — waiting ($label)..."
        sleep 30
    done
}

submit() {
    local label=$1 partition=$2 gres=$3 export_str=$4 script=$5
    local result
    if [[ "$gres" == "none" ]]; then
        result=$(sbatch --partition="$partition" --export="$export_str" "$script" 2>&1)
    else
        result=$(sbatch --partition="$partition" --gres="$gres" --export="$export_str" "$script" 2>&1)
    fi
    # Retry once if QOS limit hit (PENDING job just freed up a slot)
    if echo "$result" | grep -q "QOSMaxSubmit"; then
        log "  [$label] QOS hit — waiting 60s and retrying..."
        sleep 60
        if [[ "$gres" == "none" ]]; then
            result=$(sbatch --partition="$partition" --export="$export_str" "$script" 2>&1)
        else
            result=$(sbatch --partition="$partition" --gres="$gres" --export="$export_str" "$script" 2>&1)
        fi
    fi
    log "  [$label] $result"
    sleep 2
}

log "=== Unified final submitter started (MAX_JOBS=$MAX_JOBS total) ==="
log ""

# ── PRIORITY 1: E1 VideoMamba remaining ────────────────────────────────────
log "=== P1: E1 VideoMamba remaining ==="
for dataset in ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens; do
    n=$(find evaluations/accv2026/coverage_stride_sweep/videomamba_${dataset} -name "cov*_summary.csv" 2>/dev/null | wc -l)
    [[ $n -ge 25 ]] && { log "  SKIP E1 videomamba/$dataset ($n/25)"; continue; }
    wait_slot "E1 videomamba/$dataset"
    submit "E1 videomamba/$dataset" "cenvalarc.gpu" "gpu:1" \
        "MODEL=videomamba,DATASET=${dataset}" \
        scripts/accv2026/slurm_cov_stride_sweep.sbatch
done

# ── PRIORITY 2: E6 missing resolutions ─────────────────────────────────────
log ""
log "=== P2: E6 missing spatial resolutions ==="
for model in r3d_18 mc3_18 timesformer; do
    dir="evaluations/accv2026/spatial_resolution_sweep/${model}_ssv2"
    done=$(find "$dir" -name "res*_summary.csv" 2>/dev/null | wc -l)
    [[ $done -ge 5 ]] && { log "  SKIP E6 $model/ssv2 (5/5 done)"; continue; }
    log "  E6 $model/ssv2: $done/5 done — submitting to complete"
    part=$([[ "$model" == "r3d_18" || "$model" == "mc3_18" ]] && echo "cenvalarc.gpu" || echo "cenvalarc.gpu")
    wait_slot "E6 $model/ssv2"
    submit "E6 $model/ssv2" "$part" "gpu:1" \
        "MODEL=${model},DATASET=ssv2" \
        scripts/accv2026/slurm_spatial_resolution_sweep.sbatch
done

# ── PRIORITY 3: P3 Resolution retraining ───────────────────────────────────
log ""
log "=== P3: Resolution retraining (all models × resolutions × datasets) ==="

declare -a RETRAIN_JOBS=(
    "r3d_18        96"   "r3d_18        160"  "r3d_18        224"  "r3d_18        336"
    "mc3_18        96"   "mc3_18        160"  "mc3_18        224"  "mc3_18        336"
    "r2plus1d_18   96"   "r2plus1d_18   160"  "r2plus1d_18   224"  "r2plus1d_18   336"
    "slowfast_r50  96"   "slowfast_r50  112"  "slowfast_r50  160"  "slowfast_r50  336"
    "timesformer   96"   "timesformer   112"  "timesformer   160"  "timesformer   336"
    "vivit         96"   "vivit         112"  "vivit         160"  "vivit         336"
    "videomae      96"   "videomae      112"  "videomae      160"  "videomae      336"
    "videomamba    96"   "videomamba    112"  "videomamba    160"  "videomamba    336"
)

submitted=0; skipped=0
for job_def in "${RETRAIN_JOBS[@]}"; do
    read -r model target_res <<< "$job_def"
    for dataset in "${DATASETS[@]}"; do
        ckpt="fine_tuned_models/accv2026_${model}_${dataset}_${target_res}px_e10_h200"
        if [[ -f "${ckpt}/config.json" ]]; then
            skipped=$((skipped+1)); continue
        fi
        # CNNs → gpu (A100, faster for small models), Transformers/SSM → cenvalarc.gpu (L40s/H200)
        if [[ "$model" == "r3d_18" || "$model" == "mc3_18" || "$model" == "r2plus1d_18" ]]; then
            part="gpu"; gres="gpu:1"
        else
            part="cenvalarc.gpu"; gres="gpu:1"
        fi
        wait_slot "P3 $model/$dataset@${target_res}px"
        submit "P3 $model/$dataset@${target_res}px" "$part" "$gres" \
            "MODEL=${model},DATASET=${dataset},INPUT_SIZE=${target_res}" \
            scripts/accv2026/slurm_h200_resolution_retrain.sbatch
        submitted=$((submitted+1))
    done
done

log ""
log "=== DONE: P3 $submitted submitted, $skipped skipped ==="
log "W&B: https://wandb.ai/mi3lab/inforates-accv2026"
