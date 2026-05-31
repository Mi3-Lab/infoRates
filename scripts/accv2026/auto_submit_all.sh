#!/usr/bin/env bash
# Auto-submitter unificado: E1 (coverage×stride) + E6 (spatial resolution)
# Monitora a fila e submete próximo job quando slot abre (max 8 total)
# Usage: nohup bash scripts/accv2026/auto_submit_all.sh > evaluations/accv2026/logs/auto_submit_all.log 2>&1 &

set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX_JOBS=8
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

submit_job() {
    local label=$1; local partition=$2; local extra_args=$3
    shift 3
    local result
    result=$(sbatch --partition="$partition" $extra_args "$@" 2>&1)
    log "  → $label | $partition | $result"
    sleep 2
}

wait_for_slot() {
    local label=$1
    while true; do
        local n
        n=$(squeue -u wesleyferreiramaia --noheader 2>/dev/null | wc -l)
        if [[ $n -lt $MAX_JOBS ]]; then break; fi
        log "  $n/$MAX_JOBS jobs running — waiting for slot ($label)..."
        sleep 30
    done
}

# ── E1: coverage×stride — jobs restantes ──────────────────────────────────
E1_REMAINING=(
    "r2plus1d_18 ucf101 gpu ''"
    "slowfast_r50 ucf101 gpu ''"
    "timesformer ucf101 gpu ''"
    "vivit ucf101 gpu ''"
    "videomae ucf101 gpu ''"
    "videomae diving48 gpu ''"
    "videomae autsl gpu ''"
    "videomae driveact gpu ''"
    "videomae epic_kitchens gpu ''"
    "videomamba ucf101 cenvalarc.gpu '--gres=gpu:nvidia_h200_nvl:1'"
    "videomamba ssv2 cenvalarc.gpu '--gres=gpu:nvidia_h200_nvl:1'"
    "videomamba hmdb51 cenvalarc.gpu '--gres=gpu:nvidia_h200_nvl:1'"
    "videomamba diving48 cenvalarc.gpu '--gres=gpu:nvidia_h200_nvl:1'"
    "videomamba autsl cenvalarc.gpu '--gres=gpu:nvidia_h200_nvl:1'"
    "videomamba driveact cenvalarc.gpu '--gres=gpu:nvidia_h200_nvl:1'"
    "videomamba epic_kitchens cenvalarc.gpu '--gres=gpu:nvidia_h200_nvl:1'"
)

# ── E6: spatial resolution — jobs restantes ───────────────────────────────
E6_REMAINING=(
    "r2plus1d_18 ssv2 gpu ''"
    "slowfast_r50 ssv2 gpu ''"
    "vivit ssv2 cenvalarc.gpu '--gres=gpu:nvidia_h200_nvl:1'"
    "videomae ssv2 cenvalarc.gpu '--gres=gpu:nvidia_h200_nvl:1'"
    "videomamba ssv2 cenvalarc.gpu '--gres=gpu:nvidia_h200_nvl:1'"
)

log "=== Auto-submitter iniciado ==="
log "E1 restante: ${#E1_REMAINING[@]} jobs"
log "E6 restante: ${#E6_REMAINING[@]} jobs"
log ""

# Submit E1 remaining
log "--- Submetendo E1 (Coverage×Stride) ---"
for entry in "${E1_REMAINING[@]}"; do
    read -r model dataset partition extra_args <<< "$entry"

    # Skip if already complete
    dir="evaluations/accv2026/coverage_stride_sweep/${model}_${dataset}"
    n=$(find "$dir" -name "*_summary.csv" 2>/dev/null | wc -l)
    if [[ $n -ge 25 ]]; then
        log "  SKIP E1 $model/$dataset ($n/25 done)"
        continue
    fi

    wait_for_slot "E1 $model/$dataset"
    result=$(sbatch --partition="$partition" $extra_args \
        --export=MODEL=${model},DATASET=${dataset} \
        scripts/accv2026/slurm_cov_stride_sweep.sbatch 2>&1)
    log "  E1 $model/$dataset → $result"
    sleep 2
done

log ""
log "--- Submetendo E6 (Spatial Resolution) ---"
for entry in "${E6_REMAINING[@]}"; do
    read -r model dataset partition extra_args <<< "$entry"

    # Skip if already complete
    dir="evaluations/accv2026/spatial_resolution_sweep/${model}_${dataset}"
    n=$(find "$dir" -name "*_summary.csv" 2>/dev/null | wc -l)
    if [[ $n -ge 1 ]]; then
        log "  SKIP E6 $model/$dataset (already done)"
        continue
    fi

    wait_for_slot "E6 $model/$dataset"
    result=$(sbatch --partition="$partition" $extra_args \
        --export=MODEL=${model},DATASET=${dataset} \
        scripts/accv2026/slurm_spatial_resolution_sweep.sbatch 2>&1)
    log "  E6 $model/$dataset → $result"
    sleep 2
done

log ""
log "=== Auto-submitter concluído — todos os jobs submetidos ==="
log "Monitor: squeue -u wesleyferreiramaia"
log "W&B: https://wandb.ai/mi3lab/inforates-accv2026"
