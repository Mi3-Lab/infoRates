#!/usr/bin/env bash
# Auto-submit coverage×stride sweep jobs when slots become available
# Monitors squeue and submits next job when <4 jobs are running
# Usage: nohup bash scripts/accv2026/auto_submit_coverage_stride.sh > auto_submit.log 2>&1 &

set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

# All 8 models × 7 datasets = 56 combos
declare -a MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
declare -a DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)

TOTAL_COMBOS=$((${#MODELS[@]} * ${#DATASETS[@]}))
SUBMITTED=0

echo "=== Auto-submitter started at $(date) ==="
echo "Will submit ${TOTAL_COMBOS} jobs (max 4 concurrent)"
echo ""

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        # Poll until <4 jobs are running
        while true; do
            n_running=$(squeue -u wesleyferreiramaia --format='%.10i' 2>/dev/null | grep -c . || echo 0)
            if [[ $n_running -lt 4 ]]; then
                break
            fi
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $n_running jobs running, waiting... (need slot for ${model}/${dataset})"
            sleep 30
        done

        # Submit next job
        SUBMITTED=$((SUBMITTED + 1))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Submitting ${SUBMITTED}/${TOTAL_COMBOS}: ${model} / ${dataset}"
        sbatch --export=MODEL=${model},DATASET=${dataset} scripts/accv2026/slurm_cov_stride_sweep.sbatch
        sleep 2  # brief pause between submissions
    done
done

echo ""
echo "=== All ${TOTAL_COMBOS} jobs submitted at $(date) ==="
echo "Monitor progress with: squeue -u wesleyferreiramaia"
echo "View W&B: https://wandb.ai/mi3lab/inforates-accv2026"
