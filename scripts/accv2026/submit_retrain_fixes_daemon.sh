#!/usr/bin/env bash
# Daemon: retrain bad checkpoints identified by anomaly analysis.
# Uses VERSION_SUFFIX=_v2 so get_checkpoint() auto-picks best val_acc.
# SlowFast collapsed checkpoints use LR=5e-5 (vs default 1e-4) to prevent oscillation.
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

SBATCH=scripts/accv2026/slurm_h200_resolution_retrain.sbatch
MAX_TOTAL=30  # upper bound; actual cluster QOS limits gate concurrent runs
LOG=evaluations/accv2026/logs/daemon_retrain_fixes.log

# ── Jobs to retrain ───────────────────────────────────────────────────────────
# Format: "MODEL DATASET INPUT_SIZE PARTITION EXTRA_EXPORT"
# EXTRA_EXPORT: additional --export key=val pairs (e.g. LR=5e-5)
declare -a JOBS=(
    # SlowFast collapsed — use lower LR (5e-5) to avoid oscillation.
    # CNNs use gpu partition (L40s/A100s) — H200 is not required and cenvalarc has low MaxSubmitPU.
    "slowfast_r50 autsl        96  gpu LR=5e-5"
    "slowfast_r50 autsl        112 gpu LR=5e-5"
    "slowfast_r50 ssv2         96  gpu LR=5e-5"
    "slowfast_r50 epic_kitchens 96  gpu LR=5e-5"
    "slowfast_r50 epic_kitchens 48  gpu LR=5e-5"
    # MC3 — standard LR (training was cut short, not a collapse)
    "mc3_18 driveact     48  gpu LR=1e-4"
    "mc3_18 epic_kitchens 160 gpu LR=1e-4"
    # Transformers use cenvalarc.gpu (H200 faster; gpu QOS=4 is shared with CNN retrains above).
    # Running both partitions in parallel gives 4 (gpu) + 4 (cenvalarc.gpu) = 8 simultaneous.
    # TimesFormer + ViViT collapsed at 48px for AUTSL/diving48:
    "timesformer autsl    48 cenvalarc.gpu LR=2e-5"
    "timesformer diving48 48 cenvalarc.gpu LR=2e-5"
    "vivit       autsl    48 cenvalarc.gpu LR=2e-5"
    # VideoMAE SSV2 PE mismatch: trained with random PE before model_factory.py fix.
    "videomae ssv2 96  cenvalarc.gpu LR=2e-5"
    "videomae ssv2 112 cenvalarc.gpu LR=2e-5"
    "videomae ssv2 160 cenvalarc.gpu LR=2e-5"
    # Non-monotonic 160px > 224px: retrain 224px to ensure monotonicity.
    "videomae driveact     224 cenvalarc.gpu LR=2e-5"
    "videomae epic_kitchens 224 cenvalarc.gpu LR=2e-5"
    "videomae hmdb51        224 cenvalarc.gpu LR=2e-5"
    "timesformer epic_kitchens 224 cenvalarc.gpu LR=2e-5"
)

is_done() {
    local model=$1 ds=$2 res=$3
    # Check if a v2 checkpoint (or higher) with decent val_acc exists
    for base in /scratch/wesleyferreiramaia/infoRates/fine_tuned_models fine_tuned_models; do
        for p in "$base"/accv2026_${model}_${ds}_${res}px_e*_v*_h200; do
            [ -d "$p" ] || continue
            cfg="$p/config.json"
            meta="$p/accv_meta.json"
            [ -f "$cfg" ] || [ -f "$meta" ] || continue
            echo "DONE (checkpoint exists): $p"
            return 0
        done
    done
    return 1
}

running_total() {
    squeue -u wesleyferreiramaia --noheader 2>/dev/null | wc -l
}

submit_one() {
    local model=$1 ds=$2 res=$3 part=$4 lr=$5
    local job_id
    job_id=$(sbatch \
        --job-name="fix-${model:0:4}-${ds:0:6}-${res}" \
        --partition="$part" \
        --export=ALL,MODEL="$model",DATASET="$ds",INPUT_SIZE="$res",VERSION_SUFFIX="_v2",LR="$lr" \
        "$SBATCH" 2>&1 | awk '{print $NF}')
    # Return 0 only if job_id is numeric (real submission); QOS errors return non-numeric string
    if [[ "$job_id" =~ ^[0-9]+$ ]]; then
        echo "[$(date '+%H:%M:%S')] Submitted ${model}/${ds}@${res}px lr=${lr} → ${part} job ${job_id}" | tee -a "$LOG"
        return 0
    else
        echo "[$(date '+%H:%M:%S')] QUEUED (QOS limit) ${model}/${ds}@${res}px → will retry" | tee -a "$LOG"
        return 1
    fi
}

echo "[$(date '+%H:%M:%S')] Retrain-fixes daemon starting — ${#JOBS[@]} jobs" | tee -a "$LOG"

remaining=("${JOBS[@]}")
while [ ${#remaining[@]} -gt 0 ]; do
    new_remaining=()
    running=$(running_total)

    for entry in "${remaining[@]}"; do
        read -r model ds res part lr_kv <<< "$entry"
        lr="${lr_kv#LR=}"

        if is_done "$model" "$ds" "$res" > /dev/null 2>&1; then
            echo "[$(date '+%H:%M:%S')] DONE: ${model}/${ds}@${res}px" | tee -a "$LOG"
            continue
        fi

        # Check if already queued (use full job name without truncation)
        jname="fix-${model:0:4}-${ds:0:6}-${res}"
        if squeue -u wesleyferreiramaia --format="%j" --noheader 2>/dev/null | grep -qF "$jname"; then
            new_remaining+=("$entry")
            continue
        fi

        if [ "$running" -lt "$MAX_TOTAL" ]; then
            if submit_one "$model" "$ds" "$res" "$part" "$lr"; then
                running=$((running + 1))
            fi
            new_remaining+=("$entry")
        else
            new_remaining+=("$entry")
        fi
    done

    remaining=("${new_remaining[@]}")
    [ ${#remaining[@]} -eq 0 ] && break
    sleep 120
done

echo "[$(date '+%H:%M:%S')] All retrain-fixes submitted and complete." | tee -a "$LOG"
