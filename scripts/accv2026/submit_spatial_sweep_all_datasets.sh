#!/usr/bin/env bash
# Daemon: spatial resolution sweep (eval-only) para todos modelos × datasets
# Submete somente quando checkpoint 224px existe; fica em loop até tudo terminar.
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
DATASETS=(ucf101 hmdb51 diving48 autsl driveact epic_kitchens)  # SSv2 já tem

LOG_DIR="evaluations/accv2026/logs"
SWEEP_DIR="evaluations/accv2026/spatial_resolution_sweep"
CKPT_BASE="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"
LOCK_DIR="/data/wesleyferreiramaia/infoRates/evaluations/accv2026/locks_spatial"
mkdir -p "$LOG_DIR" "$LOCK_DIR"

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))  # 8×6=48

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] SPATIAL | $*"; }

# Conta apenas jobs de spatial sweep
n_spatial() {
    squeue -u wesleyferreiramaia --noheader --format="%j" 2>/dev/null \
        | grep "^spatial-sweep" | wc -l
}

MAX=4  # max spatial-sweep concurrent (GPU leve, 1 GPU por job)

has_ckpt() {
    local m=$1 ds=$2
    [[ -f "${CKPT_BASE}/accv2026_${m}_${ds}_224px_e10_h200/config.json" ]]
}

is_done() {
    local m=$1 ds=$2
    [[ -f "${SWEEP_DIR}/${m}_${ds}/spatial_sweep_summary.csv" ]]
}

done_count() {
    local n=0
    for m in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            is_done "$m" "$ds" && ((n++)) || true
        done
    done
    echo $n
}

log "=== Spatial Sweep Daemon — todos datasets exceto SSv2 ==="
log "Total: ${TOTAL} combos | MAX=${MAX} concurrent"

while true; do
    # Limpar locks de combos já completos
    for lockfile in "${LOCK_DIR}"/*.lock; do
        [[ -f "$lockfile" ]] || continue
        base=$(basename "$lockfile" .lock)
        IFS='@' read -r m ds <<< "$base"
        is_done "$m" "$ds" && rm -f "$lockfile" || true
    done

    done=$(done_count)
    pending=0; waiting_ckpt=0
    for m in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            is_done "$m" "$ds" && continue
            [[ -f "${LOCK_DIR}/${m}@${ds}.lock" ]] && continue
            has_ckpt "$m" "$ds" && ((pending++)) || ((waiting_ckpt++))
        done
    done

    log "Progress: ${done}/${TOTAL} done | pending=${pending} | waiting_ckpt=${waiting_ckpt} | queue=$(n_spatial)"
    [[ $done -ge $TOTAL ]] && { log "=== ALL DONE ==="; exit 0; }

    # Submeter novos jobs
    for m in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            is_done "$m" "$ds" && continue
            [[ -f "${LOCK_DIR}/${m}@${ds}.lock" ]] && continue
            has_ckpt "$m" "$ds" || continue  # checkpoint ainda não pronto

            [[ $(n_spatial) -ge $MAX ]] && break 2

            venv="/data/wesleyferreiramaia/infoRates/.venv"
            [[ "$m" == "videomamba" ]] && venv="/data/wesleyferreiramaia/infoRates/.venv_mamba"

            touch "${LOCK_DIR}/${m}@${ds}.lock"

            if JID=$(sbatch \
                --job-name="spatial-sweep-${m}-${ds}" \
                --partition=gpu \
                --gres=gpu:1 \
                --cpus-per-task=8 \
                --mem=64G \
                --time=03:00:00 \
                --output="${LOG_DIR}/spatial-sweep-${m}-${ds}-%j.out" \
                --error="${LOG_DIR}/spatial-sweep-${m}-${ds}-%j.err" \
                --wrap="
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME=/scratch/wesleyferreiramaia/infoRates/torch_cache
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_MODE=offline
source ${venv}/bin/activate
mkdir -p /data/wesleyferreiramaia/infoRates/${SWEEP_DIR}/${m}_${ds}
python /data/wesleyferreiramaia/infoRates/scripts/accv2026/sweep_spatial_resolution.py \
    --model ${m} --dataset ${ds} \
    --output-dir /data/wesleyferreiramaia/infoRates/${SWEEP_DIR}/${m}_${ds}
" 2>&1 | grep -o "[0-9]*"); then
                log "[${m}/${ds}] Submitted job ${JID}"
            else
                log "[${m}/${ds}] SBATCH FAILED"
                rm -f "${LOCK_DIR}/${m}@${ds}.lock"
            fi
            sleep 3
        done
    done

    sleep 60
done
