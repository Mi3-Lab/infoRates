#!/usr/bin/env bash
# Daemon: submete 1 job por dataset para rodar inferência em 48px com checkpoint nativo.
# O script skip logic já pula 96/112/160/224 (existem), só avalia 48px.
# MAX=1 (não sobrecarregar fila gpu — compartilhada com outros treinos)
set -uo pipefail

DATASETS=(ucf101 hmdb51 diving48 autsl driveact epic_kitchens ssv2)
LOG_DIR="/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs"
SWEEP_DIR="/data/wesleyferreiramaia/infoRates/evaluations/accv2026/spatial_resolution_sweep"
LOCK_DIR="/data/wesleyferreiramaia/infoRates/evaluations/accv2026/locks_spatial48"
MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
MAX=1

mkdir -p "$LOG_DIR" "$LOCK_DIR"

log() { echo "[$(date '+%H:%M:%S')] SPATIAL-48 | $*"; }

n_spatial48() {
    squeue -u wesleyferreiramaia --noheader --format="%j" 2>/dev/null \
        | grep "^sp48-" | wc -l
}

is_done() {
    local ds=$1
    local n=0
    for m in "${MODELS[@]}"; do
        [[ -f "${SWEEP_DIR}/${m}_${ds}/res48_summary.csv" ]] && ((n++)) || true
    done
    [[ $n -ge ${#MODELS[@]} ]]
}

done_count() {
    local n=0
    for ds in "${DATASETS[@]}"; do
        is_done "$ds" && ((n++)) || true
    done
    echo $n
}

TOTAL=${#DATASETS[@]}
log "=== Spatial 48px Daemon — ${TOTAL} datasets × 8 models ==="

while true; do
    done=$(done_count)
    log "Progress: ${done}/${TOTAL} datasets done | queue=$(n_spatial48)"
    [[ $done -ge $TOTAL ]] && { log "=== ALL DONE ==="; exit 0; }

    for ds in "${DATASETS[@]}"; do
        is_done "$ds" && continue
        [[ -f "${LOCK_DIR}/${ds}.lock" ]] && continue
        [[ $(n_spatial48) -ge $MAX ]] && break

        touch "${LOCK_DIR}/${ds}.lock"
        if JID=$(sbatch \
            --job-name="sp48-${ds}" \
            --partition=gpu \
            --gres=gpu:1 \
            --cpus-per-task=8 \
            --mem=64G \
            --time=01:00:00 \
            --output="${LOG_DIR}/sp48-${ds}-%j.out" \
            --error="${LOG_DIR}/sp48-${ds}-%j.err" \
            --wrap="
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME=/scratch/wesleyferreiramaia/infoRates/torch_cache
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_MODE=offline
export WANDB_DIR=/scratch/wesleyferreiramaia/wandb

for MODEL in r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae; do
    echo '--- \${MODEL}/${ds}@48px ---'
    source /data/wesleyferreiramaia/infoRates/.venv/bin/activate
    python /data/wesleyferreiramaia/infoRates/scripts/accv2026/sweep_spatial_resolution.py \
        --model \$MODEL --dataset ${ds} --resolutions 48 --batch-size 32 \
        || echo '[WARN] '\$MODEL/${ds}' failed'
    deactivate
done

echo '--- videomamba/${ds}@48px ---'
source /data/wesleyferreiramaia/infoRates/.venv_mamba/bin/activate
python /data/wesleyferreiramaia/infoRates/scripts/accv2026/sweep_spatial_resolution.py \
    --model videomamba --dataset ${ds} --resolutions 48 --batch-size 16 \
    || echo '[WARN] videomamba/${ds} failed'
deactivate
echo '=== DONE ${ds} ==='
" 2>&1 | grep -o "[0-9]*"); then
            log "[${ds}] Submitted job ${JID}"
        else
            log "[${ds}] SBATCH FAILED"
            rm -f "${LOCK_DIR}/${ds}.lock"
        fi
        sleep 3
    done

    # Limpar locks de datasets já completos
    for ds in "${DATASETS[@]}"; do
        is_done "$ds" && rm -f "${LOCK_DIR}/${ds}.lock" 2>/dev/null || true
    done

    sleep 60
done
