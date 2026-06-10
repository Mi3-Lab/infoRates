#!/usr/bin/env bash
# P3 daemon — 336px retraining with 2×L40s GPUs (DDP, batch=128 effective)
# Submits to cenvalarc.gpu only. Max 4 concurrent jobs (= 8 GPUs).
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX=4   # 4 jobs × 2 GPUs = 8 GPUs — safe for cenvalarc.gpu
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)
MODELS=(r3d_18 mc3_18 slowfast_r50)
RES=336
TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))  # 21

log()    { echo "[$(date '+%Y-%m-%d %H:%M:%S')] 336px-2GPU | $*"; }
n_jobs() { squeue -u wesleyferreiramaia -p cenvalarc.gpu --noheader 2>/dev/null | wc -l; }

CKPT_BASE="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"

done_count() {
    local n=0
    for m in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            [[ -f "${CKPT_BASE}/accv2026_${m}_${ds}_${RES}px_e10_h200/config.json" ]] && ((n++)) || true
        done
    done
    echo $n
}

try_submit() {
    local model=$1 dataset=$2
    local ckpt="${CKPT_BASE}/accv2026_${model}_${dataset}_${RES}px_e10_h200"
    [[ -f "${ckpt}/config.json" ]] && return 0          # já feito
    [[ $(n_jobs) -ge $MAX ]]        && return 1          # fila cheia
    local result
    result=$(sbatch --partition=cenvalarc.gpu --gres=gpu:2 \
        --export=MODEL=${model},DATASET=${dataset} \
        scripts/accv2026/slurm_336px_2gpu.sbatch 2>&1)
    log "[${model}/${dataset}@336px] $result"
    sleep 2
    return 0
}

log "=== 336px 2-GPU DAEMON started — target: ${TOTAL} checkpoints ==="
log "Config: 2×L40s per job (batch=128), cenvalarc.gpu only, max ${MAX} concurrent"

while true; do
    done=$(done_count)
    log "Progress: ${done}/${TOTAL} | queue=$(n_jobs)/${MAX}"

    [[ $done -ge $TOTAL ]] && { log "=== ALL ${TOTAL} 336px DONE ==="; exit 0; }

    submitted=0
    for model in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            ckpt="${CKPT_BASE}/accv2026_${model}_${ds}_${RES}px_e10_h200"
            [[ -f "${ckpt}/config.json" ]] && continue
            if [[ $(n_jobs) -lt $MAX ]]; then
                try_submit "$model" "$ds" && ((submitted++)) || true
            fi
        done
    done

    if [[ $submitted -eq 0 ]]; then
        sleep 60   # aguardar slot liberar
    else
        sleep 5    # recheck rápido após submissão
    fi
done
