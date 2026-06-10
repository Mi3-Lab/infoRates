#!/usr/bin/env bash
# MASTER DAEMON — submete para AMBAS as partições em paralelo
# gpu (A100, 40GB) + cenvalarc.gpu (L40s 48GB + H200 141GB)
# Limite: 4 jobs em cada partição = 8 total
# Detecta GPU em runtime e ajusta batch size automaticamente
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX_PER_PART=4   # limite QOS por partição
PARTITIONS=(cenvalarc.gpu gpu)

MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)
RESOLUTIONS=(224 336)

CKPT_BASE="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"
LOCK_DIR="/tmp/master_daemon_locks"; mkdir -p "$LOCK_DIR"

log() { echo "[$(date '+%H:%M:%S')] MASTER | $*"; }

n_part() { squeue -u wesleyferreiramaia -p "$1" --noheader 2>/dev/null | wc -l; }
n_total(){ squeue -u wesleyferreiramaia --noheader 2>/dev/null | wc -l; }

is_done() {
    local cfg="${CKPT_BASE}/accv2026_${1}_${2}_${3}px_e10_h200/config.json"
    [[ -f "$cfg" ]] && python3 -c "
import json,sys
d=json.load(open('$cfg'))
sys.exit(0 if d.get('val_acc',0)>0 and d.get('epoch',0)>=9 else 1)" 2>/dev/null
}

submit_one() {
    local model=$1 ds=$2 res=$3 part=$4
    local lock="${LOCK_DIR}/${model}_${ds}_${res}.lock"
    local venv="/data/wesleyferreiramaia/infoRates/.venv"
    [[ "$model" == "videomamba" ]] && venv="/data/wesleyferreiramaia/infoRates/.venv_mamba"

    is_done "$model" "$ds" "$res" && return 0
    [[ -f "$lock" ]] && return 0
    [[ $(n_part "$part") -ge $MAX_PER_PART ]] && return 1

    touch "$lock"
    local jid
    # H200 (cenvalarc.gpu): 1 GPU single com batch grande = muito mais rápido
    # A100 (gpu): 2 GPUs DDP com batch=16 = fallback
    local sbatch_script
    if [[ "$part" == "cenvalarc.gpu" ]]; then
        sbatch_script="scripts/accv2026/slurm_h200_single.sbatch"
    else
        sbatch_script="scripts/accv2026/slurm_h200_retrain_all.sbatch"
    fi
    jid=$(sbatch \
        --partition="$part" \
        --export=MODEL="$model",DATASET="$ds",INPUT_SIZE="$res" \
        "$sbatch_script" 2>&1 | grep -o "[0-9]*")

    if [[ -n "$jid" && "$jid" =~ ^[0-9]+$ ]]; then
        log "[${model}/${ds}@${res}px → ${part}] Job $jid"
        sleep 2; return 0
    else
        rm -f "$lock"
        return 1
    fi
}

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#RESOLUTIONS[@]} ))
log "=== MASTER DAEMON iniciado ==="
log "Target: ${TOTAL} checkpoints (8×7×2)"
log "Partições: ${PARTITIONS[*]} | MAX ${MAX_PER_PART} cada = $((MAX_PER_PART * ${#PARTITIONS[@]})) total"

while true; do
    # Limpar locks de jobs concluídos
    for lk in "${LOCK_DIR}"/*.lock; do
        [[ -f "$lk" ]] || continue
        base=$(basename "$lk" .lock)
        res="${base##*_}"; tmp="${base%_*}"; ds="${tmp##*_}"; model="${tmp%_*}"
        is_done "$model" "$ds" "$res" && rm -f "$lk" || true
    done

    done=0
    for m in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            for res in "${RESOLUTIONS[@]}"; do
                is_done "$m" "$ds" "$res" && ((done++)) || true
            done
        done
    done

    log "Progress: ${done}/${TOTAL} | fila: $(n_total) jobs"
    [[ $done -ge $TOTAL ]] && { log "=== TUDO CONCLUÍDO ==="; exit 0; }

    # Tentar submeter para cada partição
    for part in "${PARTITIONS[@]}"; do
        [[ $(n_part "$part") -ge $MAX_PER_PART ]] && continue
        for res in "${RESOLUTIONS[@]}"; do
            for model in "${MODELS[@]}"; do
                for ds in "${DATASETS[@]}"; do
                    [[ $(n_part "$part") -ge $MAX_PER_PART ]] && break 3
                    submit_one "$model" "$ds" "$res" "$part" || true
                done
            done
        done
    done

    sleep 60
done
