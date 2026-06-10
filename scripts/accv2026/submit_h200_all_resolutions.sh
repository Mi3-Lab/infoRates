#!/usr/bin/env bash
# Daemon: Retreinar TODOS os modelos em 224px e 336px na H200
# Usa lock files para evitar duplicatas
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX_TOTAL=4  # QOS limit: cenvalarc.gpu allows 4 running jobs per user

MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens finegym)
RESOLUTIONS=(96 112 160 224)

CKPT_BASE="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"
# Lock dir persistente (não some entre reinicios)
LOCK_DIR="/data/wesleyferreiramaia/infoRates/evaluations/accv2026/locks"
mkdir -p "$LOCK_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] H200-ALL | $*"; }

# Conta todos os jobs na partição cenvalarc.gpu (QOS limit se aplica a TODOS)
n_retrain() {
    squeue -u wesleyferreiramaia --noheader --format="%P" 2>/dev/null \
        | grep -c "^cenvalarc.gpu$" || true
}

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#RESOLUTIONS[@]} ))  # 8×7×2 = 112

# Checa se um checkpoint está completo: config.json existe
# (mesmo critério que o sbatch usa para pular o job)
is_done() {
    local model=$1 ds=$2 res=$3
    [[ -f "${CKPT_BASE}/accv2026_${model}_${ds}_${res}px_e10_h200/config.json" ]]
}

done_count() {
    local n=0
    for m in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            for res in "${RESOLUTIONS[@]}"; do
                is_done "$m" "$ds" "$res" && ((n++)) || true
            done
        done
    done
    echo $n
}

# Lock nomeado com @ como separador (evita colisão com _ em epic_kitchens, r3d_18 etc.)
lock_name() { echo "${LOCK_DIR}/${1}@${2}@${3}.lock"; }

try_submit() {
    local model=$1 ds=$2 res=$3
    local lock
    lock=$(lock_name "$model" "$ds" "$res")

    is_done "$model" "$ds" "$res" && return 0
    [[ -f "$lock" ]] && return 0  # já submetido, aguardando

    [[ $(n_retrain) -ge $MAX_TOTAL ]] && return 1  # fila cheia

    touch "$lock"

    local result
    if result=$(sbatch \
        --export=MODEL=${model},DATASET=${ds},INPUT_SIZE=${res} \
        scripts/accv2026/slurm_h200_retrain_all.sbatch 2>&1); then
        log "[${model}/${ds}@${res}px] $result"
    else
        log "[${model}/${ds}@${res}px] SBATCH FAILED — $result"
        rm -f "$lock"  # remove lock para tentar de novo na próxima iteração
        return 1
    fi
    sleep 2
    return 0
}

log "=== H200 ALL-RESOLUTIONS DAEMON ==="
log "Target: ${TOTAL} jobs (8 models × 7 datasets × 4 resolutions: 96+112+160+224px)"
log "MAX_TOTAL=${MAX_TOTAL} retrain jobs concurrent"

while true; do
    # Limpar locks de jobs já concluídos (parsing seguro com @ como separador)
    for lockfile in "${LOCK_DIR}"/*.lock; do
        [[ -f "$lockfile" ]] || continue
        base=$(basename "$lockfile" .lock)
        IFS='@' read -r m ds res <<< "$base"
        is_done "$m" "$ds" "$res" && rm -f "$lockfile" || true
    done

    done=$(done_count)
    log "Progress: ${done}/${TOTAL} done | retrain_queue=$(n_retrain)"
    [[ $done -ge $TOTAL ]] && { log "=== ALL DONE ==="; rm -f "${LOCK_DIR}"/*.lock; exit 0; }

    submitted=0
    for res in "${RESOLUTIONS[@]}"; do
        for model in "${MODELS[@]}"; do
            for ds in "${DATASETS[@]}"; do
                if [[ $(n_retrain) -lt $MAX_TOTAL ]]; then
                    try_submit "$model" "$ds" "$res" && ((submitted++)) || true
                fi
            done
        done
    done

    [[ $submitted -eq 0 ]] && sleep 60 || sleep 5
done
