#!/usr/bin/env bash
# Daemon paralelo: retreinar na particao gpu (A100) usando os mesmos lock files do H200 daemon.
# Coordenacao por lock files compartilhados — sem duplicatas entre os dois daemons.
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX_A100=4  # QOS limit: gpu partition allows 4 jobs per user

MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)
RESOLUTIONS=(96 112 160 224)

CKPT_BASE="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"
LOCK_DIR="/data/wesleyferreiramaia/infoRates/evaluations/accv2026/locks"
mkdir -p "$LOCK_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] A100-RETRAIN | $*"; }

n_a100() {
    squeue -u wesleyferreiramaia --noheader --format="%P" 2>/dev/null \
        | grep -c "^gpu$" || true
}

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#RESOLUTIONS[@]} ))

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

lock_name() { echo "${LOCK_DIR}/${1}@${2}@${3}.lock"; }

try_submit_a100() {
    local model=$1 ds=$2 res=$3
    local lock
    lock=$(lock_name "$model" "$ds" "$res")

    is_done "$model" "$ds" "$res" && return 0
    [[ -f "$lock" ]] && return 0  # H200 daemon ja pegou este

    [[ $(n_a100) -ge $MAX_A100 ]] && return 1

    touch "$lock"

    local result
    if result=$(sbatch \
        --partition=gpu \
        --export=MODEL=${model},DATASET=${ds},INPUT_SIZE=${res} \
        scripts/accv2026/slurm_h200_retrain_all.sbatch 2>&1); then
        log "[${model}/${ds}@${res}px] $result"
    else
        log "[${model}/${ds}@${res}px] SBATCH FAILED — $result"
        rm -f "$lock"
        return 1
    fi
    sleep 2
    return 0
}

log "=== A100 RETRAIN DAEMON (complementar ao H200) ==="
log "Particao: gpu (A100) | MAX=${MAX_A100} | Lock compartilhado com H200 daemon"
log "Total targets: ${TOTAL} (8×7×4: 96+112+160+224px) | Pega apenas os que o H200 ainda nao pegou"

while true; do
    # Limpar locks de jobs ja concluidos
    for lockfile in "${LOCK_DIR}"/*.lock; do
        [[ -f "$lockfile" ]] || continue
        base=$(basename "$lockfile" .lock)
        IFS='@' read -r m ds res <<< "$base"
        is_done "$m" "$ds" "$res" && rm -f "$lockfile" || true
    done

    done=$(done_count)
    log "Progress: ${done}/${TOTAL} done | a100_queue=$(n_a100)"
    [[ $done -ge $TOTAL ]] && { log "=== ALL DONE ==="; exit 0; }

    submitted=0
    for res in "${RESOLUTIONS[@]}"; do
        for model in "${MODELS[@]}"; do
            for ds in "${DATASETS[@]}"; do
                if [[ $(n_a100) -lt $MAX_A100 ]]; then
                    try_submit_a100 "$model" "$ds" "$res" && ((submitted++)) || true
                fi
            done
        done
    done

    [[ $submitted -eq 0 ]] && sleep 60 || sleep 5
done
