#!/usr/bin/env bash
# Daemon: Retreinar TODOS os modelos em 224px e 336px na H200
# Usa lock files para evitar duplicatas
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX_H200=6     # max jobs concurrent na H200 (3 nós × 2 GPUs = 6 jobs)
MAX_L40S=4     # fallback para L40s se H200 lotada
MAX_TOTAL=4

MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)
RESOLUTIONS=(224 336)

CKPT_BASE="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"
LOCK_DIR="/tmp/h200_retrain_locks"
mkdir -p "$LOCK_DIR"

log()     { echo "[$(date '+%Y-%m-%d %H:%M:%S')] H200-ALL | $*"; }
n_h200()  { squeue -u wesleyferreiramaia -p cenvalarc.gpu --noheader 2>/dev/null | grep "retrain-h200" | wc -l; }
n_l40s()  { squeue -u wesleyferreiramaia -p cenvalarc.gpu --noheader 2>/dev/null | grep "retrain-336px" | wc -l; }
n_total() { squeue -u wesleyferreiramaia --noheader 2>/dev/null | wc -l; }

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#RESOLUTIONS[@]} ))  # 8×7×2 = 112

done_count() {
    local n=0
    for m in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            for res in "${RESOLUTIONS[@]}"; do
                cfg="${CKPT_BASE}/accv2026_${m}_${ds}_${res}px_e10_h200/config.json"
                if [[ -f "$cfg" ]]; then
                    val=$(python3 -c "import json; d=json.load(open('$cfg')); print(1 if d.get('val_acc',0)>0 and d.get('epoch',0)>=9 else 0)" 2>/dev/null || echo 0)
                    [[ "$val" == "1" ]] && ((n++)) || true
                fi
            done
        done
    done
    echo $n
}

is_done() {
    local model=$1 ds=$2 res=$3
    local cfg="${CKPT_BASE}/accv2026_${model}_${ds}_${res}px_e10_h200/config.json"
    [[ ! -f "$cfg" ]] && return 1
    local val
    val=$(python3 -c "import json; d=json.load(open('$cfg')); print(1 if d.get('val_acc',0)>0 and d.get('epoch',0)>=9 else 0)" 2>/dev/null || echo 0)
    [[ "$val" == "1" ]]
}

try_submit() {
    local model=$1 ds=$2 res=$3
    local lock="${LOCK_DIR}/${model}_${ds}_${res}.lock"

    is_done "$model" "$ds" "$res" && return 0
    [[ -f "$lock" ]] && return 0  # já submetido

    [[ $(n_total) -ge $MAX_TOTAL ]] && return 1  # fila cheia

    touch "$lock"

    local result
    result=$(sbatch \
        --constraint=H200 \
        --export=MODEL=${model},DATASET=${ds},INPUT_SIZE=${res} \
        scripts/accv2026/slurm_h200_retrain_all.sbatch 2>&1)

    log "[${model}/${ds}@${res}px → H200] $result"
    sleep 2
    return 0
}

log "=== H200 ALL-RESOLUTIONS DAEMON ==="
log "Target: ${TOTAL} jobs (8 models × 7 datasets × 2 resolutions: 224+336px)"
log "H200: 141GB/GPU, batch=32-96 (sem collapse!)"

while true; do
    # Limpar locks de jobs já concluídos
    for lockfile in "${LOCK_DIR}"/*.lock; do
        [[ -f "$lockfile" ]] || continue
        base=$(basename "$lockfile" .lock)
        IFS='_' read -ra parts <<< "$base"
        res="${parts[-1]}"
        ds="${parts[-2]}"
        model="${base%_${ds}_${res}}"
        is_done "$model" "$ds" "$res" && rm -f "$lockfile" || true
    done

    done=$(done_count)
    log "Progress: ${done}/${TOTAL} done | queue=$(n_total)"
    [[ $done -ge $TOTAL ]] && { log "=== ALL DONE ==="; rm -f "${LOCK_DIR}"/*.lock; exit 0; }

    submitted=0
    for res in "${RESOLUTIONS[@]}"; do
        for model in "${MODELS[@]}"; do
            for ds in "${DATASETS[@]}"; do
                if [[ $(n_total) -lt $MAX_TOTAL ]]; then
                    try_submit "$model" "$ds" "$res" && ((submitted++)) || true
                fi
            done
        done
    done

    [[ $submitted -eq 0 ]] && sleep 60 || sleep 5
done
