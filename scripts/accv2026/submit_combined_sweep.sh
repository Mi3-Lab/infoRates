#!/usr/bin/env bash
# Daemon: coverage × stride × resolução — para cada modelo/dataset, roda em
# todas as resoluções não-nativas. Requer checkpoint 224px (do retrain daemon).
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

declare -A NATIVE_RES=(
    [r3d_18]=112 [mc3_18]=112 [r2plus1d_18]=112 [slowfast_r50]=224
    [timesformer]=224 [vivit]=224 [videomae]=224 [videomamba]=224
)

ALL_RES=(96 112 160 224 336)
MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)
COVERAGES="10 25 50 75 100"
STRIDES="1 2 4 8 16"

LOG_DIR="evaluations/accv2026/logs"
OUT_BASE="/data/wesleyferreiramaia/infoRates/evaluations/accv2026/coverage_stride_sweep"
CKPT_BASE="/scratch/wesleyferreiramaia/infoRates/fine_tuned_models"
LOCK_DIR="/data/wesleyferreiramaia/infoRates/evaluations/accv2026/locks_combined"
mkdir -p "$LOG_DIR" "$LOCK_DIR"

MAX=4  # max combined-sweep concurrent

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] COMB-SWEEP | $*"; }

n_comb() {
    squeue -u wesleyferreiramaia --noheader --format="%j" 2>/dev/null \
        | grep "^comb-" | wc -l
}

# Total: cada modelo tem 4 resoluções não-nativas × 7 datasets = 224 jobs
TOTAL=0
for m in "${MODELS[@]}"; do
    nat=${NATIVE_RES[$m]}
    for res in "${ALL_RES[@]}"; do
        [[ $res -eq $nat ]] && continue
        TOTAL=$(( TOTAL + ${#DATASETS[@]} ))
    done
done

has_ckpt() {
    local m=$1 ds=$2
    [[ -f "${CKPT_BASE}/accv2026_${m}_${ds}_224px_e10_h200/config.json" ]]
}

is_done() {
    local m=$1 ds=$2 res=$3
    local f="${OUT_BASE}/${m}_${ds}_res${res}/sweep_summary.csv"
    [[ ! -f "$f" ]] && return 1
    local n
    n=$(python3 -c "import pandas as pd; df=pd.read_csv('$f'); print(len(df))" 2>/dev/null || echo 0)
    [[ $n -ge 25 ]]
}

done_count() {
    local n=0
    for m in "${MODELS[@]}"; do
        nat=${NATIVE_RES[$m]}
        for res in "${ALL_RES[@]}"; do
            [[ $res -eq $nat ]] && continue
            for ds in "${DATASETS[@]}"; do
                is_done "$m" "$ds" "$res" && ((n++)) || true
            done
        done
    done
    echo $n
}

log "=== Combined Sweep Daemon: coverage × stride × resolução ==="
log "Total: ${TOTAL} jobs | MAX=${MAX} concurrent"

while true; do
    # Limpar locks de combos completos
    for lockfile in "${LOCK_DIR}"/*.lock; do
        [[ -f "$lockfile" ]] || continue
        base=$(basename "$lockfile" .lock)
        IFS='@' read -r m ds res <<< "$base"
        is_done "$m" "$ds" "$res" && rm -f "$lockfile" || true
    done

    done=$(done_count)
    waiting_ckpt=0; pending=0
    for m in "${MODELS[@]}"; do
        nat=${NATIVE_RES[$m]}
        for res in "${ALL_RES[@]}"; do
            [[ $res -eq $nat ]] && continue
            for ds in "${DATASETS[@]}"; do
                is_done "$m" "$ds" "$res" && continue
                [[ -f "${LOCK_DIR}/${m}@${ds}@${res}.lock" ]] && continue
                has_ckpt "$m" "$ds" && ((pending++)) || ((waiting_ckpt++))
            done
        done
    done

    log "Progress: ${done}/${TOTAL} done | pending=${pending} | waiting_ckpt=${waiting_ckpt} | queue=$(n_comb)"
    [[ $done -ge $TOTAL ]] && { log "=== ALL DONE ==="; exit 0; }

    # Submeter novos jobs
    for m in "${MODELS[@]}"; do
        nat=${NATIVE_RES[$m]}
        venv="/data/wesleyferreiramaia/infoRates/.venv"
        [[ "$m" == "videomamba" ]] && venv="/data/wesleyferreiramaia/infoRates/.venv_mamba"

        for res in "${ALL_RES[@]}"; do
            [[ $res -eq $nat ]] && continue
            for ds in "${DATASETS[@]}"; do
                is_done "$m" "$ds" "$res" && continue
                [[ -f "${LOCK_DIR}/${m}@${ds}@${res}.lock" ]] && continue
                has_ckpt "$m" "$ds" || continue  # aguarda retrain

                # Proteção extra: verificar se já há job com esse out_dir na fila
                out_dir="${OUT_BASE}/${m}_${ds}_res${res}"
                if find "${out_dir}" -name "*.csv" -newer "${LOCK_DIR}" 2>/dev/null | grep -q .; then
                    touch "${LOCK_DIR}/${m}@${ds}@${res}.lock"
                    continue
                fi

                [[ $(n_comb) -ge $MAX ]] && break 3

                out_dir="${OUT_BASE}/${m}_${ds}_res${res}"
                touch "${LOCK_DIR}/${m}@${ds}@${res}.lock"

                if JID=$(sbatch \
                    --job-name="comb-${m:0:6}-${ds:0:5}-${res}" \
                    --partition=gpu \
                    --gres=gpu:1 \
                    --cpus-per-task=8 \
                    --mem=64G \
                    --time=03:00:00 \
                    --output="${LOG_DIR}/comb-sweep-${m}-${ds}-${res}px-%j.out" \
                    --error="${LOG_DIR}/comb-sweep-${m}-${ds}-${res}px-%j.err" \
                    --wrap="
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME=/scratch/wesleyferreiramaia/infoRates/torch_cache
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_MODE=disabled
source ${venv}/bin/activate
echo '=== Combined Sweep: ${m} / ${ds} @ ${res}px ==='
nvidia-smi | grep -E 'Name|L40|H200|A100' | head -2 || true
mkdir -p ${out_dir}
python /data/wesleyferreiramaia/infoRates/scripts/accv2026/sweep_coverage_stride.py \
    --model ${m} --dataset ${ds} --input-size ${res} \
    --coverages ${COVERAGES} --strides ${STRIDES} \
    --output-dir ${out_dir}
echo DONE
" 2>&1 | grep -o "[0-9]*"); then
                    log "[${m}/${ds}@${res}px] Job ${JID}"
                else
                    log "[${m}/${ds}@${res}px] SBATCH FAILED"
                    rm -f "${LOCK_DIR}/${m}@${ds}@${res}.lock"
                fi
                sleep 3
            done
        done
    done

    sleep 60
done
