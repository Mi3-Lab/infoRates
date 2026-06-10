#!/usr/bin/env bash
# Daemon para Spatial Sweep + Combined Sweep (inferência only, sem treino)
# Usa 1 GPU por job (mais eficiente que treino)
# Roda em paralelo com o master_daemon sem conflitar (partição separada ou nós específicos)
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MAX=8   # max jobs de sweep simultâneos (1 GPU cada = 8 GPUs usadas)
LOG_DIR="evaluations/accv2026/logs"
SPATIAL_OUT="evaluations/accv2026/spatial_resolution_sweep"
COMBINED_OUT="evaluations/accv2026/coverage_stride_sweep"

MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)
declare -A NATIVE_RES=(
    [r3d_18]=112 [mc3_18]=112 [r2plus1d_18]=112 [slowfast_r50]=224
    [timesformer]=224 [vivit]=224 [videomae]=224 [videomamba]=224
)
ALL_RES=(96 112 160 224 336)

log()    { echo "[$(date '+%H:%M:%S')] SWEEP | $*"; }
n_sw()   { squeue -u wesleyferreiramaia --noheader 2>/dev/null | grep -c "sweep\|comb-" || true; }
n_total(){ squeue -u wesleyferreiramaia --noheader 2>/dev/null | wc -l; }

submit_job() {
    local name=$1 venv=$2 cmd=$3 out=$4 err=$5
    local n; n=$(n_sw)
    [[ $n -ge $MAX ]] && return 1
    # Limite total respeitando QOS: não ultrapassar 4 por partição
    local n_cenv; n_cenv=$(squeue -u wesleyferreiramaia -p cenvalarc.gpu --noheader 2>/dev/null | wc -l)
    local n_gpu;  n_gpu=$(squeue  -u wesleyferreiramaia -p gpu          --noheader 2>/dev/null | wc -l)

    local part=""
    if   [[ $n_cenv -lt 4 ]]; then part="cenvalarc.gpu"
    elif [[ $n_gpu  -lt 4 ]]; then part="gpu"
    else return 1
    fi

    local jid
    jid=$(sbatch \
        --job-name="$name" \
        --partition="$part" \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=48G \
        --time=03:00:00 \
        --output="$out" \
        --error="$err" \
        --wrap="
set -e
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME=/scratch/wesleyferreiramaia/infoRates/torch_cache
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_MODE=disabled
source ${venv}/bin/activate
${cmd}" 2>&1 | grep -o "[0-9]*")

    if [[ -n "$jid" && "$jid" =~ ^[0-9]+$ ]]; then
        log "[$name → $part] Job $jid"
        sleep 3; return 0
    fi
    return 1
}

log "=== SWEEP DAEMON iniciado (Spatial + Combined) ==="
log "MAX: $MAX jobs paralelos (1 GPU each)"

while true; do
    sw_done=0; sw_total=0

    # ── 1. SPATIAL SWEEP (prioridade — só 48 jobs) ──────────────────────────
    for model in "${MODELS[@]}"; do
        local_venv="/data/wesleyferreiramaia/infoRates/.venv"
        [[ "$model" == "videomamba" ]] && local_venv="/data/wesleyferreiramaia/infoRates/.venv_mamba"
        for ds in "${DATASETS[@]}"; do
            ((sw_total++)) || true
            out_dir="${SPATIAL_OUT}/${model}_${ds}"
            summary="${out_dir}/spatial_sweep_summary.csv"
            if [[ -f "$summary" ]]; then ((sw_done++)); continue; fi

            submit_job \
                "sp-${model:0:5}-${ds:0:5}" \
                "$local_venv" \
                "python /data/wesleyferreiramaia/infoRates/scripts/accv2026/sweep_spatial_resolution.py \
                    --model ${model} --dataset ${ds} \
                    --output-dir /data/wesleyferreiramaia/infoRates/${out_dir}" \
                "${LOG_DIR}/spatial-${model}-${ds}-%j.out" \
                "${LOG_DIR}/spatial-${model}-${ds}-%j.err" || true
        done
    done

    # ── 2. COMBINED SWEEP (após spatial, 224 jobs) ───────────────────────────
    for model in "${MODELS[@]}"; do
        local_venv="/data/wesleyferreiramaia/infoRates/.venv"
        [[ "$model" == "videomamba" ]] && local_venv="/data/wesleyferreiramaia/infoRates/.venv_mamba"
        nat=${NATIVE_RES[$model]}
        for res in "${ALL_RES[@]}"; do
            [[ $res -eq $nat ]] && continue   # nativa já existe
            for ds in "${DATASETS[@]}"; do
                ((sw_total++)) || true
                out_dir="${COMBINED_OUT}/${model}_${ds}_res${res}"
                summary="${out_dir}/sweep_summary.csv"
                if [[ -f "$summary" ]]; then
                    lines=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$summary')))" 2>/dev/null || echo 0)
                    [[ "$lines" -ge 25 ]] && { ((sw_done++)); continue; }
                fi

                submit_job \
                    "cb-${model:0:4}-${ds:0:4}-${res}" \
                    "$local_venv" \
                    "python /data/wesleyferreiramaia/infoRates/scripts/accv2026/sweep_coverage_stride.py \
                        --model ${model} --dataset ${ds} --input-size ${res} \
                        --coverages 10 25 50 75 100 --strides 1 2 4 8 16 \
                        --output-dir /data/wesleyferreiramaia/infoRates/${out_dir}" \
                    "${LOG_DIR}/comb-${model}-${ds}-${res}px-%j.out" \
                    "${LOG_DIR}/comb-${model}-${ds}-${res}px-%j.err" || true
            done
        done
    done

    log "Progress: spatial=$(find $SPATIAL_OUT -name 'spatial_sweep_summary.csv' 2>/dev/null | wc -l)/56 | combined=$(find $COMBINED_OUT -name 'sweep_summary.csv' -path '*_res*' 2>/dev/null | wc -l)/224 | queue=$(n_sw)"

    all_sp=$(find $SPATIAL_OUT -name 'spatial_sweep_summary.csv' 2>/dev/null | wc -l)
    all_cb=$(find $COMBINED_OUT -name 'sweep_summary.csv' -path '*_res*' 2>/dev/null | wc -l)
    [[ $all_sp -ge 56 && $all_cb -ge 224 ]] && { log "=== TUDO COMPLETO ==="; exit 0; }

    sleep 60
done
