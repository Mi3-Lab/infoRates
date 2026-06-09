#!/usr/bin/env bash
# Roda spatial resolution sweep (eval-only, sem retraining) para todos modelos × datasets
# Gera Experimento 1 completo: mesmo modelo, resolução muda só na inferência
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
DATASETS=(ucf101 hmdb51 diving48 autsl driveact epic_kitchens)  # SSv2 já tem

LOG_DIR="evaluations/accv2026/logs"
SWEEP_DIR="evaluations/accv2026/spatial_resolution_sweep"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
n_jobs() { squeue -u wesleyferreiramaia --noheader 2>/dev/null | wc -l; }
MAX=5

log "=== Spatial Sweep — todos datasets exceto SSv2 (já feito) ==="
log "Total: ${#MODELS[@]} modelos × ${#DATASETS[@]} datasets = $(( ${#MODELS[@]} * ${#DATASETS[@]} )) sweeps"

for model in "${MODELS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        # Verificar se já existe resultado
        out_dir="${SWEEP_DIR}/${model}_${ds}"
        if [[ -f "${out_dir}/spatial_sweep_summary.csv" ]]; then
            log "[SKIP] ${model}/${ds} — já existe"
            continue
        fi

        # Aguardar slot
        while [[ $(n_jobs) -ge $MAX ]]; do sleep 30; done

        # Escolher venv
        if [[ "$model" == "videomamba" ]]; then
            venv="/data/wesleyferreiramaia/infoRates/.venv_mamba"
        else
            venv="/data/wesleyferreiramaia/infoRates/.venv"
        fi

        JID=$(sbatch \
            --job-name="spatial-sweep-${model}-${ds}" \
            --partition=cenvalarc.gpu \
            --gres=gpu:1 \
            --cpus-per-task=8 \
            --mem=64G \
            --time=02:00:00 \
            --output="${LOG_DIR}/spatial-sweep-${model}-${ds}-%j.out" \
            --error="${LOG_DIR}/spatial-sweep-${model}-${ds}-%j.err" \
            --wrap="
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME=/scratch/wesleyferreiramaia/infoRates/torch_cache
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_MODE=online
source ${venv}/bin/activate
mkdir -p ${SWEEP_DIR}/${model}_${ds}
python /data/wesleyferreiramaia/infoRates/scripts/accv2026/sweep_spatial_resolution.py \
    --model ${model} --dataset ${ds} \
    --output-dir ${SWEEP_DIR}/${model}_${ds}
" 2>&1 | grep -o "[0-9]*")

        log "[${model}/${ds}] Submitted job ${JID}"
        sleep 3
    done
done

log "=== Todos submetidos. Aguardando conclusão... ==="
until [[ $(n_jobs) -eq 0 ]]; do sleep 60; done
log "=== COMPLETO ==="
