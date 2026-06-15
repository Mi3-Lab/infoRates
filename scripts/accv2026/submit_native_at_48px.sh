#!/usr/bin/env bash
# Submete inferência dos modelos na resolução nativa avaliados com input 48px
# Usa checkpoints nativos (112px CNNs, 224px Transformers) com --input-size 48
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

SBATCH_SCRIPT="scripts/accv2026/slurm_native_at_48px.sbatch"
SWEEP_ROOT="evaluations/accv2026/coverage_stride_sweep"
PARTITIONS=(cenvalarc.gpu gpu)
QOS_PER_PART=4
LOG="evaluations/accv2026/logs/submit_native_48px.log"

MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

sweep_done() {
    local model=$1 dataset=$2
    local dir="${SWEEP_ROOT}/${model}_${dataset}_res48"
    [[ -f "${dir}/sweep_summary.csv" ]] || return 1
    local n; n=$(wc -l < "${dir}/sweep_summary.csv" 2>/dev/null || echo 0)
    [[ $n -ge 26 ]]
}

count_partition_jobs() {
    squeue -u wesleyferreiramaia -p "$1" --noheader --format="%.10i" 2>/dev/null | wc -l
}

mkdir -p evaluations/accv2026/logs
log "=== Submissão: native checkpoint @ 48px input ==="

while true; do
    all_done=true
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            sweep_done "$model" "$dataset" && continue
            all_done=false

            # Encontrar partição com slot livre
            target_part=""
            for part in "${PARTITIONS[@]}"; do
                n=$(count_partition_jobs "$part")
                if [[ $n -lt $QOS_PER_PART ]]; then
                    target_part="$part"
                    break
                fi
            done
            [[ -z "$target_part" ]] && continue

            mkdir -p "${SWEEP_ROOT}/${model}_${dataset}_res48"
            jid=$(sbatch \
                --partition="$target_part" \
                --output="evaluations/accv2026/logs/native48-${model}-${dataset}-%j.out" \
                --error="evaluations/accv2026/logs/native48-${model}-${dataset}-%j.err" \
                --export="ALL,MODEL=${model},DATASET=${dataset}" \
                "$SBATCH_SCRIPT" 2>&1 | awk '{print $NF}')
            if [[ "$jid" =~ ^[0-9]+$ ]]; then
                log "  Submitted ${model}/${dataset} → job ${jid} [${target_part}]"
            else
                log "  [WARN] ${model}/${dataset}: $jid"
            fi
            sleep 2
        done
    done

    $all_done && { log "=== TUDO CONCLUÍDO ==="; exit 0; }
    sleep 120
done
