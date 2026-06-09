#!/usr/bin/env bash
# Combined sweep: coverage × stride × TODAS as resoluções
# Para cada modelo/dataset, roda o temporal sweep (5 strides × 5 coverages)
# em TODAS as resoluções não-nativas.
# A resolução nativa já existe em dashboard/data/sweep_summary.csv
#
# Total: 8 modelos × 4 res-extras × 7 datasets = 224 jobs
# Cada job: 25 configs (eval-only, ~45-90 min)
# ~2-3 dias com MAX=4 concurrent

set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

# Resoluções nativas por modelo — NÃO incluir (já temos no temporal sweep)
declare -A NATIVE_RES=(
    [r3d_18]=112 [mc3_18]=112 [r2plus1d_18]=112 [slowfast_r50]=224
    [timesformer]=224 [vivit]=224 [videomae]=224 [videomamba]=224
)

ALL_RES=(96 112 160 224 336)
MODELS=(r3d_18 mc3_18 r2plus1d_18 slowfast_r50 timesformer vivit videomae videomamba)
DATASETS=(ucf101 ssv2 hmdb51 diving48 autsl driveact epic_kitchens)

# Todos os strides e coverages (iguais ao temporal sweep original)
COVERAGES="10 25 50 75 100"
STRIDES="1 2 4 8 16"

LOG_DIR="evaluations/accv2026/logs"
OUT_BASE="evaluations/accv2026/coverage_stride_sweep"
MAX=4

log() { echo "[$(date '+%H:%M:%S')] COMB-SWEEP | $*"; }
n_jobs() { squeue -u wesleyferreiramaia --noheader 2>/dev/null | wc -l; }

# Contar total de jobs
total=0
for model in "${MODELS[@]}"; do
    nat=${NATIVE_RES[$model]}
    for res in "${ALL_RES[@]}"; do
        [[ $res -eq $nat ]] && continue
        total=$(( total + ${#DATASETS[@]} ))
    done
done

log "=== Combined Sweep: coverage × stride × resolução ==="
log "Total: ${total} jobs (8 modelos × 4 res-extras × 7 datasets)"
log "Configs por job: 5 strides × 5 coverages = 25"
log "MAX concurrent: ${MAX} | Estimativa: ~$((total * 70 / MAX / 60))h"

submitted=0
skipped=0

for model in "${MODELS[@]}"; do
    nat=${NATIVE_RES[$model]}
    venv="/data/wesleyferreiramaia/infoRates/.venv"
    [[ "$model" == "videomamba" ]] && venv="/data/wesleyferreiramaia/infoRates/.venv_mamba"

    for res in "${ALL_RES[@]}"; do
        [[ $res -eq $nat ]] && continue  # nativa já existe

        for ds in "${DATASETS[@]}"; do
            out_dir="${OUT_BASE}/${model}_${ds}_res${res}"
            summary="${out_dir}/sweep_summary.csv"

            if [[ -f "$summary" ]]; then
                # Verificar se está completo (25 configs)
                n=$(python3 -c "import pandas as pd; df=pd.read_csv('$summary'); print(len(df))" 2>/dev/null || echo 0)
                if [[ "$n" -ge 25 ]]; then
                    ((skipped++)) || true
                    continue
                fi
            fi

            # Aguardar slot
            while [[ $(n_jobs) -ge $MAX ]]; do sleep 30; done

            JID=$(sbatch \
                --job-name="comb-${model:0:6}-${ds:0:5}-${res}" \
                --partition=cenvalarc.gpu \
                --gres=gpu:1 \
                --cpus-per-task=8 \
                --mem=64G \
                --time=03:00:00 \
                --output="${LOG_DIR}/comb-sweep-${model}-${ds}-${res}px-%j.out" \
                --error="${LOG_DIR}/comb-sweep-${model}-${ds}-${res}px-%j.err" \
                --wrap="
set -e
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME=/scratch/wesleyferreiramaia/infoRates/torch_cache
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_MODE=disabled
source ${venv}/bin/activate
echo '=== Combined Sweep: ${model} / ${ds} @ ${res}px ==='
nvidia-smi | grep -E 'Name|L40|H200|A100' | head -2 || true
python /data/wesleyferreiramaia/infoRates/scripts/accv2026/sweep_coverage_stride.py \
    --model ${model} \
    --dataset ${ds} \
    --input-size ${res} \
    --coverages ${COVERAGES} \
    --strides ${STRIDES} \
    --output-dir /data/wesleyferreiramaia/infoRates/${out_dir}
echo DONE
" 2>&1 | grep -o "[0-9]*")

            if [[ -n "$JID" && "$JID" =~ ^[0-9]+$ ]]; then
                log "[${model}/${ds}@${res}px] Job ${JID}"
                ((submitted++)) || true
            else
                log "[${model}/${ds}@${res}px] FALHOU — QOS limit, tentando de novo em 60s"
                sleep 60
                # Re-tentar uma vez
                JID=$(sbatch \
                    --job-name="comb-${model:0:6}-${ds:0:5}-${res}" \
                    --partition=cenvalarc.gpu \
                    --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 \
                    --output="${LOG_DIR}/comb-sweep-${model}-${ds}-${res}px-%j.out" \
                    --error="${LOG_DIR}/comb-sweep-${model}-${ds}-${res}px-%j.err" \
                    --wrap="
cd /scratch/wesleyferreiramaia/infoRates
export PYTHONPATH=/data/wesleyferreiramaia/infoRates/src
export TORCH_HOME=/scratch/wesleyferreiramaia/infoRates/torch_cache
export HF_HOME=/scratch/wesleyferreiramaia/hf_unified
export WANDB_MODE=disabled
source ${venv}/bin/activate
python /data/wesleyferreiramaia/infoRates/scripts/accv2026/sweep_coverage_stride.py \
    --model ${model} --dataset ${ds} --input-size ${res} \
    --coverages ${COVERAGES} --strides ${STRIDES} \
    --output-dir /data/wesleyferreiramaia/infoRates/${out_dir}
" 2>&1 | grep -o "[0-9]*")
                [[ -n "$JID" ]] && log "[RETRY OK] Job ${JID}" || log "[RETRY FALHOU] Continuando..."
            fi

            sleep 3
        done
    done
done

log "=== Submissão concluída: ${submitted} novos jobs, ${skipped} já existentes ==="
log "Aguardando término..."
until [[ $(n_jobs) -eq 0 ]]; do
    done_count=$(find "${OUT_BASE}" -name "sweep_summary.csv" -newer "${OUT_BASE}" 2>/dev/null | wc -l || echo 0)
    log "Progress: ${done_count}/${total} | jobs na fila: $(n_jobs)"
    sleep 300
done
log "=== TUDO COMPLETO ==="
