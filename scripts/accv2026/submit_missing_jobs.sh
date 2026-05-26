#!/usr/bin/env bash
# Submit missing per-dataset all-model jobs respecting QOS limits.
# GPU (A100): runs r3d_18 + mc3_18 + slowfast_r50 sequentially per dataset.
# H200:       runs timesformer + vivit sequentially per dataset.
# Idempotent: skips datasets where all models already have temporal_metrics.csv.
# Usage: bash scripts/accv2026/submit_missing_jobs.sh
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates

GPU_LIMIT=4
H200_LIMIT=4

DATASETS=(hmdb51 diving48 epic_kitchens autsl driveact)

# CNN models checkpoint paths (skip if temporal_metrics exists for all 3)
cnn_done() {
    local ds=$1
    local done=0
    for model in r3d_18 mc3_18 slowfast_r50; do
        local d="evaluations/accv2026/fixed_budget/${model}_${ds}_full_e10_a100"
        [[ -f "${d}/temporal_metrics.csv" ]] && done=$((done+1))
    done
    [[ $done -eq 3 ]]
}

transformer_done() {
    local ds=$1
    local done=0
    for model in timesformer vivit; do
        local d="evaluations/accv2026/fixed_budget/${model}_${ds}_full_e10_h200"
        [[ -f "${d}/temporal_metrics.csv" ]] && done=$((done+1))
    done
    [[ $done -eq 2 ]]
}

in_queue() {
    local pattern=$1
    squeue -u wesleyferreiramaia --format="%j" --noheader 2>/dev/null | grep -q "$pattern"
}


gpu_count=$(squeue -u wesleyferreiramaia -p gpu --format="%i" --noheader 2>/dev/null | wc -l || echo 0)
h200_count=$(squeue -u wesleyferreiramaia -p cenvalarc.gpu --format="%i" --noheader 2>/dev/null | wc -l || echo 0)
gpu_slots=$((GPU_LIMIT - gpu_count))
h200_slots=$((H200_LIMIT - h200_count))

echo "Queue: GPU=${gpu_count}/${GPU_LIMIT} (${gpu_slots} free)  H200=${h200_count}/${H200_LIMIT} (${h200_slots} free)"
echo ""

submitted=0
skipped=0

for ds in "${DATASETS[@]}"; do
    # A100 job
    if cnn_done "$ds"; then
        echo "  [DONE]  A100 allcnn @ ${ds}"
        skipped=$((skipped+1))
    elif in_queue "allcnn-${ds}"; then
        echo "  [QUEUE] A100 allcnn @ ${ds}"
        skipped=$((skipped+1))
    elif [[ $gpu_slots -le 0 ]]; then
        echo "  [FULL]  A100 allcnn @ ${ds} — no GPU slots"
    else
        jid=$(sbatch --parsable \
            --job-name="accv26-a100-allcnn-${ds}" \
            --partition=gpu --nodes=1 --gres=gpu:1 --cpus-per-task=8 \
            --mem=32G --time=2-00:00:00 \
            --output="evaluations/accv2026/logs/%x-%j.out" \
            --error="evaluations/accv2026/logs/%x-%j.err" \
            --wrap="set -uo pipefail; cd /data/wesleyferreiramaia/infoRates; [[ -f .venv/bin/activate ]] && source .venv/bin/activate; export DATASET=${ds} WANDB_PROJECT=inforates-accv2026 EPOCHS=10 NUM_WORKERS=4 HF_HOME=/scratch/wesleyferreiramaia/hf_unified TRANSFORMERS_CACHE=/scratch/wesleyferreiramaia/infoRates/hf_cache; srun --ntasks=1 --cpus-per-task=8 --gpus=1 bash scripts/accv2026/run_a100_dataset_all_cnn.sh")
        echo "  [SUBMIT] A100 $jid — allcnn @ ${ds}"
        gpu_slots=$((gpu_slots-1))
        submitted=$((submitted+1))
    fi

    # H200 job
    if transformer_done "$ds"; then
        echo "  [DONE]  H200 alltrans @ ${ds}"
        skipped=$((skipped+1))
    elif in_queue "alltrans-${ds}"; then
        echo "  [QUEUE] H200 alltrans @ ${ds}"
        skipped=$((skipped+1))
    elif [[ $h200_slots -le 0 ]]; then
        echo "  [FULL]  H200 alltrans @ ${ds} — no H200 slots"
    else
        jid=$(sbatch --parsable \
            --job-name="accv26-h200-alltrans-${ds}" \
            --partition=cenvalarc.gpu --nodes=1 --gres=gpu:1 --cpus-per-task=8 \
            --mem=40G --time=2-00:00:00 \
            --output="evaluations/accv2026/logs/%x-%j.out" \
            --error="evaluations/accv2026/logs/%x-%j.err" \
            --wrap="set -uo pipefail; cd /data/wesleyferreiramaia/infoRates; [[ -f .venv/bin/activate ]] && source .venv/bin/activate; export DATASET=${ds} WANDB_PROJECT=inforates-accv2026 EPOCHS=10 NUM_WORKERS=4 HF_HOME=/scratch/wesleyferreiramaia/hf_unified TRANSFORMERS_CACHE=/scratch/wesleyferreiramaia/infoRates/hf_cache; srun --ntasks=1 --cpus-per-task=8 --gpus=1 bash scripts/accv2026/run_h200_dataset_all_transformer.sh")
        echo "  [SUBMIT] H200 $jid — alltrans @ ${ds}"
        h200_slots=$((h200_slots-1))
        submitted=$((submitted+1))
    fi
    echo ""
done

echo "────────────────────────────────────"
echo "Submitted: ${submitted} | Already done/queued: ${skipped}"
[[ $submitted -gt 0 ]] || echo "Run again when jobs finish to fill new slots."
