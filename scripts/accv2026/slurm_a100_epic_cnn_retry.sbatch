#!/usr/bin/env bash
#SBATCH --job-name=accv26-epic-cnn-retry
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -uo pipefail
cd /data/wesleyferreiramaia/infoRates
source .venv/bin/activate

export PYTHONPATH=src
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="inforates-accv2026"
export ACCV_JOB_ID="${SLURM_JOB_ID}"

# Use fewer workers and lower prefetch to avoid DataLoader worker OOM on EPIC-Kitchens
# (high-res 1920x1080 clips, many participants, ~77k clips total)
export NUM_WORKERS=4
export DATASET=epic_kitchens
export EPOCHS=10

echo "=== EPIC-Kitchens CNN retry — job ${SLURM_JOB_ID} ==="
echo "Node: $(hostname) | $(date)"
nvidia-smi | head -6 || true

MANIFEST_DIR="evaluations/accv2026/manifests"
EVAL_MANIFEST="${MANIFEST_DIR}/epic_kitchens_val_20_per_class.csv"
mkdir -p evaluations/accv2026/fixed_budget fine_tuned_models evaluations/accv2026/logs "${MANIFEST_DIR}"

build_manifest_if_missing() {
    if [[ ! -f "${EVAL_MANIFEST}" ]]; then
        echo "[epic-retry] Building eval manifest"
        python - <<PYEOF
import sys; sys.path.insert(0, "src")
from info_rates.data.datasets import build_eval_manifest
df = build_eval_manifest("epic_kitchens", samples_per_class=20)
df.to_csv("${EVAL_MANIFEST}", index=False)
print(f"  Wrote {len(df)} rows to ${EVAL_MANIFEST}")
PYEOF
    fi
}

train_and_eval() {
    local model=$1
    local resume_from=$2
    local ckpt="fine_tuned_models/accv2026_${model}_epic_kitchens_full_e10_a100"
    local out_dir="evaluations/accv2026/fixed_budget/${model}_epic_kitchens_full_e10_a100"
    local ckpt_name="$(basename "${ckpt}")"
    local summary="${out_dir}/epic_kitchens_val_${ckpt_name}_fixed_budget_summary.csv"

    if [[ -f "${ckpt}/config.json" ]]; then
        echo "[epic-retry] Checkpoint exists: ${ckpt}"
    else
        echo "[epic-retry] Training ${model} on epic_kitchens"
        python scripts/accv2026/train_torchvision.py \
            --dataset epic_kitchens \
            --model "${model}" \
            --epochs "${EPOCHS}" \
            --batch-size 24 \
            --num-workers "${NUM_WORKERS}" \
            --save-path "${ckpt}" \
            ${resume_from:+--resume-from "${resume_from}"} \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-run-name "train-${model}-epic_kitchens-retry-job${ACCV_JOB_ID}" \
            --wandb-tags accv2026 a100 epic_kitchens "${model}" retry || \
            { echo "[epic-retry] WARN: ${model} training failed, continuing"; return; }
    fi

    if [[ ! -f "${summary}" ]]; then
        echo "[epic-retry] Fixed-budget eval: ${model}"
        build_manifest_if_missing
        python scripts/accv2026/eval_fixed_budget.py \
            --manifest "${EVAL_MANIFEST}" \
            --dataset-name epic_kitchens \
            --split val \
            --checkpoint "${ckpt}" \
            --budgets 4 8 16 32 \
            --model-frames 8 \
            --batch-size 24 \
            --resize 224 \
            --output-dir "${out_dir}" \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-run-name "eval-${model}-epic_kitchens-retry-job${ACCV_JOB_ID}" \
            --wandb-tags accv2026 a100 epic_kitchens "${model}" retry evaluation
    fi

    if [[ -f "${summary}" && ! -f "${out_dir}/temporal_metrics.csv" ]]; then
        python scripts/accv2026/05_compute_temporal_metrics.py \
            --summary "${summary}" --output "${out_dir}/temporal_metrics.csv"
    fi
}

train_and_eval r3d_18  "fine_tuned_models/accv2026_r3d_18_epic_kitchens_full_e10_a100_epoch4"
train_and_eval mc3_18  "fine_tuned_models/accv2026_mc3_18_epic_kitchens_full_e10_a100_epoch2"
train_and_eval r2plus1d_18  ""

echo "=== EPIC-Kitchens CNN retry COMPLETE — $(date) ==="
