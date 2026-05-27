#!/usr/bin/env bash
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=evaluations/accv2026/logs/vmamba-eval-%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/vmamba-eval-%x-%j.err
set -euo pipefail
cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs

source .venv_mamba/bin/activate
pip install decord --quiet 2>/dev/null || true
export PYTHONPATH=src
export WANDB_MODE=online
export ACCV_JOB_ID="${SLURM_JOB_ID}"

DATASET="${DATASET:?DATASET must be set}"
CHECKPOINT="fine_tuned_models/accv2026_videomamba_${DATASET}_full_e10_h200"
OUT_DIR="evaluations/accv2026/fixed_budget/videomamba_${DATASET}_full_e10_h200"
MANIFEST_PREFIX="${DATASET}"
[[ "${DATASET}" == "ssv2" ]] && MANIFEST_PREFIX="somethingv2"
EVAL_MANIFEST="evaluations/accv2026/manifests/${MANIFEST_PREFIX}_val_20_per_class.csv"

echo "=== VideoMamba eval: ${DATASET} job ${SLURM_JOB_ID} ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Manifest: ${EVAL_MANIFEST} ($(wc -l < ${EVAL_MANIFEST}) rows)"
nvidia-smi | head -6 || true

python scripts/accv2026/eval_fixed_budget.py \
    --manifest "${EVAL_MANIFEST}" \
    --dataset-name "${DATASET}" \
    --split val \
    --checkpoint "${CHECKPOINT}" \
    --budgets 4 8 16 32 \
    --model-frames 8 \
    --batch-size 16 \
    --resize 224 \
    --output-dir "${OUT_DIR}" \
    --wandb-project inforates-accv2026 \
    --wandb-run-name "eval-videomamba-${DATASET}-e10-job${SLURM_JOB_ID}" \
    --wandb-tags accv2026 h200 "${DATASET}" videomamba ssm full evaluation "job-${SLURM_JOB_ID}"

SUMMARY=$(ls "${OUT_DIR}/${DATASET}_val_"*"_fixed_budget_summary.csv" 2>/dev/null | tail -1)
if [[ -f "${SUMMARY}" ]]; then
    python scripts/accv2026/05_compute_temporal_metrics.py \
        --summary "${SUMMARY}" \
        --output "${OUT_DIR}/temporal_metrics.csv"
    echo "=== ${DATASET} eval DONE ==="
    cat "${SUMMARY}"
fi
