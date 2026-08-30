#!/usr/bin/env bash
#SBATCH --job-name=rebuttal-sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err
set -uo pipefail

# Manifest video_path columns are relative to the scratch data root, so the job
# must run from there; code and outputs stay on /data.
REPO=/data/wesleyferreiramaia/infoRates
cd /scratch/wesleyferreiramaia/infoRates

# VideoMamba needs .venv_mamba (mamba_ssm + causal_conv1d). Note that its
# import fails on the login node and succeeds on a GPU node: the CUDA runtime
# libraries it links against are present only on the compute nodes.
if [[ "${MODELS}" == *videomamba* ]]; then
    source "${REPO}/.venv_mamba/bin/activate"
else
    source "${REPO}/.venv/bin/activate"
fi
export PYTHONPATH="${REPO}/src"
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export HF_HOME="${HF_HOME:-/scratch/wesleyferreiramaia/infoRates/hf_cache}"
export WANDB_MODE=disabled   # inference only; one run per config would be noise

MODE="${MODE:?MODE must be set (matched|span|covstride)}"
MODELS="${MODELS:?MODELS must be set}"
DATASETS="${DATASETS:?DATASETS must be set}"
LIMIT="${LIMIT:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# Passed through verbatim: used by the TRA evaluation to supply --checkpoint and
# --tag, which point the sweep at a non-standard checkpoint and keep its results
# in a separate output directory.
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=== Rebuttal sweep — job ${SLURM_JOB_ID} — mode=${MODE} ==="
echo "Models:   ${MODELS}"
echo "Datasets: ${DATASETS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for model in ${MODELS}; do
    for dataset in ${DATASETS}; do
        echo "--- ${model} / ${dataset} (${MODE}) ---"
        python "${REPO}/scripts/accv2026/rebuttal_matched_evidence_sweep.py" \
            --model "${model}" \
            --dataset "${dataset}" \
            --mode "${MODE}" \
            --batch-size "${BATCH_SIZE}" \
            --limit "${LIMIT}" \
            ${EXTRA_ARGS} \
            || echo "  [FAIL] ${model}/${dataset}"
    done
done

echo "=== DONE $(date) ==="
find "${REPO}/evaluations/accv2026/rebuttal_sweeps/${MODE}" -name summary.csv 2>/dev/null \
    | wc -l | xargs echo "summary.csv files for mode=${MODE}:"
