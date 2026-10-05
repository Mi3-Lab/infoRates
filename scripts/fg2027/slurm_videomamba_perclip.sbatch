#!/usr/bin/env bash
#SBATCH --job-name=vm-perclip
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=/data/wesleyferreiramaia/infoRates/evaluations/accv2026/logs/%x-%j.err
# Re-run the VideoMamba coverage x stride grid keeping per-clip predictions.
# The original VideoMamba sweep used the decode-once fast path, which stores only
# per-config accuracy, so the repeated-measures ANOVA and the held-out cascade
# could not include it. --no-fast takes the per-config path that writes
# cov*_s*_samples.csv. Results go to a separate directory; nothing is overwritten.
#
#   DATASET=ssv2 sbatch scripts/fg2027/slurm_videomamba_perclip.sbatch
#   DATASET=autsl TRAIN_RES=224 sbatch ...   (AUTSL numbers come from the 224px run)
set -uo pipefail
REPO=/data/wesleyferreiramaia/infoRates
# Manifest video paths are relative to the scratch data root.
cd /scratch/wesleyferreiramaia/infoRates
source "${REPO}/.venv_mamba/bin/activate"
export PYTHONPATH="${REPO}/src"
export TORCH_HOME="${TORCH_HOME:-/scratch/wesleyferreiramaia/infoRates/torch_cache}"
export HF_HOME="${HF_HOME:-/scratch/wesleyferreiramaia/infoRates/hf_cache}"
export WANDB_MODE=disabled

DATASET="${DATASET:?DATASET must be set}"
EXTRA=""
[[ -n "${TRAIN_RES:-}" ]] && EXTRA="--train-res ${TRAIN_RES}"
OUT="${REPO}/evaluations/accv2026/coverage_stride_sweep_perclip/videomamba_${DATASET}"

echo "=== VideoMamba per-clip sweep — job ${SLURM_JOB_ID} — ${DATASET} ${EXTRA} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python "${REPO}/scripts/accv2026/sweep_coverage_stride.py" \
    --model videomamba --dataset "${DATASET}" --no-fast \
    --output-dir "${OUT}" ${EXTRA}
echo "=== DONE $(date) ==="
ls "${OUT}" | grep -c samples.csv | xargs echo "sample files:"
