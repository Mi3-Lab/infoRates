#!/usr/bin/env bash
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=vmamba3-validate
#SBATCH --output=evaluations/accv2026/logs/vmamba3-validate-%j.out
#SBATCH --error=evaluations/accv2026/logs/vmamba3-validate-%j.err
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
mkdir -p evaluations/accv2026/logs evaluations/accv2026/videomamba3
source .venv_mamba/bin/activate
export PYTHONPATH=src

nvidia-smi || true

python -u scripts/accv2026/validate_videomamba3_system.py \
  --variants ${VARIANTS:-complex trapezoidal} \
  --mamba3-impl "${MAMBA3_IMPL:-auto}" \
  --model-size "${MODEL_SIZE:-tiny}" \
  --depths ${DEPTHS:-4 8 12 24} \
  --modes ${MODES:-inference train} \
  --num-frames "${NUM_FRAMES:-2}" \
  --input-size "${INPUT_SIZE:-112}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --ssm-d-state "${SSM_D_STATE:-16}" \
  --ssm-expand "${SSM_EXPAND:-1}" \
  --ssm-headdim "${SSM_HEADDIM:-32}" \
  --ssm-mimo-rank "${SSM_MIMO_RANK:-2}" \
  --warmup "${WARMUP:-2}" \
  --iters "${ITERS:-5}" \
  --output "${OUTPUT:-evaluations/accv2026/videomamba3/system_validation_h200.csv}"

python scripts/accv2026/18_compile_videomamba3_results.py
