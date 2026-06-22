#!/usr/bin/env bash
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=vmamba3-bench
#SBATCH --output=evaluations/accv2026/logs/vmamba3-bench-%j.out
#SBATCH --error=evaluations/accv2026/logs/vmamba3-bench-%j.err
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
source .venv_mamba/bin/activate
export PYTHONPATH=src

python scripts/accv2026/benchmark_videomamba3.py \
  --mamba3-impl "${MAMBA3_IMPL:-auto}" \
  --model-size "${MODEL_SIZE:-tiny}" \
  --depth "${DEPTH:-8}" \
  --num-frames "${NUM_FRAMES:-2}" \
  --input-size "${INPUT_SIZE:-112}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --ssm-d-state "${SSM_D_STATE:-16}" \
  --ssm-expand "${SSM_EXPAND:-1}" \
  --ssm-headdim "${SSM_HEADDIM:-32}" \
  --ssm-mimo-rank "${SSM_MIMO_RANK:-2}" \
  --warmup "${WARMUP:-1}" \
  --iters "${ITERS:-3}" \
  --output "${OUTPUT:-evaluations/accv2026/videomamba3/latency_benchmark.csv}"

python scripts/accv2026/18_compile_videomamba3_results.py
