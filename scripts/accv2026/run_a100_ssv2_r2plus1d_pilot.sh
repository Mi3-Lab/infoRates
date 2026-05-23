#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-r2plus1d_18}"
export MODEL_SLUG="${MODEL_SLUG:-r2plus1d-18}"
export TRAIN_TAG="${TRAIN_TAG:-r2plus1d-18-ssv2-5k-e1-a100}"
export CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_r2plus1d_18_ssv2_5k_e1_a100}"
export OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/r2plus1d_18_ssv2_5k_e1_a100}"
export TRAIN_GPU="${TRAIN_GPU:-0}"
export EVAL_GPU="${EVAL_GPU:-0}"

bash scripts/accv2026/run_a100_ssv2_torchvision_pilot.sh
