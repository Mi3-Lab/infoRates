#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-r3d_18}"
export MODEL_SLUG="${MODEL_SLUG:-r3d-18}"
export TRAIN_TAG="${TRAIN_TAG:-r3d-18-ssv2-full-e5-a100}"
export CHECKPOINT="${CHECKPOINT:-fine_tuned_models/accv2026_r3d18_ssv2_full_e5_a100}"
export OUT_DIR="${OUT_DIR:-evaluations/accv2026/fixed_budget/r3d18_ssv2_full_e5_a100}"
export EPOCHS="${EPOCHS:-5}"

bash scripts/accv2026/run_a100_ssv2_torchvision_full.sh
