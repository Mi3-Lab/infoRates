#!/bin/bash
# DDP-optimized evaluation launcher for all models
# Usage: bash scripts/eval_ddp_all.sh

set -e

MODELS_DIR="fine_tuned_models"
GPUS=${1:-2}

echo "========================================================================"
echo "DDP Evaluation of All Models (${GPUS} GPUs each)"
echo "========================================================================"

# Find all model directories
for model_dir in "$MODELS_DIR"/*/; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    model_name=$(basename "$model_dir")
    echo ""
    echo "→ Evaluating: $model_name on $GPUS GPUs"
    
    # Run DDP evaluation
    torchrun --standalone --nproc_per_node=$GPUS \
        scripts/run_eval.py \
        --model-path "$model_dir" \
        --per-class \
        --sample-size 100 \
        --batch-size 16 \
        --workers 2 \
        --coverages 10 25 50 75 100 \
        --strides 1 2 4 8 16 \
        --ddp
    
    echo "✅ $model_name evaluation complete"
done

echo ""
echo "========================================================================"
echo "✅ All DDP evaluations complete!"
echo "========================================================================"
