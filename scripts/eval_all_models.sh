#!/bin/bash
# Evaluate all fine-tuned models with temporal sampling
# Results organized by model: results/{timesformer,videomae,vivit}/

set -e

RESULTS_DIR="UCF101_data/results"
MODELS_DIR="fine_tuned_models"

echo "========================================================================"
echo "Evaluating All Fine-Tuned Models"
echo "========================================================================"
echo "Results will be saved to:"
echo "  - $RESULTS_DIR/timesformer/"
echo "  - $RESULTS_DIR/videomae/"
echo "  - $RESULTS_DIR/vivit/"
echo ""

# Find all model directories
for model_dir in "$MODELS_DIR"/*/; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    model_name=$(basename "$model_dir")
    echo "→ Evaluating: $model_name"
    
    # Run evaluation with per-class analysis
    python scripts/run_eval.py \
        --model-path "$model_dir" \
        --per-class \
        --sample-size 200 \
        --batch-size 16 \
        --workers 8 \
        --coverages 10 25 50 75 100 \
        --strides 1 2 4 8 16
    
    echo "✅ $model_name evaluation complete"
    echo ""
done

echo "========================================================================"
echo "✅ All evaluations complete!"
echo "========================================================================"
echo "Results structure:"
ls -la "$RESULTS_DIR"
