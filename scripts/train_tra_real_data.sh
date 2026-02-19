#!/bin/bash
# 🚀 Treinar com TRA - Dados Reais UCF101
# GPU: H200 (já alocada)
# Logging: WandB
#
# Execute este script para treinar baseline e TRA com dados reais

cd /home/wesleyferreiramaia/data/infoRates
source .venv/bin/activate

# Configuração
MODEL="timesformer"      # Opções: timesformer, videomae, vivit
EPOCHS=5
BATCH_SIZE=16            # H200 tem 141GB VRAM, pode usar batch maior
LR=2e-5
NUM_WORKERS=8            # H200 tem muitos cores, paralelize I/O

echo "=========================================="
echo "🎯 Treinamento TRA - UCF101 Real Dataset"
echo "=========================================="
echo "GPU: H200 (141GB VRAM)"
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Dataset: UCF101 (9537 train, 3783 test)"
echo "=========================================="
echo ""

# PASSO 1: Treinar BASELINE (sem TRA)
echo "📌 [1/3] Treinando BASELINE (sem TRA)..."
echo "Tempo estimado: ~1.5h na H200"
echo ""

python scripts/train_with_tra.py \
  --model $MODEL \
  --tra-mode baseline \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --num-workers $NUM_WORKERS \
  --save-dir fine_tuned_models/tra_experiments \
  --eval-robustness \
  --wandb

if [ $? -ne 0 ]; then
    echo "❌ Erro no treinamento baseline!"
    exit 1
fi

echo ""
echo "✅ Baseline training complete!"
echo ""

# PASSO 2: Treinar com TRA
echo "📌 [2/3] Treinando com TRA..."
echo "Tempo estimado: ~1.7h na H200 (um pouco mais lento devido à augmentação)"
echo ""

python scripts/train_with_tra.py \
  --model $MODEL \
  --tra-mode tra \
  --p-augment 0.5 \
  --coverage-range 25 50 75 100 \
  --stride-range 1 2 3 \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --num-workers $NUM_WORKERS \
  --save-dir fine_tuned_models/tra_experiments \
  --eval-robustness \
  --wandb

if [ $? -ne 0 ]; then
    echo "❌ Erro no treinamento TRA!"
    exit 1
fi

echo ""
echo "✅ TRA training complete!"
echo ""

# PASSO 3: Comparar Resultados
echo "📌 [3/3] Comparando Baseline vs TRA..."
echo ""

python scripts/compare_baseline_vs_tra.py \
  --model $MODEL \
  --plot \
  --save-latex

echo ""
echo "=========================================="
echo "🎉 EXPERIMENTOS COMPLETOS!"
echo "=========================================="
echo ""
echo "📊 Resultados salvos em:"
echo "  - Modelos:"
echo "    • fine_tuned_models/tra_experiments/baseline/$MODEL/"
echo "    • fine_tuned_models/tra_experiments/tra/$MODEL/"
echo ""
echo "  - Robustness JSON:"
echo "    • fine_tuned_models/tra_experiments/baseline/robustness_$MODEL.json"
echo "    • fine_tuned_models/tra_experiments/tra/robustness_$MODEL.json"
echo ""
echo "  - Visualizações:"
echo "    • docs/figures/tra_comparison_$MODEL.png"
echo "    • docs/tables/tra_comparison_$MODEL.tex"
echo ""
echo "  - WandB Logs:"
echo "    • https://wandb.ai/YOUR_USERNAME/infoRates"
echo ""
echo "=========================================="
echo "📈 Próximos passos:"
echo "  1. Abrir comparison plot:"
echo "     open docs/figures/tra_comparison_$MODEL.png"
echo ""
echo "  2. Ver métricas no WandB dashboard"
echo ""
echo "  3. Adicionar tabela LaTeX ao paper:"
echo "     \\input{tables/tra_comparison_$MODEL.tex}"
echo "=========================================="
