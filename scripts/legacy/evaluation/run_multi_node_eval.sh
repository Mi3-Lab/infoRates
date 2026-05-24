#!/bin/bash
# Script para executar avaliação multi-node usando srun diretamente

# Configurar variáveis de ambiente
export MASTER_PORT=12345
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ativar ambiente virtual
source /data/wesleyferreiramaia/infoRates/.venv/bin/activate

echo "Executando avaliação multi-node com 4 GPUs (2 nós x 2 GPUs)"

# Usar srun para lançar o job multi-node
# Usando nós disponíveis: gnode001 e gnode008 (ajuste conforme disponibilidade)
srun --partition=gpu --nodelist=gnode001,gnode008 --nodes=2 --ntasks-per-node=2 --gres=gpu:a100:2 \
    torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    scripts/run_eval.py \
    --config config.yaml \
    --dataset kinetics400 \
    --model-path facebook/timesformer-base-finetuned-k400 \
    --sample-size 0 \
    --batch-size 38 \
    --workers 4 \
    --ddp \
    --per-class \
    --per-class-sample-size -1 \
    --jitter-coverage-pct 0.0 \
    --wandb-project inforates-kinetics400