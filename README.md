# InfoRates: Temporal Sampling for Action Recognition

This repository explores how temporal sampling (coverage and stride) affects action recognition across modern video models.

Quick start: see START_HERE.txt for commands, or the full docs/UNIFIED_GUIDE.md for end‑to‑end docs.

Key entry points
- Training (multi-model, DDP-ready): scripts/data_processing/train_ddp.sh → launches scripts/data_processing/train_multimodel.py
- Evaluation (multi-model): scripts/evaluation/run_eval_multimodel.py
- Plotting (all analysis plots): scripts/plotting/generate_analysis_plots.py --model MODEL --dataset DATASET
- Legacy DDP eval of a saved model: scripts/evaluation/run_eval.py and scripts/data_processing/pipeline_eval.sh

Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Examples
```bash
# 2 GPUs, fine-tune VideoMAE
bash scripts/data_processing/train_ddp.sh --model videomae --gpus 2 --epochs 5

# Evaluate all models with temporal sampling
python scripts/evaluation/run_eval_multimodel.py --model all --batch-size 16

# Generate all analysis plots for VideoMAE on UCF101
python scripts/plotting/generate_analysis_plots.py --model videomae --dataset ucf101

# Generate plots for all models on Kinetics400
python scripts/plotting/generate_analysis_plots.py --model all --dataset kinetics400
```

More details: docs/UNIFIED_GUIDE.md

## Repository Structure

- `data/`: Raw datasets (UCF101, Kinetics400, HMDB51)
- `evaluations/`: Model evaluation results and plots
  - `kinetics400/`: Kinetics400 results by model
  - `ucf101/`: UCF101 results by model
- `scripts/`: Utility scripts
  - `data_processing/`: Data preparation, training scripts
  - `evaluation/`: Evaluation and testing scripts
  - `plotting/`: Plotting and statistical analysis scripts
- `src/`: Source code (models, analysis modules)
- `docs/`: Documentation and evaluation reports
- `fine_tuned_models/`: Saved model checkpoints
- `models/`: Pre-trained models

## Datasets

The repository supports multiple video action recognition datasets:

- **UCF101**: Main dataset for temporal sampling analysis
- **Kinetics400**: Additional evaluation dataset  
- **HMDB51**: Additional evaluation dataset
- **Something-Something V2**: Planned for future analysis

To download datasets:
- UCF101: See `data/UCF101_data/README.md`
- HMDB51: `scripts/data_processing/download_hmdb51.sh`
- Kinetics400 subsets: `scripts/data_processing/download_kinetics_mini.sh` or `download_kinetics50_subset.sh`
- Something-Something V2: `scripts/data_processing/download_ssv2.sh` (Note: Large dataset ~100GB)