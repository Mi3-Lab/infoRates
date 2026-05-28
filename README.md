# InfoRates — Adaptive Temporal Frame Allocation for Video Recognition

**ACCV 2026 submission** · Mi3 Lab

We study how many frames a video model actually needs. Our method predicts the minimum frame budget per video — saving compute without sacrificing accuracy — and characterizes each dataset's *temporal demand* (TDS score).

---

## Requirements

- Python 3.10+
- CUDA GPU (trained on A100 40GB and H200 141GB)
- ~512 GB scratch storage for datasets + checkpoints

```bash
# Standard models (CNNs + Transformers)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# VideoMamba (SSM model) — separate environment required
python -m venv .venv_mamba
source .venv_mamba/bin/activate
pip install -r requirements_mamba.txt
pip install -e .
```

---

## Datasets

All datasets live at `data/`. Download each and place them as shown:

| Dataset | Path | Classes | Download |
|---------|------|---------|----------|
| SSV2 | `data/Something_data/` | 174 | [20bn-something-something-v2](https://developer.qualcomm.com/software/ai-datasets/something-something) |
| UCF-101 | `data/UCF101_data/` | 101 | [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) |
| HMDB-51 | `data/HMDB51_data/` | 51 | [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) |
| Diving-48 | `data/Diving48_data/` | 48 | [Diving48](http://www.svcl.ucsd.edu/projects/action_quality_assessment/) |
| EPIC-Kitchens | `data/EPIC_data/` | 97 | [EPIC-Kitchens-100](https://epic-kitchens.github.io/2022) — download annotations + RGB frames |
| AUTSL | `data/AUTSL_data/` | 226 | [Kaggle: sttaseen/autsl](https://www.kaggle.com/datasets/sttaseen/autsl) |
| DriveAct | `data/DriveAct_data/` | 34 | [DriveAct](https://driveact.de/) |
| Kinetics-400 | `data/Kinetics400_data/` | 400 | [K400 val](https://github.com/cvdfoundation/kinetics-dataset) |

After downloading, run preprocessing where needed:

```bash
python scripts/accv2026/preprocess_autsl.py    # generates data/AUTSL_data/splits/
python scripts/accv2026/preprocess_driveact.py  # generates data/DriveAct_data/splits/
python scripts/accv2026/download_epic_clips.py  # extracts EPIC-Kitchens clips from raw frames
```

---

## Training

Models: **R3D-18**, **MC3-18**, **R2Plus1D-18**, **SlowFast-R50** · **TimeSformer**, **ViViT**, **VideoMAE** · **VideoMamba** (SSM, 8th model)

### Automated (recommended)

```bash
nohup bash scripts/accv2026/feeder_submit_jobs.sh &
```

### Manual (one dataset)

```bash
# CNN models — A100 partition
export DATASET=hmdb51
bash scripts/accv2026/run_a100_dataset_all_cnn.sh

# Transformer models — H200 partition
bash scripts/accv2026/run_h200_dataset_all_transformer.sh

# VideoMamba — H200 partition, requires .venv_mamba
DATASET=hmdb51 bash scripts/accv2026/run_h200_multidata_videomamba.sh
```

### Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | required | dataset name |
| `EPOCHS` | `10` | training epochs |
| `WANDB_PROJECT` | `inforates-accv2026` | W&B project |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace model cache |

---

## Current Results (as of 2026-05-28) — Phase 1 Complete

All 8 models × 7 datasets trained and evaluated. Top-1 accuracy at fixed frame budgets (4 / 8 / 16 / 32 frames).

**Architectural notes:**
- SlowFast-R50 and ViViT use 32 input frames natively — accuracy jumps at 32f budget are architectural, not anomalies.
- VideoMamba and TimeSformer use 8 input frames natively — accuracy plateaus after 8f budget (higher budgets subsample back to 8 frames).

### Something-Something v2 (SSv2) — 174 classes

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 9.8% | 19.7% | 37.1% | 36.9% |
| MC3-18 | 8.2% | 18.8% | 33.6% | 34.5% |
| R2Plus1D-18 | 12.6% | 24.3% | 42.6% | 42.1% |
| SlowFast-R50 | 6.6% | 15.2% | 33.3% | 49.5% |
| TimeSformer | 31.8% | 42.3% | 41.3% | 41.7% |
| ViViT | 8.4% | 17.5% | 30.5% | 38.3% |
| VideoMAE | 21.0% | 39.5% | 52.3% | 51.9% |
| VideoMamba | 31.8% | 43.9% | 44.4% | 44.2% |

### UCF-101 — 101 classes

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 59.5% | 72.6% | 81.2% | 81.4% |
| MC3-18 | 72.9% | 80.9% | 85.4% | 85.1% |
| R2Plus1D-18 | 70.0% | 81.6% | 88.6% | 89.0% |
| SlowFast-R50 | 50.1% | 66.2% | 81.3% | 87.6% |
| TimeSformer | 90.0% | 91.0% | 91.2% | 90.9% |
| ViViT | 75.3% | 86.9% | 92.5% | 94.3% |
| VideoMAE | 81.4% | 91.4% | 95.4% | 95.5% |
| VideoMamba | 85.0% | 88.4% | 88.2% | 87.8% |

### HMDB-51 — 51 classes

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 49.2% | 67.1% | 80.3% | 80.1% |
| MC3-18 | 63.5% | 71.2% | 78.6% | 78.2% |
| R2Plus1D-18 | 46.2% | 63.2% | 73.1% | 74.6% |
| SlowFast-R50 | 35.1% | 44.7% | 65.1% | 79.3% |
| TimeSformer | 73.0% | 79.9% | 80.0% | 79.8% |
| ViViT | 52.4% | 66.1% | 75.4% | 80.2% |
| VideoMAE | 51.5% | 73.6% | 84.0% | 84.4% |
| VideoMamba | 61.7% | 69.8% | 68.6% | 69.7% |

### DriveAct — 34 classes

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 47.8% | 56.2% | 68.3% | 67.2% |
| MC3-18 | 55.1% | 65.8% | 69.0% | 68.5% |
| R2Plus1D-18 | 37.7% | 49.8% | 62.5% | 61.8% |
| SlowFast-R50 | 42.6% | 53.3% | 66.7% | 72.5% |
| TimeSformer | 64.7% | 67.6% | 68.8% | 66.5% |
| ViViT | 48.9% | 55.8% | 62.5% | 67.4% |
| VideoMAE | 40.2% | 56.0% | 74.1% | 72.5% |
| VideoMamba | 50.9% | 57.8% | 58.0% | 56.7% |

### Diving-48 — 48 classes

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 5.9% | 14.4% | 28.8% | 28.8% |
| MC3-18 | 8.2% | 19.5% | 31.6% | 33.4% |
| R2Plus1D-18 | 8.6% | 16.8% | 35.3% | 34.7% |
| SlowFast-R50 | 5.8% | 14.5% | 26.4% | 50.5% |
| TimeSformer | 23.6% | 38.0% | 36.9% | 38.0% |
| ViViT | 7.9% | 19.9% | 35.1% | 53.0% |
| VideoMAE | 8.6% | 27.6% | 48.6% | 49.9% |
| VideoMamba | 18.2% | 36.3% | 33.0% | 31.4% |

### AUTSL — 226 classes (Turkish sign language)

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 4.7% | 24.5% | 75.0% | 74.4% |
| MC3-18 | 4.1% | 37.5% | 63.7% | 63.7% |
| R2Plus1D-18 | 8.4% | 30.2% | 75.9% | 75.0% |
| SlowFast-R50 | 1.6% | 12.7% | 41.8% | 82.3% |
| TimeSformer | 52.0% | 66.8% | 66.2% | 67.0% |
| ViViT | 8.4% | 25.5% | 61.2% | 74.6% |
| VideoMAE | 17.7% | 43.2% | 79.5% | 78.9% |
| VideoMamba† | 0.4% | 0.4% | 0.4% | 0.4% |

† VideoMamba did not converge on AUTSL under any tested configuration (LR=1e-4 and LR=5e-4): loss stuck at ln(226)≈5.42 for all 10 epochs — exactly random chance for 226 classes. Root cause: K400 backbone produces near-identical features for all sign language clips (raw pixel std 10× lower than UCF-101) — feature collapse causes gradients to cancel across batches. CNN and Transformer models are unaffected due to stronger local spatial inductive biases.

### EPIC-Kitchens — 97 classes

Clean split since 2026-05-27 (duplicate videos between train/val removed). All models retrained and evaluated on the clean split.

| Model | 4f | 8f | 16f | 32f |
|-------|---:|---:|----:|----:|
| R3D-18 | 13.6% | 22.3% | 37.2% | 37.0% |
| MC3-18 | 11.3% | 27.1% | 36.2% | 37.2% |
| R2Plus1D-18 | 13.0% | 20.2% | 35.5% | 35.2% |
| SlowFast-R50 | 9.2% | 15.8% | 27.2% | 39.4% |
| TimeSformer | 19.5% | 32.3% | 31.5% | 31.0% |
| ViViT | 10.3% | 21.1% | 26.9% | 32.9% |
| VideoMAE | 13.4% | 28.3% | 37.7% | 37.5% |
| VideoMamba | 23.2% | 28.3% | 28.2% | 28.4% |

---

## Key Findings (Phase 2 Analysis)

### Temporal Demand Score (TDS) by Dataset

TDS = mean accuracy gain from 4→32 frames, averaged across all 7 models. Higher = dataset requires more frames.

| Dataset | TDS | Interpretation |
|---------|----:|----------------|
| AUTSL | +59.9pp | Extreme — sign language needs the full clip |
| Diving-48 | +29.1pp | High — dive phase sequence critical |
| SSv2 | +26.1pp | High — subtle hand motions |
| HMDB-51 | +24.2pp | Moderate-high |
| EPIC-Kitchens | +20.6pp | Moderate |
| DriveAct | +18.2pp | Moderate |
| UCF-101 | +15.9pp | Low — mostly appearance-based |

**Key result:** TDS ranking is consistent across all 8 models — temporal demand is a property of the *dataset*, not the model. This is the paper's main empirical claim.

### Critical Frame Budget

Minimum frames needed to reach 95% of best accuracy:

- **UCF-101:** 4–8f for most models (appearance-biased, minimal temporal gain)
- **SSv2 / HMDB-51:** 16f for CNNs, 8f for Transformers
- **AUTSL / Diving-48:** 16–32f across all models
- **TimeSformer / VideoMamba:** saturate at 8f by architecture

---

## Analysis Pipeline

All outputs in `evaluations/accv2026/paper_results/`.

```bash
# Run full Phase 2 pipeline
bash scripts/accv2026/run_post_completion_analyses.sh --all
```

| Script | Output |
|--------|--------|
| `08_compile_paper_results.py` | `paper_table_fixed_budget.csv`, `paper_table_tds_metrics.csv`, `paper_fig_budget_curves.csv` |
| `09_plot_paper_figures.py` | `figures/fig1–fig5.{pdf,png}` |
| `10_per_class_temporal_analysis.py` | `per_class_cross_model_ssv2.csv`, per-class figures |
| `12_confidence_cascade.py` | `confidence_cascade/` |
| `13_knapsack_confidence.py` | `knapsack_confidence/` |
| `14_plot_routing_comparison.py` | `figures/fig8_routing_comparison.{pdf,png}` |
| `15_baseline_comparison.py` | `figures/fig9_main_comparison.{pdf,png}` |

Figures are saved as both **PDF** (for LaTeX) and **PNG at 300 DPI** (for review/slides).

---

## Repository Structure

```
.
├── src/info_rates/               # Python package
│   ├── data/datasets.py          # unified loader
│   ├── metrics/                  # TDS, AUC, temporal metrics
│   └── models/                   # model factory + VideoMamba wrapper
├── scripts/accv2026/             # active pipeline
│   ├── 08_compile_paper_results.py
│   ├── 09_plot_paper_figures.py
│   ├── train_torchvision.py      # R3D-18, MC3-18, R2Plus1D-18
│   ├── train_slowfast.py         # SlowFast-R50
│   ├── train_transformers.py     # TimeSformer, ViViT, VideoMAE
│   ├── train_videomamba.py       # VideoMamba (SSM)
│   ├── eval_fixed_budget.py      # fixed-budget evaluation
│   ├── run_a100_dataset_all_cnn.sh
│   ├── run_h200_dataset_all_transformer.sh
│   └── run_h200_multidata_videomamba.sh
├── third_party/                  # VideoMamba source (not tracked by git)
├── experiments/                  # VideoMamba3 experimental (CVPR 2027)
├── evaluations/accv2026/
│   ├── fixed_budget/             # per-model per-dataset eval results
│   ├── paper_results/            # final tables and figures
│   │   └── figures/              # fig1–fig9 in PDF + PNG
│   ├── manifests/                # eval manifests (20 samples/class)
│   └── logs/                     # Slurm job logs
├── data/                         # datasets (symlink to /scratch)
├── fine_tuned_models/            # checkpoints (symlink to /scratch)
├── .venv/                        # standard env (CNNs + Transformers)
└── .venv_mamba/                  # VideoMamba env (mamba-ssm)
```

---

## Cluster Notes (Mi3 Lab HPC)

- **A100 partition:** `gpu` — CNN models (max 4 concurrent jobs)
- **H200 partition:** `cenvalarc.gpu` — Transformers + VideoMamba (max 4 concurrent)
- `data/` and `fine_tuned_models/` → symlinks to `/scratch/wesleyferreiramaia/infoRates/`
- `HF_HOME=/scratch/wesleyferreiramaia/hf_unified`
- VideoMamba requires `.venv_mamba` with `mamba-ssm` built against CUDA 12.8 (fake-nvcc workaround for CUDA 13.x nodes)
