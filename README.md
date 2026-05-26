# InfoRates — Adaptive Temporal Frame Allocation for Video Recognition

**ACCV 2026 submission** · Mi3 Lab

We study how many frames a video model actually needs. Our method predicts the minimum frame budget per video — saving compute without sacrificing accuracy — and characterizes each dataset's *temporal demand* (TDS score).

---

## Requirements

- Python 3.10+
- CUDA GPU (trained on A100 40GB and H200 80GB)
- ~512 GB scratch storage for datasets + checkpoints

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs src/info_rates as a package
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
| EPIC-Kitchens | `data/EPIC_data/` | 97 | `python scripts/accv2026/download_epic_clips.py` |
| AUTSL | `data/AUTSL_data/` | 226 | [Kaggle: sttaseen/autsl](https://www.kaggle.com/datasets/sttaseen/autsl) |
| DriveAct | `data/DriveAct_data/` | 34 | [DriveAct](https://driveact.de/) |
| Kinetics-400 | `data/Kinetics400_data/` | 400 | [K400 val](https://github.com/cvdfoundation/kinetics-dataset) |

For AUTSL and DriveAct, run preprocessing after download:

```bash
python scripts/accv2026/preprocess_autsl.py   # generates data/AUTSL_data/splits/
python scripts/accv2026/preprocess_driveact.py # generates data/DriveAct_data/splits/
```

---

## Training

Models: **R3D-18**, **MC3-18**, **SlowFast-R50** · **TimeSformer**, **ViViT**

### Automated (recommended)

The feeder daemon submits Slurm jobs as GPU slots open, until all datasets × models are done:

```bash
nohup bash scripts/accv2026/feeder_submit_jobs.sh &
```

It calls `submit_missing_jobs.sh` every 2 minutes — idempotent, safe to re-run.

### Manual (one dataset)

```bash
# CNN models (R3D-18, MC3-18, SlowFast-R50) — requires A100
export DATASET=hmdb51   # hmdb51 | diving48 | epic_kitchens | autsl | driveact
bash scripts/accv2026/run_a100_dataset_all_cnn.sh

# Transformers (TimeSformer, ViViT) — requires H200 or large-memory GPU
bash scripts/accv2026/run_h200_dataset_all_transformer.sh
```

Each script is idempotent: skips training if a checkpoint exists, skips eval if results exist.

### Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | required | dataset name (see table above) |
| `EPOCHS` | `10` | training epochs |
| `WANDB_PROJECT` | `inforates-accv2026` | W&B project |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace model cache |

---

## Analysis Pipeline

After all training jobs finish:

```bash
bash scripts/accv2026/run_post_completion_analyses.sh
```

Or run steps individually (numbered scripts = pipeline order):

| Script | What it does |
|--------|-------------|
| `02_make_manifests.py` | Build per-class eval manifests |
| `04_compute_temporal_demand.py` | Compute TDS score per dataset |
| `05_compute_temporal_metrics.py` | AUC and critical budget per model |
| `07_dataset_temporal_demand.py` | Dataset-level TDS summary |
| `08_compile_paper_results.py` | Paper tables |
| `09_plot_paper_figures.py` | Paper figures (Fig 1–9) |
| `10_per_class_temporal_analysis.py` | Per-class temporal analysis |
| `06_fde_adaptive_routing.py` | FDE routing evaluation |
| `11_spectral_router.py` | Spectral router |
| `12_confidence_cascade.py` | Confidence cascade analysis |
| `13_knapsack_confidence.py` | Knapsack frame allocator |
| `14_plot_routing_comparison.py` | Routing comparison figures |
| `15_baseline_comparison.py` | Multi-dataset baseline table |

Results are written to `evaluations/accv2026/paper_results/`.

---

## Repository Structure

```
.
├── src/info_rates/               # Python package
│   ├── data/datasets.py          # unified loader: load_dataset(name, root)
│   ├── metrics/                  # TDS, AUC, temporal metrics
│   └── models/                   # model factory
├── scripts/accv2026/             # active pipeline
│   ├── 00–15_*.py                # numbered analysis steps
│   ├── train_torchvision.py      # R3D-18, MC3-18 trainer
│   ├── train_slowfast.py         # SlowFast-R50 trainer
│   ├── train_transformers.py     # TimeSformer, ViViT trainer
│   ├── eval_fixed_budget.py      # fixed-budget evaluation
│   ├── feeder_submit_jobs.sh     # automation daemon
│   ├── submit_missing_jobs.sh    # idempotent job submission
│   ├── run_a100_dataset_all_cnn.sh
│   ├── run_h200_dataset_all_transformer.sh
│   ├── preprocess_autsl.py
│   └── preprocess_driveact.py
├── scripts/legacy/               # archived experiments (ECCV, SSV2-only)
├── evaluations/accv2026/
│   ├── fixed_budget/             # per-model per-dataset eval results
│   ├── paper_results/            # final tables and figures
│   ├── manifests/                # eval manifests
│   └── logs/                     # Slurm job logs
├── data/                         # datasets (symlink to /scratch on cluster)
├── fine_tuned_models/            # checkpoints (symlink to /scratch on cluster)
└── requirements.txt
```

---

## Cluster Notes (Mi3 Lab HPC)

- **A100 partition:** `gpu` — CNN models (max 4 concurrent jobs)
- **H200 partition:** `cenvalarc.gpu` — transformers (max 4 concurrent jobs)
- `data/` and `fine_tuned_models/` are symlinked to `/scratch` (512 GB)
- `HF_HOME=/scratch/wesleyferreiramaia/hf_unified` — consolidated model cache
