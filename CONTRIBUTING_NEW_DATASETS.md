# Contributing to InfoRates

Thank you for your interest in extending this benchmark. Contributions of new **datasets** and new **architectures** help build a more complete picture of spatiotemporal aliasing across the video recognition landscape.

This guide walks through the technical workflow. If you have questions at any step, open an issue or email us at the address at the [bottom of this page](#contact).

---

## Table of Contents

- [Contributing a New Dataset](#contributing-a-new-dataset)
- [Contributing a New Architecture](#contributing-a-new-architecture)
- [Submitting Your Results](#submitting-your-results)
- [Contact](#contact)

---

## Contributing a New Dataset

### Overview

Each dataset goes through three stages:

1. **Validation manifest** — a balanced clip list used for all evaluations
2. **P3-retrained checkpoints** — one fine-tuned model per (architecture, resolution) pair
3. **Coverage × stride × resolution sweep** — 1,000 evaluation configurations per architecture

The output integrates into the dashboard via three CSV files:
- `dashboard/data/sweep_summary.csv` — temporal sweep at native resolution
- `dashboard/data/p3_results.csv` — cross-resolution accuracy (stride=1, cov=100%)
- `dashboard/data/retrained_spatial.csv` — P3-retrained checkpoint accuracies

---

### Step 1 — Prepare the Validation Manifest

The manifest is a CSV listing the clips used for evaluation. We aim for a **balanced, fixed-size sample per class** so that all datasets are evaluated under the same statistical conditions.

**Target**: up to 20 clips per class, stratified random sampling from the official validation or test split.

```
evaluations/accv2026/manifests/{dataset_name}_val.csv
```

Columns:

| Column | Type | Description |
|--------|------|-------------|
| `video_path` | str | Absolute or relative path to the video file |
| `label` | str | Class name (use the dataset's original naming) |
| `dataset` | str | Dataset key (lowercase, underscores, e.g. `my_dataset`) |

Example rows:
```
/data/videos/my_dataset/class_a/clip_001.mp4,class_a,my_dataset
/data/videos/my_dataset/class_b/clip_047.mp4,class_b,my_dataset
```

**What if a class has fewer than 20 clips?** Use all available clips for that class. The minimum we recommend is 5 clips per class; classes below this threshold should be excluded from evaluation to avoid high-variance accuracy estimates. Document any such exclusions in your submission.

**What if the dataset has no official validation split?** Apply a stratified 80/20 train-validation split at the video level (not the clip level if videos have multiple clips). Use a fixed random seed (we use `seed=42`).

---

### Step 2 — Fine-Tune P3-Retrained Checkpoints

For each of the 8 architectures, fine-tune from the pretrained checkpoint at each target spatial resolution (48, 96, 112, 160, 224 px) for 10 epochs. This is referred to as **P3 retraining** in the paper.

Architecture configurations used in this project:

| Model | Frames | Native Res | Notes |
|-------|-------:|----------:|-------|
| R3D-18 | 16 | 112 px | |
| MC3-18 | 16 | 112 px | |
| R2+1D | 16 | 112 px | |
| SlowFast-R50 | 32 | 224 px | Slow + Fast pathway |
| TimeSformer | 8 | 224 px | Divided space-time attention |
| ViViT | 32 | 224 px | Factorised encoder |
| VideoMAE | 16 | 224 px | Masked autoencoder pretraining |
| VideoMamba | 8 | 224 px | Bidirectional SSM |

**Important for Transformers and SSMs**: when loading a checkpoint at a resolution that differs from its pretraining resolution, positional embeddings must be bicubic-interpolated before fine-tuning. Passing `ignore_mismatched_sizes=True` in HuggingFace silently discards the positional embeddings, causing a large accuracy drop. The correct implementation is in `src/info_rates/models/model_factory.py`.

Checkpoint naming convention:
```
accv2026_{model}_{dataset}_{resolution}px_e10_{gpu_suffix}
```

Example:
```
accv2026_timesformer_my_dataset_48px_e10_h200/
accv2026_timesformer_my_dataset_96px_e10_h200/
accv2026_r3d_18_my_dataset_112px_e10_a100/
```

---

### Step 3 — Run the Evaluation Sweep

With checkpoints in place, run the coverage × stride × resolution sweep for each architecture:

```bash
# Single architecture:
python scripts/accv2026/sweep_coverage_stride_resolution.py \
  --model timesformer \
  --dataset my_dataset \
  --manifest evaluations/accv2026/manifests/my_dataset_val.csv \
  --resolutions 48 96 112 160 224 \
  --batch-size 64 \
  --num-workers 8

# All 8 architectures (sequential):
python scripts/accv2026/sweep_all_models_coverage_stride_resolution.py \
  --dataset my_dataset \
  --manifest evaluations/accv2026/manifests/my_dataset_val.csv \
  --resolutions 48 96 112 160 224 \
  --batch-size 64 \
  --num-workers 8
```

This runs 5 resolutions × 5 coverages × 5 strides = **125 configurations per architecture** (1,000 total for 8 architectures). Expected runtime: 30–60 minutes per architecture on a single A100/H200.

Output layout:
```
evaluations/accv2026/coverage_stride_resolution_sweep/
└── {model}_{dataset}/
    ├── res48px/
    │   ├── sweep_summary.csv        # 25 rows: 5 cov × 5 stride
    │   └── cov{C}_s{S}_summary.csv  # per-config detailed results
    ├── res96px/
    ├── res112px/
    ├── res160px/
    └── res224px/
```

Each `sweep_summary.csv` has columns: `resolution, coverage, stride, top1, n, model, dataset`.

---

### Step 4 — Evaluate P3-Retrained Checkpoints (Spatial Only)

This extracts the single-point accuracy at stride=1, coverage=100% for each (architecture, resolution) pair, which populates the spatial resolution analysis in the dashboard:

```bash
python scripts/accv2026/eval_p3_retrained.py --dataset my_dataset
```

Output: `evaluations/accv2026/p3_retrained/{model}_{dataset}/res{N}_*_summary.csv`

---

### Step 5 — Integrate into the Dashboard

#### 5a. Register the dataset in `dashboard/app.py`

```python
# Add to DS_KEYS list (~line 37):
DS_KEYS = [..., "my_dataset"]

# Add to DS_LABELS dict (~line 38):
DS_LABELS = {
    ...,
    "my_dataset": "My Dataset (Domain Description)",
}
```

#### 5b. Run the integration script

Copy `scripts/accv2026/integrate_finegym_to_dashboard.py` and adapt the `dataset_name` variable and source paths for your dataset. Then run:

```bash
python scripts/accv2026/integrate_my_dataset_to_dashboard.py
```

The script appends to the three dashboard CSVs without touching existing entries.

#### 5c. Verify

```python
import pandas as pd

df = pd.read_csv("dashboard/data/sweep_summary.csv")
assert "my_dataset" in df["dataset"].unique(), "Dataset not found in sweep_summary.csv"
assert len(df[df["dataset"] == "my_dataset"]) == 200, \
    "Expected 200 rows (8 models × 5 coverage × 5 stride at native resolution)"

df_p3 = pd.read_csv("dashboard/data/p3_results.csv")
assert "my_dataset" in df_p3["dataset"].unique()

print("Integration verified.")
```

#### 5d. Update the eval config count in `dashboard/app.py`

Find the metric line and increment the dataset count accordingly. The formula is:

```
N_configs = num_datasets × 8 models × 5 resolutions × 5 coverages × 5 strides
```

#### 5e. Smoke-test the dashboard

```bash
streamlit run dashboard/app.py
```

Verify: (1) dataset appears in the Overview TDS chart, (2) the Accuracy Explorer shows results when your dataset is selected with non-default stride/coverage settings, (3) Spatial Resolution curves are populated.

---

## Contributing a New Architecture

Adding a new model requires three things: a loader in `model_factory.py`, a sweep run, and dashboard registration.

### Step 1 — Add the Model Loader

Implement a loader in `src/info_rates/models/model_factory.py` that:
- Accepts `(model_name, num_classes, resolution, pretrained=True)` as arguments
- Returns a `torch.nn.Module` ready for `model.eval()`
- Handles positional embedding interpolation if the model uses patch-based attention (see `_interp_pos_embed` in that file for reference)

### Step 2 — Register in Constants

In `dashboard/app.py`, add to:
```python
MODEL_KEYS  = [..., "my_model"]
MODEL_NAMES = {..., "my_model": "My Model"}
FAMILIES    = {..., "my_model": "Transformer"}   # CNN | Dual-CNN | Transformer | SSM | Other
FAM_COLOR   = {...}  # add a color if introducing a new family
```

### Step 3 — Run the Sweep Across All Existing Datasets

Run the full sweep for your model against all datasets currently in the benchmark. The sweep script accepts `--model my_model` directly once the loader is registered.

```bash
for dataset in autsl ssv2 hmdb51 diving48 driveact epic_kitchens ucf101 finegym; do
  python scripts/accv2026/sweep_coverage_stride_resolution.py \
    --model my_model --dataset $dataset \
    --resolutions 48 96 112 160 224
done
```

### Step 4 — Integrate and Verify

Same as Steps 4–5 in the dataset workflow: run the integration script (adding a `model` filter rather than `dataset`), verify row counts, and smoke-test the dashboard.

---

## Submitting Your Results

Once your contribution is ready:

1. **Open a pull request** to this repository with:
   - The validation manifest (`evaluations/accv2026/manifests/`)
   - Updated `dashboard/data/` CSVs (sweep_summary, p3_results, retrained_spatial)
   - Any changes to `dashboard/app.py` (DS_KEYS, DS_LABELS, MODEL_KEYS, etc.)
   - Your integration script under `scripts/accv2026/`

2. **Include in the PR description**:
   - Dataset or model name and a one-sentence description
   - Number of classes, approximate number of evaluation clips
   - TDS score and a brief interpretation (what drives temporal demand in this domain?)
   - Any deviations from the standard protocol (e.g., fewer clips per class, custom split)
   - Hardware used and approximate runtime

3. **If you cannot open a pull request** (e.g., the dataset is under a restricted license and results cannot be published directly), contact us by email with your result CSVs and we will integrate them manually.

---

## Contact

**Wesley Maia** — Mi3 Lab, UC Merced
wesleymaia999@gmail.com

For questions about the evaluation protocol, sweep scripts, or integration workflow, open a GitHub issue or reach out by email. We are happy to assist with datasets that require non-standard handling (restricted licenses, unusual class distributions, non-standard video formats).
