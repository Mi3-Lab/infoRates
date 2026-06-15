# Contributing New Datasets to InfoRates

This guide explains how to add new datasets and models to the InfoRates project, following the same workflow used to integrate FineGym.

## Overview

The InfoRates project evaluates 8 video action recognition architectures across multiple datasets using coverage × stride × resolution sweeps. To contribute a new dataset:

1. **Prepare evaluation data** (Coverage × Stride × Resolution sweep)
2. **Run the sweep scripts**
3. **Integrate results into the dashboard**
4. **Update documentation**

## Project Structure

```
/mnt/datasets/infoRates/
├── evaluations/accv2026/
│   ├── manifests/                          # Dataset split files (20 samples per class)
│   │   └── {dataset}_val_20_per_class.csv
│   ├── coverage_stride_resolution_sweep/   # Multi-resolution sweeps (NEW datasets)
│   │   └── {model}_{dataset}/
│   │       ├── res48px/
│   │       ├── res96px/
│   │       ├── res112px/
│   │       ├── res160px/
│   │       └── res224px/
│   ├── p3_retrained/                       # P3-retrained checkpoint evaluations
│   │   └── {model}_{dataset}/
│   └── coverage_stride_sweep/              # Legacy sweeps (original 7 datasets)
├── scripts/accv2026/
│   ├── sweep_coverage_stride_resolution.py # Single-model sweep script
│   ├── sweep_all_models_coverage_stride_resolution.py # Batch all models
│   └── integrate_finegym_to_dashboard.py   # Integration script
├── dashboard/
│   ├── app.py                              # Streamlit app
│   └── data/
│       ├── sweep_summary.csv               # Aggregated temporal data
│       ├── p3_results.csv                  # Spatial resolution results
│       └── retrained_spatial.csv           # P3-retrained checkpoints
└── fine_tuned_models/                      # Trained checkpoint directory
```

## Step-by-Step: Adding a New Dataset

### Step 1: Prepare the Dataset Manifest

Create a validation split file with 20 samples per class (stratified random sampling):

```bash
# File: evaluations/accv2026/manifests/{new_dataset}_val_20_per_class.csv
# Columns: video_path, label, dataset
# Example:
# path/to/video_001.mp4,action_label,new_dataset
# path/to/video_002.mp4,action_label,new_dataset
```

The manifest should contain exactly 20 clips per action class for consistent evaluation.

### Step 2: Train P3-Retrained Checkpoints

For each model and resolution (48px, 96px, 112px, 160px, 224px), fine-tune the base checkpoint:

```python
# Model configurations (frames, native resolution, checkpoint suffix):
MODEL_CFG = {
    "r3d_18":       dict(frames=16, resize=112, ckpt_suffix="a100"),
    "mc3_18":       dict(frames=16, resize=112, ckpt_suffix="a100"),
    "r2plus1d_18":  dict(frames=16, resize=112, ckpt_suffix="a100"),
    "slowfast_r50": dict(frames=32, resize=224, ckpt_suffix="a100"),
    "timesformer":  dict(frames=8,  resize=224, ckpt_suffix="h200"),
    "vivit":        dict(frames=32, resize=224, ckpt_suffix="h200"),
    "videomae":     dict(frames=16, resize=224, ckpt_suffix="h200"),
    "videomamba":   dict(frames=8,  resize=224, ckpt_suffix="h200"),
}
```

Trained checkpoints must follow the naming pattern:
```
accv2026_{model}_{dataset}_{resolution}px_e10_{suffix}
```

Example:
```
accv2026_r3d_18_new_dataset_48px_e10_a100/
accv2026_r3d_18_new_dataset_96px_e10_a100/
accv2026_r3d_18_new_dataset_112px_e10_a100/
accv2026_r3d_18_new_dataset_160px_e10_a100/
accv2026_r3d_18_new_dataset_224px_e10_a100/
```

**Important**: Use bicubic interpolation for positional embeddings when loading at non-native resolutions (HuggingFace transformers). See `src/info_rates/models/model_factory.py` for implementation details.

### Step 3: Run the Coverage × Stride × Resolution Sweep

Once checkpoints are trained and stored in `/scratch/wesleyferreiramaia/infoRates/fine_tuned_models/`, run the evaluation sweep:

```bash
# Single model example:
python scripts/accv2026/sweep_coverage_stride_resolution.py \
  --model r3d_18 \
  --dataset new_dataset \
  --resolutions 48 96 112 160 224 \
  --batch-size 96 \
  --num-workers 8

# All 8 models (sequential):
python scripts/accv2026/sweep_all_models_coverage_stride_resolution.py \
  --dataset new_dataset \
  --resolutions 48 96 112 160 224 \
  --batch-size 96 \
  --num-workers 8
```

Output structure:
```
evaluations/accv2026/coverage_stride_resolution_sweep/
└── {model}_{dataset}/
    ├── res48px/
    │   ├── cov10_s1_samples.csv     # Raw predictions (5 coverages × 5 strides = 25 configs)
    │   ├── cov10_s1_summary.csv
    │   ├── cov10_s2_samples.csv
    │   ...
    │   └── sweep_summary.csv        # Aggregated for this resolution
    ├── res96px/
    └── ...
    └── sweep_summary_all_resolutions.csv  # All 1000 configs (5 res × 5 cov × 5 stride × 8 models)
```

**Output CSVs contain:**
- `resolution`: spatial resolution (48, 96, 112, 160, 224)
- `coverage`: percentage of frames observed (10, 25, 50, 75, 100)
- `stride`: frame sampling interval (1, 2, 4, 8, 16)
- `top1`: accuracy (decimal, 0–1)
- `n`: number of evaluation samples
- `model`: model name
- `dataset`: dataset name

### Step 4: Evaluate P3-Retrained Checkpoints at Native Resolution

Extract single-point accuracy (stride=1, coverage=100%) for each model-dataset-resolution combination:

```bash
python scripts/accv2026/eval_p3_retrained.py --dataset new_dataset
```

This creates `evaluations/accv2026/p3_retrained/{model}_{dataset}/res{N}_*_summary.csv` files.

### Step 5: Integrate into Dashboard

Update the app to recognize the new dataset:

#### 5a. Add dataset to constants (`dashboard/app.py`):

```python
# Line ~37
DS_KEYS = ["autsl", "diving48", "ssv2", "hmdb51", "driveact", "epic_kitchens", 
           "ucf101", "finegym", "new_dataset"]

DS_LABELS = {
    ...existing datasets...
    "new_dataset": "New Dataset (Domain Description)",
}
```

#### 5b. Run the integration script:

```bash
python scripts/accv2026/integrate_finegym_to_dashboard.py
```

**Adapt the script for your dataset:**
- Update `dataset_name` variable
- Ensure paths match your data structure
- Verify CSV columns: `resolution, coverage, stride, top1, n, model, dataset`

The script updates:
- `dashboard/data/sweep_summary.csv` (temporal sweep @ native resolution)
- `dashboard/data/p3_results.csv` (spatial resolution evaluation)
- `dashboard/data/retrained_spatial.csv` (P3-retrained checkpoint results)

#### 5c. Verify integration:

```python
import pandas as pd
df = pd.read_csv('dashboard/data/sweep_summary.csv')
print(f"Datasets: {sorted(df['dataset'].unique())}")
print(f"New dataset rows: {len(df[df['dataset']=='new_dataset'])}")
```

Expected: 200 rows (8 models × 5 coverage × 5 stride at native resolution)

### Step 6: Update App Display Numbers

Update statistics in `dashboard/app.py`:

```python
# Line ~363
c3.metric("Eval configs", "8,000+", "8 models × 8 datasets × res × cov × stride")
```

Recalculate as: `(num_datasets × 8 models × 5 res × 5 cov × 5 stride)`

### Step 7: Test the Dashboard

Restart Streamlit and verify:

```bash
pkill -9 streamlit
cd /mnt/datasets/infoRates/dashboard
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Check:
1. **Overview & TDS**: New dataset appears in table with TDS score
2. **Accuracy Explorer**: Select new dataset, vary stride/coverage → results appear
3. **Spatial Resolution**: Compare retrained vs native models
4. **Aliasing Curves**: Stride degradation patterns visible
5. **Architecture Recommender**: New dataset in keyword matching

## Key Implementation Details

### Positional Embedding Interpolation

When evaluating transformers at non-native resolutions, use bicubic interpolation:

```python
# In model_factory.py
def _interp_pos_embed(model, native_size: int, target_size: int) -> None:
    """Bicubic-interpolate spatial positional embeddings in-place."""
    patch_size = 16
    H_nat = native_size // patch_size
    H_tgt = target_size // patch_size
    
    for pname, param in model.named_parameters():
        if "position_embed" in pname.lower():
            # Reshape, interpolate, reshape back
            grid = param.data.reshape(1, H_nat, H_nat, D).permute(0, 3, 1, 2)
            grid = F.interpolate(grid, size=(H_tgt, H_tgt), mode="bicubic")
            # ... update parameter
```

### Checkpoint Loading Priority

The sweep script searches for checkpoints in this order:

1. **P3-retrained @ exact resolution**: `accv2026_{model}_{dataset}_{resolution}px_e10_{suffix}`
2. **Special hardcoded names** (SSv2, FineGym)
3. **Native resolution fallback**: `accv2026_{model}_{dataset}_224px_e10_{suffix}`

### Data Aggregation Formula

- **Temporal sweep** (sweep_summary.csv): native resolution only
  - 7 original datasets: `7 × 8 models × 5 coverage × 5 stride = 1,400 rows`
  - Per new dataset: `1 × 8 models × 5 coverage × 5 stride = 200 rows`

- **Multi-resolution sweep** (coverage_stride_resolution_sweep/):
  - Per new dataset: `8 models × 5 resolution × 5 coverage × 5 stride = 1,000 rows`

- **Total configs**: `(num_datasets × 8 × 5 × 5 × 5)`

## Troubleshooting

### Missing checkpoint error in sweep script

```
FileNotFoundError: Checkpoint not found for {model}/{dataset}@{resolution}px
```

**Solution**: Verify checkpoint exists at:
```
/scratch/wesleyferreiramaia/infoRates/fine_tuned_models/
  accv2026_{model}_{dataset}_{resolution}px_e10_{suffix}/
```

### CSV column mismatch

Ensure sweep output CSVs have exactly these columns (order matters):
```
resolution, coverage, stride, top1, n, model, dataset
```

### Dashboard doesn't show new dataset

1. Run `integrate_finegym_to_dashboard.py` (or equivalent)
2. Check `dashboard/data/sweep_summary.csv` contains dataset name
3. Restart Streamlit (`pkill -9 streamlit`)
4. Clear browser cache (Ctrl+Shift+Delete)

## Performance Benchmarks

- **Single model sweep** (1 dataset × 5 res × 5 cov × 5 stride):
  - Time: ~30-60 minutes (RTX 6000 Blackwell)
  - Memory: 84GB VRAM
  
- **All 8 models** (sequential):
  - Time: ~4-8 hours
  - Batch size: 96 (adjust for your GPU)

- **Integration script**:
  - Time: <5 minutes
  - Reads from disk, no GPU required

## Contributing Back

Once your dataset evaluation is complete:

1. Create a pull request with:
   - New manifest file
   - Updated `dashboard/app.py` (DS_KEYS, DS_LABELS)
   - Integration script results (updated CSVs)
   - Brief dataset description (domain, class count, typical accuracy range)

2. Include in PR description:
   - Dataset characteristics (# classes, # videos, action categories)
   - Temporal demand score (TDS) interpretation
   - Any dataset-specific findings (e.g., resolution sensitivity)

3. Example PR template:
   ```
   ## Dataset: New Dataset
   
   - **Domain**: [Sport/Gesture/Egocentric/etc]
   - **Classes**: [N]
   - **Videos**: [M]
   - **TDS Score**: [X.X]pp (coverage=100%, stride=1→16 drop)
   - **Key Finding**: [Observation about temporal aliasing sensitivity]
   
   Evaluation complete: 8,000 configs (8 models × 5 res × 5 cov × 5 stride)
   ```

## References

- **Main sweep script**: `scripts/accv2026/sweep_coverage_stride_resolution.py`
- **Batch runner**: `scripts/accv2026/sweep_all_models_coverage_stride_resolution.py`
- **Integration**: `scripts/accv2026/integrate_finegym_to_dashboard.py`
- **Dashboard app**: `dashboard/app.py`
- **Model loading**: `src/info_rates/models/model_factory.py`

## Questions?

For questions about the workflow, refer to:
1. This guide (step-by-step instructions)
2. Script docstrings (function signatures and parameters)
3. Git commit history (`git log --oneline --grep="FineGym"`)
4. Recent PRs adding datasets to the project
