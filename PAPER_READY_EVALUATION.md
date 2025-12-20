# Paper-Ready Evaluation Guide

## Overview
This guide explains how to generate publication-quality results for your temporal sampling paper using the enhanced evaluation and plotting scripts.

## What's Implemented

### 1. **Pareto Frontier Analysis**
- Extracts configurations that balance accuracy and latency optimally
- Eliminates dominated configurations (worse accuracy AND higher latency)
- **Logged to W&B**: `pareto_frontier` table
- **Plot**: `pareto_frontier.png` (latency vs accuracy scatter with frontier highlighted)

### 2. **Per-Class Aliasing Sensitivity**
- Measures how much each class suffers from undersampling (coverage drop: 100% → 25%)
- Identifies brittle vs robust classes
- **Logged to W&B**: `per_class_aliasing_drop` (top 20 most sensitive)
- **CSV**: `per_class_aliasing_drop.csv` (all classes ranked by sensitivity)

### 3. **Efficiency Metrics**
- Accuracy per second: combines speed and accuracy into single scalar
- Best efficiency configuration tracked separately from best accuracy
- **Logged to W&B**: `best_efficiency` in summary metrics
- **Plot**: `accuracy_per_second.png`

### 4. **Summary Report**
- Markdown file with key findings
- Includes: best overall config, best per-stride, best efficiency, Pareto frontier table
- **File**: `results_summary.md`

### 5. **Robustness via Jitter Evaluation** *(already in code)*
- Optional: evaluate with random coverage jitter (±%) to test stability
- Flag: `--jitter-coverage-pct 10` adds ±10% coverage randomness
- Shows which configs/classes remain robust under sampling variations

## Running Full Pipeline

### Step 1: Run Evaluation
```bash
export NPROC_PER_NODE=1
source .venv/bin/activate
cd /home/wesleyferreiramaia/data/infoRates

# Standard aggregate eval
python scripts/run_eval.py --config config.yaml --ddp \
  --out UCF101_data/results/ucf101_50f_finetuned.csv

# With per-class analysis
python scripts/run_eval.py --config config.yaml --ddp \
  --per-class \
  --per-class-out UCF101_data/results/ucf101_50f_per_class.csv

# With robustness jitter (optional, slower)
python scripts/run_eval.py --config config.yaml --ddp \
  --jitter-coverage-pct 5 \
  --out UCF101_data/results/ucf101_50f_jittered.csv
```

### Step 2: Generate Plots and Summary
```bash
# Generate aggregate plots and summary
python scripts/plot_results.py --config config.yaml \
  --csv UCF101_data/results/ucf101_50f_finetuned.csv

# Include per-class aliasing analysis
python scripts/plot_results.py --config config.yaml \
  --csv UCF101_data/results/ucf101_50f_finetuned.csv \
  --per-class-csv UCF101_data/results/ucf101_50f_per_class.csv
```

### Step 3: Check Outputs
All outputs go to `UCF101_data/results/`:
- `accuracy_vs_coverage.png` - Line plot: accuracy by stride and coverage
- `accuracy_heatmap.png` - Heatmap: stride × coverage → accuracy
- `accuracy_per_second.png` - Efficiency plot: accuracy/time by stride
- `pareto_frontier.png` - Pareto frontier with frontier configs highlighted
- `results_summary.md` - Summary report with all key metrics and tables
- `per_class_aliasing_drop.csv` - Per-class aliasing sensitivity (if per-class eval run)

## For the Paper

### Key Figures
1. **Accuracy vs Coverage** - Main result showing temporal sampling effects
2. **Pareto Frontier** - Shows efficiency trade-offs (best for resource-constrained scenarios)
3. **Aliasing Sensitivity** - Per-class breakdown; connect to "critical frequency" narrative

### Key Metrics to Report
From `results_summary.md`:

| Metric | Value | Insight |
|--------|-------|---------|
| Best Overall Accuracy | *from summary* | Maximum achievable performance |
| Best Efficiency (acc/sec) | *from summary* | Best speed-accuracy trade-off |
| Pareto Points | *table* | Optimal configurations at different latency budgets |
| Top Aliasing-Sensitive Classes | *from per_class_aliasing_drop.csv* | High-frequency motion actions |

### Narrative Examples

**Nyquist Connection:**
- Classes with high aliasing-drop (from per_class_aliasing_drop.csv) are likely high-frequency (rapid motion)
- Classes with low drop are robust (slow motion) → tie to sampling theorem

**Efficiency Story:**
- Report Pareto frontier points: "At X ms latency, we achieve Y% accuracy" 
- Useful for edge/mobile deployment discussion

**Robustness:**
- If you run jittered eval: report accuracy delta under ±5% coverage noise
- Shows stability vs sensitivity to sampling variations

## W&B Logging

All metrics automatically logged during `run_eval.py`:
- `results_table` - Full aggregate results
- `pareto_frontier` - Pareto-optimal configs
- `per_class_table` - All per-class results (if `--per-class`)
- `per_class_best` - Best config per class (if `--per-class`)
- `per_class_aliasing_drop` - Top 20 aliasing-sensitive classes (if `--per-class`)

## Quick Example: Running Now

```bash
cd /home/wesleyferreiramaia/data/infoRates
source .venv/bin/activate

# Full pipeline: aggregate + per-class
export NPROC_PER_NODE=1
python scripts/run_eval.py --config config.yaml --ddp \
  --per-class \
  --per-class-out UCF101_data/results/ucf101_50f_per_class.csv

# Generate plots
python scripts/plot_results.py --config config.yaml \
  --per-class-csv UCF101_data/results/ucf101_50f_per_class.csv

# Check summary
cat UCF101_data/results/results_summary.md
```

---

## Future Extensions (Next Phase)

Once you add new datasets/models:
1. Run same pipeline on each dataset → compare Pareto frontiers
2. Compare per-class aliasing sensitivity across datasets → universal vs domain-specific patterns
3. Add cross-model comparison → show if temporal sensitivity depends on architecture
4. Bootstrap CIs (infrastructure ready; just needs activation in code if needed)

---

**Ready for paper!** All figures and metrics are publication-quality PNG/CSV/Markdown.
