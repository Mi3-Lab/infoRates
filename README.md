# InfoRates: Temporal Sampling for Action Recognition

This repository explores how temporal sampling (coverage and stride) affects action recognition across modern video models.

## 🆕 **Spectral Analysis: Now Available!**

**Quantitative validation of the Nyquist-Shannon sampling theorem for video action recognition.**

- 📊 Analyze optical flow frequency content from action videos  
- 🎯 Prove high-frequency actions require dense sampling; low-frequency actions tolerate subsampling
- 📈 Complete implementation with publication-quality visualizations
- ✅ All tests passing (6/6)

**Quick start:** `python scripts/run_spectral_analysis.py --output-dir evaluations/spectral_demo`

**Documentation:** See [docs/SPECTRAL_ANALYSIS.md](docs/SPECTRAL_ANALYSIS.md).

---

**Interactive Dashboard**: Visit our [GitHub Pages dashboard](https://mi3-lab.github.io/infoRates/) for an interactive reference tool with results tables, plots, and recommendations.

Quick start: see START_HERE.txt for commands, or the full docs/UNIFIED_GUIDE.md for end‑to‑end docs.

Key entry points
- Training (multi-model, DDP-ready): scripts/data_processing/train_ddp.sh → launches scripts/data_processing/train_multimodel.py
- Evaluation (multi-model): scripts/evaluation/run_eval_multimodel.py
- Plotting (all analysis plots): scripts/plotting/generate_analysis_plots.py --model MODEL --dataset DATASET
- Data management (build manifests, fix paths, download subsets): `scripts/manage_data.py` (subcommands: `build-manifest`, `fix-manifest`, `download`)
- Archived scripts: `scripts/archived/` contains deprecated scripts preserved for provenance; prefer `scripts/manage_data.py` for new workflows.
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

## Advanced Analysis Features

### Critical Frequency Analysis
Analyze action dynamics and identify optimal sampling rates:
```bash
# Analyze critical frequencies for a dataset
python scripts/analysis/critical_frequency_analysis.py --dataset ucf101 --sample-size 100
```

### Temporal Mitigation Strategies
Implement advanced techniques to combat aliasing:
```bash
# Temporal augmentation for robust training
python scripts/analysis/temporal_mitigation.py --mode augment

# Adaptive sampling based on model confidence
python scripts/analysis/temporal_mitigation.py --mode adaptive --video-path path/to/video.mp4

# Multiresolution analysis
python scripts/analysis/temporal_mitigation.py --mode multiresolution --video-path path/to/video.mp4
```

### Comprehensive Hyperparameter Sweep
Systematic testing across frame rates and clip durations:
```bash
# Full hyperparameter sweep (as per research milestones)
python scripts/analysis/hyperparameter_sweep.py --models timesformer videomae vivit --datasets ucf101 kinetics400 --dry-run

# Execute sweep
python scripts/analysis/hyperparameter_sweep.py --models timesformer videomae --max-workers 4
```

### Research Report Generation
Generate publication-ready reports with graphs and tables:
```bash
# Generate complete research report
python scripts/analysis/generate_research_report.py --results-dir evaluations --output-dir docs/research_report
```

## Paper and Figures

The comprehensive results analysis is documented in `docs/COMPREHENSIVE_RESULTS_ANALYSIS.md`, including all figures and LaTeX code for the paper.

Example figures demonstrating temporal aliasing are available in `docs/figures/` (YoYo action frames at different sampling rates).

## Benchmark Results and Recommendations

This repository serves as a reference for optimal temporal sampling configurations across different action recognition models and datasets. The table below summarizes key findings from our experiments, providing recommendations for coverage and stride based on activity type.

### Recommended Configurations by Activity Type

| Activity Type | Characteristics | Recommended Model | Coverage | Stride | Rationale |
|---------------|-----------------|-------------------|----------|--------|-----------|
| High-Frequency Actions (e.g., YoYo, JumpingJack, SalsaSpin) | Explosive, non-repetitive motions | TimeSformer | 100% | 1-2 | Requires dense sampling to capture rapid state changes |
| Moderate-Frequency Actions (e.g., Sports, tool use) | Dynamic controlled motions | ViViT | 75-100% | 2-4 | Balanced performance with some stride tolerance |
| Low-Frequency Actions (e.g., Billiards, Typing, locomotion) | Gentle, rhythmic motions | VideoMAE | 50-75% | 4-8 | Robust to subsampling, efficient for resource-constrained scenarios |

### Full Experimental Results Summary

| Dataset | Model | Peak Accuracy | Best Coverage-Stride | Mean Drop (100%→25%) | Latency (ms) | Notes |
|---------|-------|---------------|----------------------|----------------------|-------------|-------|
| UCF-101 | TimeSformer | 85.09% | 100%-stride2 | 6.86% | 0.000 | Most robust to stride changes |
| UCF-101 | VideoMAE | 86.90% | 100%-stride1 | 17.18% | 0.000 | Highest sensitivity to coverage reduction |
| UCF-101 | ViViT | 85.49% | 100%-stride1 | 13.18% | 0.000 | Good balance, occasional paradoxical improvements |
| Kinetics-400 | TimeSformer | 74.19% | 100%-stride4 | 10.60% | 0.000 | Consistent performance across configurations |
| Kinetics-400 | VideoMAE | 76.52% | 50%-stride2 | 7.16% | 0.000 | Benefits from moderate subsampling |
| Kinetics-400 | ViViT | 76.19% | 100%-stride1 | 8.23% | 0.000 | Stable at high coverage |

**Extensibility**: This benchmark is designed to be extensible. If you conduct additional experiments (e.g., new models, datasets, or activity types), please contribute by submitting a pull request with updated results. Include the evaluation CSV, statistical analysis, and a brief description of the setup. For questions or contributions, open an issue or email the maintainers.

**Note on Sign Language Recognition**: Sign language datasets (e.g., WLASL, MS-ASL) were not included in the current evaluation due to focus on general human activities. However, the temporal aliasing principles apply similarly, and we encourage extensions to include sign language recognition for fine-grained temporal analysis.