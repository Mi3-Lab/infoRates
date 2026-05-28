# InfoRates — Research Progress & Roadmap

**ACCV 2026** · Mi3 Lab · Wesley Maia
Last updated: 2026-05-28 — **Phase 1 COMPLETE (56 runs) · Phase 2 COMPLETE**

---

## Research Purpose

**Central question:** How many frames does a video model actually need to correctly classify a video?

**Hypothesis:** Temporal demand varies by dataset, class, and model. A fixed frame budget wastes compute on simple videos and fails on complex ones. Our method adaptively allocates frames per video, reducing FLOPs without sacrificing accuracy.

### Three Paper Contributions

1. **TDS Score (Temporal Demand Score)** — metric quantifying how much a dataset depends on temporal information. Computed as the accuracy drop when reducing frames from 32→4. Enables cross-dataset comparison on a common scale.

2. **InfoRates Adaptive Router** — given a video and a global compute budget (e.g., 8 frames/video on average), allocates more frames to hard videos and fewer to easy ones. Four variants:
   - **FDE Router** (Feature Diversity Estimation)
   - **Spectral Router** (temporal frequency of the video)
   - **Confidence Cascade** (early-exit by model confidence)
   - **Knapsack Allocator** (combinatorial optimization under global budget)

3. **Cross-dataset temporal analysis** — first systematic study with 7 datasets × 8 architectures under fixed budgets. Shows temporal demand is a property of the dataset, not the model.

### Main Paper Claim

> *"Not all videos need 32 frames. Our method identifies per-video temporal demand and allocates frames accordingly, matching full-budget accuracy at X% of the compute on Y% of videos."*

---

## Phase 1 Status — Fine-tuning + Fixed-Budget Evaluation

**Target:** 7 models × 7 datasets = 49 runs + VideoMamba × 8 datasets = 8 additional runs

| Dataset | R3D-18 | MC3-18 | R2Plus1D | SlowFast | TSF | ViViT | VideoMAE | VideoMamba | Status |
|---------|--------|--------|----------|----------|-----|-------|----------|------------|--------|
| SSv2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETE** |
| UCF-101 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETE** |
| HMDB-51 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETE** |
| DriveAct | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETE** |
| Diving-48 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETE** |
| AUTSL | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 7/8 — VideoMamba feature collapse (K400→sign language domain gap) |
| EPIC-Kitchens | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETE** |

---

## Fixed-Budget Results — Top-1 Accuracy by Frame Budget

| Dataset | Model | 4f | 8f | 16f | 32f |
|---------|-------|---:|---:|----:|----:|
| SSv2 | R3D-18 | 9.8% | 19.7% | 37.1% | 36.9% |
| SSv2 | MC3-18 | 8.2% | 18.8% | 33.6% | 34.5% |
| SSv2 | R2Plus1D-18 | 12.6% | 24.3% | 42.6% | 42.1% |
| SSv2 | SlowFast-R50 | 6.6% | 15.2% | 33.3% | 49.5% |
| SSv2 | TimeSformer | 31.8% | 42.3% | 41.3% | 41.7% |
| SSv2 | ViViT | 8.4% | 17.5% | 30.5% | 38.3% |
| SSv2 | VideoMAE | 21.0% | 39.5% | 52.3% | 51.9% |
| SSv2 | VideoMamba | 31.8% | 43.9% | 44.4% | 44.2% |
| UCF-101 | R3D-18 | 59.5% | 72.6% | 81.2% | 81.4% |
| UCF-101 | MC3-18 | 72.9% | 80.9% | 85.4% | 85.1% |
| UCF-101 | R2Plus1D-18 | 70.0% | 81.6% | 88.6% | 89.0% |
| UCF-101 | SlowFast-R50 | 50.1% | 66.2% | 81.3% | 87.6% |
| UCF-101 | TimeSformer | 90.0% | 91.0% | 91.2% | 90.9% |
| UCF-101 | ViViT | 75.3% | 86.9% | 92.5% | 94.3% |
| UCF-101 | VideoMAE | 81.4% | 91.4% | 95.4% | 95.5% |
| UCF-101 | VideoMamba | 85.0% | 88.4% | 88.2% | 87.8% |
| HMDB-51 | R3D-18 | 49.2% | 67.1% | 80.3% | 80.1% |
| HMDB-51 | MC3-18 | 63.5% | 71.2% | 78.6% | 78.2% |
| HMDB-51 | R2Plus1D-18 | 46.2% | 63.2% | 73.1% | 74.6% |
| HMDB-51 | SlowFast-R50 | 35.1% | 44.7% | 65.1% | 79.3% |
| HMDB-51 | TimeSformer | 73.0% | 79.9% | 80.0% | 79.8% |
| HMDB-51 | ViViT | 52.4% | 66.1% | 75.4% | 80.2% |
| HMDB-51 | VideoMAE | 51.5% | 73.6% | 84.0% | 84.4% |
| HMDB-51 | VideoMamba | 61.7% | 69.8% | 68.6% | 69.7% |
| DriveAct | R3D-18 | 47.8% | 56.2% | 68.3% | 67.2% |
| DriveAct | MC3-18 | 55.1% | 65.8% | 69.0% | 68.5% |
| DriveAct | R2Plus1D-18 | 37.7% | 49.8% | 62.5% | 61.8% |
| DriveAct | SlowFast-R50 | 42.6% | 53.3% | 66.7% | 72.5% |
| DriveAct | TimeSformer | 64.7% | 67.6% | 68.8% | 66.5% |
| DriveAct | ViViT | 48.9% | 55.8% | 62.5% | 67.4% |
| DriveAct | VideoMAE | 40.2% | 56.0% | 74.1% | 72.5% |
| DriveAct | VideoMamba | 50.9% | 57.8% | 58.0% | 56.7% |
| Diving-48 | R3D-18 | 5.9% | 14.4% | 28.8% | 28.8% |
| Diving-48 | MC3-18 | 8.2% | 19.5% | 31.6% | 33.4% |
| Diving-48 | R2Plus1D-18 | 8.6% | 16.8% | 35.3% | 34.7% |
| Diving-48 | SlowFast-R50 | 5.8% | 14.5% | 26.4% | 50.5% |
| Diving-48 | TimeSformer | 23.6% | 38.0% | 36.9% | 38.0% |
| Diving-48 | ViViT | 7.9% | 19.9% | 35.1% | 53.0% |
| Diving-48 | VideoMAE | 8.6% | 27.6% | 48.6% | 49.9% |
| Diving-48 | VideoMamba | 18.2% | 36.3% | 33.0% | 31.4% |
| AUTSL | R3D-18 | 4.7% | 24.5% | 75.0% | 74.4% |
| AUTSL | MC3-18 | 4.1% | 37.5% | 63.7% | 63.7% |
| AUTSL | R2Plus1D-18 | 8.4% | 30.2% | 75.9% | 75.0% |
| AUTSL | SlowFast-R50 | 1.6% | 12.7% | 41.8% | 82.3% |
| AUTSL | TimeSformer | 52.0% | 66.8% | 66.2% | 67.0% |
| AUTSL | ViViT | 8.4% | 25.5% | 61.2% | 74.6% |
| AUTSL | VideoMAE | 17.6% | 43.2% | 79.5% | 78.9% |
| AUTSL | VideoMamba | 0.4% | 0.4% | 0.4% | 0.4% |
| EPIC-Kitchens | R3D-18 | 13.6% | 22.3% | 37.2% | 37.0% |
| EPIC-Kitchens | MC3-18 | 11.3% | 27.1% | 36.2% | 37.2% |
| EPIC-Kitchens | R2Plus1D-18 | 13.0% | 20.2% | 35.5% | 35.2% |
| EPIC-Kitchens | SlowFast-R50 | 9.2% | 15.8% | 27.2% | 39.4% |
| EPIC-Kitchens | TimeSformer | 19.5% | 32.3% | 31.5% | 31.0% |
| EPIC-Kitchens | ViViT | 10.3% | 21.1% | 26.9% | 32.9% |
| EPIC-Kitchens | VideoMAE | 13.4% | 28.3% | 37.7% | 37.5% |
| EPIC-Kitchens | VideoMamba | 23.2% | 28.3% | 28.2% | 28.4% |

---

## Key Findings

- **TimeSformer saturates fast:** HMDB-51 73%→80% in just 8f. UCF-101 already 90% at 4f. critical_frame_budget = 4–8f.
- **SlowFast needs many frames:** HMDB-51 35%→79% (4→32f), Diving-48 5.8%→50.5%. Architectural — model_frames=32.
- **AUTSL (sign language):** sharp jump 4.7% (4f) → 75.0% (16f) → plateau. Minimum 16-frame temporal window required.
- **Diving-48 is the most temporally demanding dataset:** ViViT 7.9%→53.0%, SlowFast 5.8%→50.5%. Largest gain from more frames.
- **SSv2 vs UCF-101:** SSv2 has much higher TDS — subtle motions vs. coarse actions — confirms hypothesis.
- **EPIC-Kitchens (clean split):** All models converge to ~28–37% at 16f. VideoMAE is NOT superior on EPIC — previous 78% was inflated by data leakage (train/val video overlap). Dataset is uniformly hard for all models.
- **VideoMamba plateau after 8f:** architectural — model_frames=8; higher budgets subsample back to 8 frames from different temporal positions.
- **VideoMamba AUTSL (0.4%):** loss stuck at ln(226)≈5.42 throughout all 10 epochs with both LR=1e-4 and LR=5e-4 — exactly random chance for 226 classes. Root cause: K400 backbone produces near-identical features for all sign language clips (raw pixel std 10× lower than UCF-101). Feature collapse → gradients cancel across batches → no learning. CNN and Transformer models learn AUTSL fine because their local spatial inductive biases can detect hand shape even from K400 initialization.
- **Latency:** CNNs (R3D/MC3) are 6–25× faster than Transformers. MC3-18: ~1.6ms/sample; ViViT: ~41ms/sample. File: `evaluations/accv2026/paper_results/latency_summary.csv`.

---

## Bugs Found and Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| EPIC-Kitchens train/val video overlap | Inflated results for TSF/ViViT/SlowFast/R2Plus1D (e.g. VideoMAE 78%→37%) | Split rebuilt 2026-05-27; contaminated models retrained |
| R2Plus1D EPIC eval `--model-frames 8` (should be 16) | Results 3.1–6.1% instead of ~35% | Reeval with `--model-frames 16` |
| R2Plus1D EPIC eval `--resize 224` (trained at 112px) | Spatial mismatch → 11.8% instead of ~35% | Reeval with `--resize 112` |
| VideoMAE EPIC `--model-frames 8` in old result (78%) | Contaminated + wrong frames; inflated | Retrained on clean split; correct eval |
| SSv2 manifest using `videos/` path (empty dir) | VideoMamba SSV2 eval: 100% skipped | Manifest rebuilt with `videos_full/` paths |
| VideoMamba eval `--dataset-name ssv2` (manifest has `somethingv2`) | Manifest filter returns empty → crash | Fixed to `--dataset-name somethingv2` |
| VideoMamba eval `--split val` (SSv2/UCF-101 use `validation`) | iter_manifest filter returns empty → crash | Fixed to `--split validation` for SSv2/UCF-101 |

---

## Running Jobs

None — cluster is idle.

---

## TODO — Phase 3: Paper Writing

- [ ] **Review figures** — open `evaluations/accv2026/paper_results/figures/*.png` and check quality
- [ ] **Write Introduction** — motivation, TDS claim, preview of main results
- [ ] **Write Section 3: TDS** — formal definition, Fig 3 (TDS ranking bar), dataset ranking table
- [ ] **Write Section 4: InfoRates Routers** — FDE, Spectral, Cascade, Knapsack
- [ ] **Write Section 5: Experiments** — Table 1 from `paper_table_fixed_budget.csv`, Figs 1–2, Fig 8–9
- [ ] **Write Section 6: Conclusion**
- [ ] **VideoMamba AUTSL** — document as finding (K400→sign language domain gap), not a failure

---

## Phase 2 — Post-Training Analyses — COMPLETE ✅

All scripts ran successfully. Results in `evaluations/accv2026/paper_results/`.

| Script | Status | Output |
|--------|--------|--------|
| `08_compile_paper_results.py` | ✅ | `paper_table_fixed_budget.csv` (56 rows), `paper_table_tds_metrics.csv`, `paper_fig_budget_curves.csv` |
| `09_plot_paper_figures.py` | ✅ | `fig1–fig5.{pdf,png}` |
| `10_per_class_temporal_analysis.py` | ✅ | `per_class_cross_model_ssv2.csv`, per-class figures |
| `12_confidence_cascade.py` | ✅ | `confidence_cascade/cascade_global_summary.csv` |
| `13_knapsack_confidence.py` | ✅ | `knapsack_confidence/knapsack_global_results.csv` |
| `14_plot_routing_comparison.py` | ✅ | `fig8_routing_comparison.{pdf,png}` |
| `15_baseline_comparison.py` | ✅ | `fig9_main_comparison.{pdf,png}`, `paper_table_main_comparison.csv` |

**All figures saved as PDF (LaTeX) + PNG 300 DPI (review/slides).**

---

## VideoMamba3 Experiment (CVPR 2027 direction)

Applies 3 innovations from Mamba-3 (ICLR 2026, arXiv 2603.15569) to video understanding:
1. **Trapezoidal discretization** — 2nd-order integration, more temporally stable
2. **Complex states via RoPE** — complex-valued states without 4× parameter overhead
3. **MIMO low-rank state update** — matrix product vs. outer product (higher capacity)

**Ablation (job 72151 — 1 epoch, 512 samples, 2f, 112px):**

| Variant | Val acc | Speed | Memory |
|---------|---------|-------|--------|
| complex | **50.0%** | 1.6 vid/s | 57MB |
| trapezoidal | 35.2% | 2.7 vid/s | 57MB |
| mimo | 29.7% | 1.3 vid/s | 60MB |

**Key idea for CVPR 2027:** Use TDS score as a prior for task-adaptive VideoMamba3 configuration — high-TDS datasets (AUTSL, Diving-48) → longer scan / complex variant; low-TDS datasets (UCF-101) → trapezoidal (faster). This is a novel contribution because TDS was not available before this work.

**Status:** ablation done, mid-scale UCF-101 training started. Full multi-dataset training planned after ACCV 2026 submission.

---

## Infrastructure

- **Cluster:** Mi3 Lab HPC — A100 (gpu partition), H200 (cenvalarc.gpu partition), max 4 concurrent jobs per partition
- **Storage:** `data/` and `fine_tuned_models/` → symlinks to `/scratch/wesleyferreiramaia/infoRates/`
- **HF cache:** `/scratch/wesleyferreiramaia/hf_unified/` (~205GB)
- **VideoMamba env:** `.venv_mamba` (PyTorch 2.8.0+cu128, mamba-ssm with fake-nvcc workaround for CUDA 13.x)
- **W&B:** project `inforates-accv2026`
- **ACCV 2026 deadline:** check accv2026.org for official dates
