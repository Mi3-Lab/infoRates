# InfoRates — Research Progress & Roadmap

**ACCV 2026** · Mi3 Lab · Wesley Maia · PI: Ross Greer (UC Merced)
Last updated: 2026-05-28

---

## Context — Why This Paper Exists

A previous paper from this lab (ECCV 2026 submission #2612: *"On the Limits of Temporal Sampling in Video Action Recognition"*) was **rejected 3/3 reviewers** with these specific criticisms:

| Reviewer Concern | Our Response in InfoRates |
|-----------------|--------------------------|
| Transformers only (TSF/VMAE/ViViT) | 8 models: CNNs + Transformers + SSM (VideoMamba) |
| Datasets are background-biased (UCF-101, K400) | 7 datasets: AUTSL (+59.9pp TDS), Diving-48, SSv2, DriveAct — temporally demanding |
| TRA is just augmentation, not a real method | Adaptive routing (FDE, Cascade, Knapsack) beats fixed budget |

**The rejected paper already has:** coverage×stride grid, TRA, spectral analysis, ANOVA/Levene, sensitivity taxonomy — but only on 2 datasets × 3 models. InfoRates must replicate this analysis at scale and add the method.

---

## Research Contributions (Target)

| # | Contribution | One-line claim | Status |
|---|-------------|----------------|--------|
| **C1** | **TDS Score** | Dataset-level temporal demand consistent across all 8 models — proves aliasing is a dataset property, not architecture artifact | ✅ Data exists |
| **C2** | **Coverage × Stride aliasing analysis** | First cross-architecture (CNN/Transformer/SSM) × cross-domain aliasing characterization at scale | ❌ Not run |
| **C3** | **Real adaptive method** | Confidence-based routing beats fixed budget at same average compute, validated on high-TDS datasets | 🔶 Partial (oracle works, proxy weak) |

---

## What We Have

### ✅ Fixed-Budget Baseline (stride=1, budgets 4/8/16/32f)

59 eval runs: 8 models × 7 datasets, all clean splits.

| Dataset | R3D | MC3 | R2+1D | SF | TSF | ViViT | VMAE | VMamba | TDS |
|---------|:---:|:---:|:-----:|:--:|:---:|:-----:|:----:|:------:|----:|
| AUTSL | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌† | **+59.9pp** |
| Diving-48 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | +29.1pp |
| SSv2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | +26.1pp |
| HMDB-51 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | +24.2pp |
| EPIC | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | +20.6pp |
| DriveAct | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | +18.2pp |
| UCF-101 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | +15.9pp |

† VideoMamba AUTSL: feature collapse (K400→sign language domain gap). Valid finding.

### ✅ Analysis Pipeline

- `paper_table_fixed_budget.csv`, `paper_table_tds_metrics.csv`, `paper_fig_budget_curves.csv`
- Figures fig1–fig5 (PDF + PNG 300 DPI)
- Oracle routing gap: **+6.74pp** at same compute (proves adaptive routing is feasible)
- Confidence cascade and knapsack results

---

## Implementation Table — Everything Still Needed

Priority: 🔴 Blocking · 🟡 Important · 🟢 Good to have

| # | Experiment | What it produces | Addresses | Effort | Compute | Priority |
|---|-----------|-----------------|-----------|--------|---------|----------|
| **E1** | **Coverage × Stride grid** (C ∈ {10,25,50,75,100}%, s ∈ {1,2,4,8,16}) on all 8 models × all 7 datasets | Aliasing curves, heatmaps per model family | C2 — the core aliasing analysis. Reviewer: "transformers only" → we show CNNs/SSMs alias differently | 4 A100 jobs + 4 H200 jobs | ~12h | 🔴 |
| **E2** | **Variance analysis** — per-class accuracy std at each (C,s) config | Levene's test p-values, variance inflation plots | Shows aliasing is stochastic not just mean-accuracy drop. Mirrors ECCV paper section 4.2 | Post-processing of E1 | CPU only | 🔴 |
| **E3** | **Spectral analysis** — optical flow magnitude → dominant frequency per action class | Table: class freq (Hz) vs aliasing sensitivity (Pearson r≈0.99 expected) | Nyquist theoretical validation. The smoking gun that aliasing is real, not just compute | Python script on existing videos | ~2h CPU | 🔴 |
| **E4** | **ANOVA statistical analysis** (two-way: coverage × stride effects) | η² effect sizes, post-hoc pairwise, interaction significance | Statistical rigor. Reviewers always ask for this | Post-processing of E1 | CPU only | 🔴 |
| **E5** | **Action sensitivity taxonomy** — classify action classes into High/Moderate/Low aliasing sensitivity | Table 3-tier taxonomy (like ECCV paper Table 3) | Per-class insight. Shows the diversity the rejected paper lacked | Post-processing of E1 | CPU only | 🟡 |
| **E6** | **Spatial resolution sweep** (resize ∈ {112, 224, 448}px on existing checkpoints) | Spatial aliasing curves per dataset | "Spatiotemporal" claim in title — spatial is the missing half. AUTSL expected to cliff at 112px | 2 A100 jobs | ~4h | 🟡 |
| **E7** | **Better adaptive routing** — replace FDE proxy with per-frame model entropy | Routing accuracy above fixed-8f baseline | Reviewer: "TRA is just augmentation" → need a real method that beats fixed budget | Modify cascade script | ~2h | 🟡 |
| **E8** | **TRA (Temporal Robustness Augmentation)** — retrain key models with randomized C and s | TRA vs baseline robustness curves | Shows our analysis leads to actionable training improvement | Retrain 2-3 models | ~24h GPU | 🟢 |
| **E9** | **Comparison table vs prior adaptive methods** (AdaFocus, AR-Net, FrameExit) | Table with method comparison | Reviewers always ask "how does this compare?" | Literature + 1 reimplementation | ~4h | 🟢 |
| **E10** | **Clip duration analysis** — truncate clips to 1s/2s/5s at same frame count | Duration vs accuracy per dataset | Third dimension of spatiotemporal analysis | Post-processing of E1 subset | CPU only | 🟢 |

---

## E1 Protocol (Coverage × Stride) — Details

This is the core missing experiment. The rejected ECCV paper used:
- C ∈ {10, 25, 50, 75, 100}%, s ∈ {1, 2, 4, 8, 16} → 25 configs per model/dataset
- Fixed base clip of 50 frames; evaluate each config independently

**Our extension:**
- Same 25 configs
- Applied to all 8 models × 7 datasets → 1,400 eval runs total
- Key insight we add: **CNN vs Transformer vs SSM aliasing differs** — CNNs alias faster (local temporal conv) vs Transformers (global attention tolerates sparser sampling)
- Key dataset insight: **AUTSL at stride=4 will already alias** (signs change in <0.1s); **UCF-101 at stride=16 still works** (walking, lifting are slow)

**Implementation:** `eval_fixed_budget.py` already supports `--stride N` and `--coverage P`. Need to loop over the 25 configs.

---

## Architecture Aliasing Hypothesis (testable with E1)

| Architecture | Expected Aliasing Behavior | Mechanism |
|-------------|--------------------------|-----------|
| R3D-18 / MC3-18 | Moderate sensitivity, collapses at stride>4 | Local 3D conv needs nearby frames; stride breaks local temporal correlation |
| R2Plus1D | Similar to R3D but slightly more robust | Separable conv has some temporal flexibility |
| SlowFast | Robust at low strides (slow pathway compensates) | Dual pathway; slow path OK with sparse frames |
| TimeSformer | Most robust (confirmed by ECCV paper: 6.86% drop) | Global attention can aggregate from sparse frames |
| ViViT | Intermediate (confirmed: 13.18% drop) | Factorized attention, position-dependent |
| VideoMAE | Fragile at high stride (confirmed: 17.18% drop) | Reconstruction objective assumes dense local correlations |
| VideoMamba | Unknown — SSM processes frames as sequence | Mamba selective scan: may alias differently than attention |

Testing these hypotheses across 7 diverse datasets is the core scientific contribution.

---

## Running Jobs

| Job | Partition | Task | Status |
|-----|-----------|------|--------|
| 72830 | A100 | Coverage×Stride sweep — CNNs (R3D/MC3/R2+1D/SlowFast) × 7 datasets × 25 configs | Running |
| 72831 | A100 | Coverage×Stride sweep — Transformers (TSF/ViViT/VideoMAE) × 7 datasets × 25 configs | Running |
| 72832 | H200 | Coverage×Stride sweep — VideoMamba × 6 datasets × 25 configs | Pending |

---

## Execution Plan

```
This week — E1 coverage×stride sweep:
  Write slurm_coverage_stride_sweep.sbatch
  Submit for 3 representative models × all 7 datasets
  (Full 8-model run in parallel if slots available)

Same week — E3 spectral analysis:
  Compute optical flow on 20 videos per class for SSV2 + UCF-101
  Measure dominant frequency → correlate with TDS

Next week — E2 + E4 + E5:
  Variance analysis, ANOVA, sensitivity taxonomy (all post-processing of E1)

Following week — E6 spatial sweep + E7 better routing:
  Re-eval existing checkpoints at 112/224/448px
  Replace FDE with entropy-based routing

Paper writing in parallel throughout.
```

---

## Infrastructure

- **Eval script:** `eval_fixed_budget.py` supports `--stride N`, `--resize P`, `--coverage C`
- **A100 partition:** `gpu`, max 4 concurrent
- **H200 partition:** `cenvalarc.gpu`, max 4 concurrent
- **VideoMamba:** `.venv_mamba` env required
- **All checkpoints ready** — E1, E6 run on existing checkpoints, no retraining
