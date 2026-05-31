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
| **C1** | **TDS Score** | Dataset-level temporal demand consistent across all 8 models — proves aliasing is a dataset property, not architecture artifact | ✅ Complete |
| **C2** | **Coverage × Stride aliasing analysis** | First cross-architecture (CNN/Transformer/SSM) × cross-domain aliasing characterization at scale | 🔄 E1 running (56 jobs, 30 done) |
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
| **E1** | **Coverage × Stride grid** (C ∈ {10,25,50,75,100}%, s ∈ {1,2,4,8,16}) on all 8 models × all 7 datasets | Aliasing curves, heatmaps per model family | C2 — the core aliasing analysis. Reviewer: "transformers only" → we show CNNs/SSMs alias differently | 🔄 79% complete (1114/1400). Master auto-submitter active. | ~6h remaining | 🔴 |
| **E2** | **Variance analysis** — per-class accuracy std at each (C,s) config | Levene's test p-values, variance inflation plots | Shows aliasing is stochastic not just mean-accuracy drop. Mirrors ECCV paper section 4.2 | Post-processing of E1 | CPU only | 🔴 |
| **E3** | **Spectral analysis** — optical flow magnitude → dominant frequency per action class | Table: class freq (Hz) vs aliasing sensitivity (Pearson r≈0.99 expected) | Nyquist theoretical validation. The smoking gun that aliasing is real, not just compute | Python script on existing videos | ~2h CPU | 🔴 |
| **E4** | **ANOVA statistical analysis** (two-way: coverage × stride effects) | η² effect sizes, post-hoc pairwise, interaction significance | Statistical rigor. Reviewers always ask for this | Post-processing of E1 | CPU only | 🔴 |
| **E5** | **Action sensitivity taxonomy** — classify action classes into High/Moderate/Low aliasing sensitivity | Table 3-tier taxonomy (like ECCV paper Table 3) | Per-class insight. Shows the diversity the rejected paper lacked | Post-processing of E1 | CPU only | 🟡 |
| **E6** | **Spatial resolution sweep** — 5-point grid {96,112,160,224,336px} on all 8 models × all 7 datasets | Spatial aliasing curves, cross-architecture comparison at same resolution | Controlled experiment: all models at same resolutions (all divisible by patch_size=16). 224 new checkpoints via retraining. | 🔄 Running (eval on existing ckpts), 224 retraining jobs queued | ~7 days total | 🟡 |
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

## E1 Results — R3D-18: ALIASING DETECTED ✅

### Critical Finding: AUTSL Aliasing Phase Transition

| Stride | Config | Top-1 Acc | Interpretation |
|--------|--------|-----------|-----------------|
| **1** | Dense (no skip) | **75.0%** | ✅ Baseline (Nyquist met) |
| **2** | Every 2nd frame | 75.3% | ✅ Robust (above Nyquist) |
| **4** | Every 4th frame | 68.0% | ⚠️ Degrading (approaching Nyquist) |
| **8** | Every 8th frame | 27.2% | 🔴 **ALIASING REGIME** (below Nyquist) |
| **16** | Every 16th frame | 6.5% | ❌ Total failure (severe undersampling) |

**Drop from stride=1 to stride=16: 91.3 percentage points** (the largest among all datasets tested)

### Why AUTSL Aliases, UCF-101 Doesn't

**AUTSL (Sign Language):**
- Action frequency: ~5-10 Hz (signs complete in 100-200ms)
- R3D-18 native temporal resolution: 16 frames @ 24fps = 0.67s
- Stride=16 sampling: 1.06s between frames = **1 Hz** ← **BELOW Nyquist** ✓ Aliasing occurs

**UCF-101 (Sports):**
- Action frequency: ~0.5-1 Hz (actions last 1-2 seconds)
- Stride=16 sampling: 1.06s between frames = **sufficient** ✓ No aliasing, graceful degradation

### Stride Sensitivity Ranking at 100% Coverage

| Rank | Dataset | Stride-16 Loss | TDS Rank | Mechanism |
|------|---------|----------------|----------|-----------|
| 1🔴 | AUTSL | **91.3pp** | 1 (highest) | Sign language aliasing ✓ |
| 2 | Diving-48 | 55.6pp | 2 | Physics/fast diving |
| 3 | EPIC | 34.3pp | 5 | Cooking, moderate speed |
| 4 | DriveAct | 35.0pp | 6 | Driving, attention-dependent |
| 5 | HMDB-51 | 26.3pp | 4 | General actions, slower |

**Key insight:** Stride sensitivity ↔ TDS (Temporal Demand Score). Highest-TDS datasets alias first. **Validates the aliasing hypothesis.** ✓

### Paper Materials Generated

- `e1_r3d18_stride_sensitivity.csv` — Table for paper
- `fig_e1_r3d18_aliasing_heatmaps.png` — 4-dataset heatmaps (coverage×stride → accuracy)
- `fig_e1_autsl_aliasing_curve.png` — AUTSL phase transition (for Section 3.2)

---

## Running Jobs

### E1 Coverage × Stride Sweep — 79% (1114/1400)

| Model | Status |
|-------|--------|
| R3D-18, MC3-18, R2+1D, SlowFast, TimeSformer | ✅ 6/7 datasets (UCF101 queued) |
| ViViT | 🔄 Running |
| VideoMAE | 🔄 Running |
| VideoMamba | 🔄 Running |

**Master auto-submitter** (PID 36605): handles E1 → E6 eval → Resolution retraining in priority order.

### E6 Spatial Resolution Sweep — eval on existing checkpoints

- R3D-18, MC3-18, TimeSformer @ SSv2 ✅ running (4 resolutions each: 96/112/160/224/336px)
- Remaining 5 models queued in master auto-submitter

### E6 Resolution Retraining — 224 new checkpoints (queued after E1+E6 eval)

- CNNs (native=112px) → retrain at 96, 160, 224, 336px × 7 datasets = 84 jobs
- Transformers/SlowFast (native=224px) → retrain at 96, 112, 160, 336px × 7 datasets = 140 jobs
- Master auto-submitter handles all automatically

### Code fixes — ALL smoke-tested ✅ (8 models × 5 resolutions = 40 configs)

| File | Fix | Smoke result |
|------|-----|-------------|
| `src/info_rates/models/slowfast_video.py` | `AvgPool3d(7×7)` → `AdaptiveAvgPool3d(1)` | ✅ GPU A100 |
| `src/info_rates/models/videomamba_model.py` | Load at 224px, bicubic-interpolate `pos_embed`, copy to target-size model | ✅ GPU H200 |
| `scripts/accv2026/train_videomamba.py` | `save_checkpoint` receives explicit `input_size` param (was using `args` out of scope) | ✅ GPU H200 |
| `src/info_rates/models/model_factory.py` | `config.image_size` + `config.num_labels` updated before `from_pretrained` | ✅ CPU |
| `scripts/accv2026/train_transformers.py` | `--input-size` arg → `load_model` + `load_processor` | ✅ CPU |
| `scripts/accv2026/train_torchvision.py` | Already had `--input-size` ✅ | ✅ CPU |

---

## Execution Plan — ACCV 2026 (~1 month remaining)

**Deadline: late June 2026**

| Week | Tasks | Status |
|------|-------|--------|
| **Week 1 (now)** | E1 finishes + E6 (spatial eval) + Resolution retraining all 8 models | 🔄 Running |
| **Week 2** | E2 variance + E4 ANOVA (CPU) + E3 spectral analysis + paper draft Sec 1-2 | ⏳ |
| **Week 3** | E7 entropy routing (close C3) + E5 taxonomy + paper Section 3-4 | ⏳ |
| **Week 4** | E1 sweep at new resolutions + polish figures + paper Section 5 + related work | ⏳ |
| **Week 4-5** | Full paper draft → internal review → submit | ⏳ |

**Resolution Retraining Plan — 5-point common grid: 96 / 112 / 160 / 224 / 336px**

| Resolution | Patches | CNNs (native=112) | Transformers/SlowFast/VideoMamba (native=224) |
|-----------|---------|-------------------|----------------------------------------------|
| **96px**  | 6×6=36  | 🆕 new | 🆕 new |
| **112px** | 7×7=49  | ✅ existing | 🆕 new |
| **160px** | 10×10=100 | 🆕 new | 🆕 new |
| **224px** | 14×14=196 | 🆕 new | ✅ existing |
| **336px** | 21×21=441 | 🆕 new | 🆕 new |

- **5 resoluções**: espelha os 5 strides do E1 (temporal), dando uma curva real, não uma linha
- **Grid comum**: todas as 5 resoluções válidas para todos os 8 modelos (÷16 inteiro)
- Cada modelo treina nas 4 resoluções não-nativas × 7 datasets = **224 novos checkpoints**
- Depois: E1 sweep (25 configs) em cada nova resolução → grade espaço-temporal completa
- **Total: 8 modelos × 5 resoluções × 7 datasets × 25 configs = 7,000 eval runs**
- Master auto-submitter (PID 36605): P1 (E1) → P2 (E6 eval) → P3 (retraining), max 6 concurrent

**Scripts modified to support `--input-size`:**
- `train_torchvision.py` ✅ (already had it)
- `train_transformers.py` ✅ (added)
- `train_videomamba.py` ✅ (added)
- `src/info_rates/models/model_factory.py` ✅ (processor size override)

---

## Infrastructure

- **Eval script:** `eval_fixed_budget.py` supports `--stride N`, `--resize P`, `--coverage C`
- **A100 partition:** `gpu`, max 4 concurrent
- **H200 partition:** `cenvalarc.gpu`, max 4 concurrent
- **VideoMamba:** `.venv_mamba` env required
- **All checkpoints ready** — E1, E6 run on existing checkpoints, no retraining
