# ACCV 2026 — Run Status

> Last updated: 2026-05-25

---

## Overview

| Dataset | Models | Eval Status |
|---------|--------|-------------|
| SSV2 | R2+1D, R3D-18, MC3-18, SlowFast, TimeSformer, VideoMAE, ViViT | ✅ Complete |
| UCF101 | R2+1D, SlowFast, VideoMAE | ✅ Complete |
| HMDB51 | R2+1D, SlowFast, VideoMAE | ✅ Complete |
| Diving48 | R2+1D, SlowFast, VideoMAE | ✅ Complete |
| EPIC-Kitchens | VideoMAE, R2+1D | 🔄 Running |
| WLASL100 | VideoMAE, R2+1D | 🔄 Training |
| WLASL2000 | — | ❌ Dropped |

---

## Fixed-budget evaluation results (completed)

Accuracy at each frame budget (4 / 8 / 16 / 32 frames). Best budget in **bold**.

### SSV2 (Something-Something V2)

| Model | Family | 4f | 8f | 16f | 32f |
|-------|--------|----|----|-----|-----|
| VideoMAE | Transformer | 21.0% | 39.5% | **52.3%** | 51.9% |
| TimeSformer | Transformer | 31.8% | **42.3%** | 41.3% | 41.7% |
| ViViT | Transformer | 8.4% | 17.5% | 30.5% | **38.3%** |
| SlowFast R50 | SlowFast | 6.6% | 15.2% | 33.3% | **49.5%** |
| R(2+1)D-18 | 3D CNN | 12.6% | 24.3% | **42.6%** | 42.1% |
| R3D-18 | 3D CNN | 9.8% | 19.7% | **37.1%** | 36.9% |
| MC3-18 | 3D CNN | 8.2% | 18.8% | **34.5%** | 34.5% |

### UCF101

| Model | Family | 4f | 8f | 16f | 32f |
|-------|--------|----|----|-----|-----|
| VideoMAE | Transformer | 81.4% | 91.4% | 95.4% | **95.5%** |
| SlowFast R50 | SlowFast | 50.1% | 66.2% | 81.3% | **87.6%** |
| R(2+1)D-18 | 3D CNN | 70.0% | 81.6% | 88.6% | **89.0%** |

### HMDB51

| Model | Family | 4f | 8f | 16f | 32f |
|-------|--------|----|----|-----|-----|
| VideoMAE | Transformer | 51.5% | 73.6% | 84.0% | **84.4%** |
| SlowFast R50 | SlowFast | 35.1% | 44.7% | 65.1% | **79.3%** |
| R(2+1)D-18 | 3D CNN | 46.2% | 63.2% | 73.1% | **74.6%** |

### Diving48

| Model | Family | 4f | 8f | 16f | 32f |
|-------|--------|----|----|-----|-----|
| SlowFast R50 | SlowFast | 5.8% | 14.5% | 26.4% | **50.5%** |
| VideoMAE | Transformer | 8.6% | 27.6% | 48.6% | **49.9%** |
| R(2+1)D-18 | 3D CNN | 8.6% | 16.8% | **35.3%** | 34.7% |

---

## Running jobs

| Job ID | Description | GPU | Time Limit | Status |
|--------|-------------|-----|------------|--------|
| 70780 | EPIC VideoMAE training | H200 | 8h | Epoch 9/10 done, best val=**64.66%** (ep6) |
| 70846 | EPIC R2+1D eval-only | A100 | 4h | Running from epoch-7 ckpt (59.55%) |
| 70850 | WLASL100 VideoMAE training | H200 | 1 day | Epoch 1 in progress |
| 70851 | WLASL100 R2+1D training | A100 | 1 day | Epoch 1 in progress |

### EPIC-Kitchens training history

**VideoMAE (H200, job 70780):**

| Epoch | val_acc | Note |
|-------|---------|------|
| 1 | 58.52% | |
| 2 | 61.75% | |
| 3 | 63.95% | |
| 4 | 62.36% | |
| 5 | 64.04% | |
| 6 | **64.66%** | ← best |
| 7 | 61.09% | |
| 8 | 64.22% | |
| 9 | 64.08% | |
| 10 | in progress | |

**R2+1D (A100, job 70782 — timed out at 6h):**

| Epoch | val_acc | Note |
|-------|---------|------|
| 1 | 56.04% | |
| 2 | 57.82% | |
| 3 | **60.01%** | ← best |
| 4 | 58.67% | |
| 5 | 59.61% | |
| 6 | 59.25% | |
| 7 | 59.55% | ← last checkpoint; eval running now |

---

## Dropped datasets

| Dataset | Reason |
|---------|--------|
| WLASL2000 (2000 classes) | VideoMAE 4.4%, R2+1D <1%. Natural video pre-training does not transfer to fine-grained sign language at 2000 classes (~10 videos/class). Replaced by WLASL100. |

WLASL100 (first 100 classes of same dataset) has ~10 videos/class for training but domain gap is much more tractable. Expected: VideoMAE 40-60%, R2+1D 30-45%.

---

## Next steps (after all jobs finish)

1. **Run post-completion analyses:**
   ```bash
   bash scripts/accv2026/run_post_completion_analyses.sh
   ```
   Compiles paper tables, TDS correlation, budget curves, confidence cascade, knapsack allocator.

2. **Add WLASL100 to routing scripts** (`12_confidence_cascade.py`, `13_knapsack_confidence.py`).

3. **Run baseline comparison:**
   ```bash
   python scripts/accv2026/15_baseline_comparison.py
   ```
   FrameExit vs. knapsack allocator per dataset.

4. **Paper writing** — intro, related work, method, experiments sections.
