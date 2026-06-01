# Paper Figures Index

Generated for ACCV 2026 submission: "Spatiotemporal Aliasing in Video Action Recognition"

## Main Paper Figures (3 total)

| Figure | File | Section | Description |
|--------|------|---------|-------------|
| **Fig 1** | `main_fig1_aliasing_curves.pdf` | Results (4.2) | Accuracy vs. stride at full coverage for three representative datasets (AUTSL, SSv2, UCF-101). Shows aliasing cliff per architecture. |
| **Fig 3** | `main_fig3_tds_spectral.pdf` | Results (4.1) | TDS ranking (left) and optical-flow magnitude correlation (right). Validates that \TDS{} is architecture-independent. |
| **Fig 5** | `main_fig5_spatial_resolution.pdf` | Results (4.4) | Spatial aliasing: accuracy vs. resolution (96–336px) on SSv2. Shows CNN collapse, Transformer robustness. |

## Supplementary Figures (13 total)

### Coverage×Stride Heatmaps (S1: 8 figures)
Full accuracy (\%) heatmaps for all 8 architectures × 7 datasets = 56 individual heatmaps.
Rows = coverage (10%, 25%, 50%, 75%, 100%); Cols = stride (1, 2, 4, 8, 16).

| Figure | Models | Coverage |
|--------|--------|----------|
| `sup1_heatmap_r3d_18.pdf` | R3D-18 (CNN-3D) | All 7 datasets (AUTSL, Diving, SSv2, HMDB, DriveAct, EPIC, UCF-101) |
| `sup1_heatmap_mc3_18.pdf` | MC3-18 (CNN-mix) | All 7 datasets |
| `sup1_heatmap_r2plus1d_18.pdf` | R2+1D (CNN-sep) | All 7 datasets |
| `sup1_heatmap_slowfast_r50.pdf` | SlowFast (dual-path) | All 7 datasets |
| `sup1_heatmap_timesformer.pdf` | TimeSformer (div-attn) | All 7 datasets |
| `sup1_heatmap_videomae.pdf` | VideoMAE (MAE) | All 7 datasets |
| `sup1_heatmap_vivit.pdf` | ViViT (fact-attn) | All 7 datasets |
| `sup1_heatmap_videomamba.pdf` | VideoMamba (SSM) | All 7 datasets (AUTSL excluded: feature collapse) |

### Analysis Figures

| Figure | File | Section | Description |
|--------|------|---------|-------------|
| **S2** | `sup2_levene_variance.pdf` | S2 | Levene variance inflation: std at stride=1 vs stride=16. Scatter plot showing instability under sparse sampling. |
| **S3** | `sup3_anova_eta2.pdf` | S3 | ANOVA effect sizes ($\eta^2$) per model. Bar chart showing coverage vs stride dominance by architecture. |
| **S4** | `sup4_taxonomy.pdf` | S4 | Action sensitivity taxonomy: aliasing loss (pp) per tier and dataset. Shows all-high AUTSL vs all-low UCF-101. |
| **S5** | `sup5_routing_all_models.pdf` | S5 | E7 entropy routing curves for all 8 models on SSv2. Accuracy vs avg frames at threshold sweep. |
| **S6** | `sup6_clip_duration.pdf` | S6 | Clip duration analysis (E10): aliasing loss vs clip length. Counter-intuitive: shorter clips alias more. |

## Figure Usage Summary

### Main Paper
- **Total figures**: 3
- **Coverage**: temporal aliasing (1 figure), TDS/spectral (1 figure), spatial aliasing (1 figure)
- **Page allocation**: ~2–3 pages for main figures

### Supplementary
- **Total figures**: 13
- **Section breakdown**:
  - S1 (heatmaps): 8 figures covering all 8 architectures
  - S2–S6 (analysis): 5 figures for variance, ANOVA, taxonomy, routing, duration
- **Page allocation**: ~16–18 pages for supplementary figures

## Notes

- **Naming convention**: `main_fig*.pdf` for paper, `sup*.pdf` for supplementary
- **Original sources**: `evaluations/accv2026/paper_figures/{main,supplementary}/`
- **All figures**: 300 dpi PNG + PDF versions available in source folder
- **VideoMamba/AUTSL**: AUTSL heatmap in `sup1_heatmap_videomamba.pdf` shows feature collapse (0.4% acc all strides) due to K400→sign language domain gap

## TODO for Paper Finalization

- [ ] Compile manuscript.tex with pdflatex/tectonic to verify all figure paths resolve
- [ ] Compile supplementary.tex and check page breaks
- [ ] Add caption references in text (e.g., "Figure~\ref{fig:aliasing_curves}")
- [ ] Verify all figure captions match descriptions in manuscript
- [ ] Check that figure DPI is 300+ for print quality
