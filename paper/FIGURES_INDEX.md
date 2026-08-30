# Paper Figures Index

Generated for ACCV 2026 submission: "Measuring Spatiotemporal Sampling Sensitivity in Video Action Recognition"

## Main Paper Figures (5 total)

| Figure | File | Section | Description |
|--------|------|---------|-------------|
| **Fig 1** | `main_fig1_aliasing_curves.pdf` | Results | Descriptive Top-1 curves under the original fixed-budget stride protocol for AUTSL, SSv2, and UCF-101. |
| **Fig 2** | `main_fig2_aliasing_heatmap.pdf` | Results | Accuracy-drop heatmap (stride 1 to 16); cells report evidence loss, not Top-1 accuracy. |
| **Fig 3** | `main_fig3_tds_spectral.pdf` | Results | TDS ranking (left) and exploratory whole-frame motion-proxy correlation (right). |
| **Fig 4** | `main_fig4_routing_comparison.pdf` | Results | Descriptive in-sample maximum-probability cascade curves; held-out calibration is reported in the table. |
| **Fig 5** | `main_fig5_spatial_resolution.pdf` | Results | Input-scale sensitivity after source-frame pre-resizing (96–224px) under unchanged native checkpoint preprocessing. |

## Supplementary Figures (13 total)

### Coverage×Stride Heatmaps (S1: 8 figures)
Full accuracy (\%) heatmaps for all 8 architectures × 7 datasets = 56 individual heatmaps.
Rows = coverage (10%, 25%, 50%, 75%, 100%); Cols = stride (1, 2, 4, 8, 16).

| Figure | Models | Coverage |
|--------|--------|----------|
| `sup1_heatmap_r3d_18.pdf` | R3D-18 (CNN-3D) | All 8 datasets |
| `sup1_heatmap_mc3_18.pdf` | MC3-18 (CNN-mix) | All 8 datasets |
| `sup1_heatmap_r2plus1d_18.pdf` | R2+1D (CNN-sep) | All 8 datasets |
| `sup1_heatmap_slowfast_r50.pdf` | SlowFast (dual-path) | All 8 datasets |
| `sup1_heatmap_timesformer.pdf` | TimeSformer (div-attn) | All 8 datasets |
| `sup1_heatmap_videomae.pdf` | VideoMAE (MAE) | All 8 datasets |
| `sup1_heatmap_vivit.pdf` | ViViT (fact-attn) | All 8 datasets |
| `sup1_heatmap_videomamba.pdf` | VideoMamba (SSM) | All 8 datasets |

### Analysis Figures

| Figure | File | Section | Description |
|--------|------|---------|-------------|
| **S2** | `sup2_levene_variance.pdf` | S2 | Levene variance inflation: std at stride=1 vs stride=16. Scatter plot showing instability under sparse sampling. |
| **S3** | `sup3_anova_eta2.pdf` | S3 | ANOVA effect sizes ($\eta^2$) per model. Bar chart showing coverage vs stride dominance by architecture. |
| **S4** | `sup4_taxonomy.pdf` | S4 | Action sensitivity taxonomy: empirical evidence loss (pp) per tier and dataset. |
| **S5** | `sup5_routing_all_models.pdf` | S5 | Descriptive maximum-probability cascade curves for all 8 models on SSv2. |
| **S6** | `sup6_clip_duration.pdf` | S6 | Clip duration analysis: evidence loss vs clip length. |

## Figure Usage Summary

### Main Paper
- **Total figures**: 5
- **Coverage**: original-protocol curves and evidence-loss heatmap (2), TDS/motion proxy (1), confidence cascade (1), input-scale sensitivity (1)
- **Page allocation**: ~3–4 pages for main figures

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
## TODO for Paper Finalization

- [ ] Compile manuscript.tex with pdflatex/tectonic to verify all figure paths resolve
- [ ] Compile supplementary.tex and check page breaks
- [ ] Add caption references in text (e.g., "Figure~\ref{fig:aliasing_curves}")
- [ ] Verify all figure captions match descriptions in manuscript
- [ ] Check that figure DPI is 300+ for print quality
