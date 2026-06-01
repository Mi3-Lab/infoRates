# ACCV 2026 Paper Submission

## Files Organization

### 📄 Main Submission
- **`main.tex`** ← **USE THIS FOR SUBMISSION** (603 lines, ACCV official format)
  - Complete paper with 6 sections
  - ACCV 2026 official packages and formatting
  - TODO: Replace `ID=*****` on line 8 with your submission ID
  - Bibliography: `main.bib` (45 references, splncs04.bst style)

### 📑 Supplementary Material  
- **`supplementary.tex`** ← **USE THIS FOR SUPPLEMENTARY SUBMISSION** (25KB, 9 sections)
  - S1: Coverage×Stride heatmaps (8 figures, all models/datasets)
  - S2: Levene variance inflation analysis
  - S3: Full ANOVA effect size tables
  - S4: Action sensitivity taxonomy
  - S5: E7 entropy routing curves
  - S6: Clip duration analysis
  - S7: P3 resolution retraining results
  - S8: Spectral correlation detail
  - S9: Implementation details

### 🖼️ Figures
- **`images/`** directory (16 PDF figures + PNG 300dpi versions)
  - `main_fig*.pdf` (3 figures for main paper)
  - `sup*.pdf` (13 figures for supplementary)
  - `FIGURES_INDEX.md` — figure-to-section mapping

### 📚 Bibliography
- **`main.bib`** (9.1KB, 45 references)
  - All architectures: R3D, MC3, R2+1D, SlowFast, TimeSformer, ViViT, VideoMAE, VideoMamba
  - All 7 datasets
  - Video SSMs, adaptive methods, signal processing theory
  - Use with `\bibliographystyle{splncs04}`

### 📋 Reference Documents
- **`manuscript.tex`** — Development version (for reference only)
- **`FIGURES_INDEX.md`** — Figure-to-section cross-reference

---

## Submission Workflow

### Step 1: Prepare for Review
```bash
# In the paper directory:
pdflatex main.tex
bibtex main
pdflatex main
pdflatex main
```

Before submission, ensure:
- Line 8: Keep `\usepackage[review,year=2026,ID=*****]{accv}` 
- Replace `*****` with your submission ID when you receive it
- Line numbering will appear automatically (review version feature)
- Authors anonymized (no affiliation in \institute{})

### Step 2: Submit to ACCV
1. `main.pdf` — Main paper
2. `main.bib` — Bibliography file
3. `supplementary.pdf` — Supplementary material (compile separately)
4. `images/` folder or zipped file with all figures

### Step 3: Camera-Ready Version (if accepted)
```bash
# Comment line 8:
% \usepackage[review,year=2026,ID=*****]{accv}

# Uncomment line 10:
\usepackage{accv}
```
Then recompile. Line numbers will disappear, camera-ready formatting applies.

---

## Important Notes

### Title
- **Main:** "Spatiotemporal Aliasing in Video Action Recognition: A Cross-Architecture Analysis at Scale"
- **Running head:** "Spatiotemporal Aliasing in Video Action Recognition"

### Abstract
- Concise summary of 8 architectures, 7 datasets, 1,400 configs
- Key findings: TDS metric, attention type determines aliasing, spatial aliasing is training artifact, E7 routing
- Keywords: Temporal Aliasing, Spatial Aliasing, Video Action Recognition, Cross-Architecture Analysis, Temporal Demand Score, Adaptive Routing

### Page Limit
- Main paper: 14 pages (excluding references)
- Supplementary: 20+ pages (references/supplementary not counted toward limit)
- Current: ~12–14 pages LNCS format (optimal)

### Figures & Tables
- All figures included in `images/`
- Figure paths in `main.tex` use relative references: `images/main_fig*.pdf`
- All captions follow ACCV style (under figures, above tables)

### References
- 45 total references with DOIs where available
- Format: LNCS style (splncs04.bst)
- Cross-citations: Use `\cite{}` and `\cref{}` for figures/tables

---

## Compilation Notes

### Requirements
- LaTeX with `accv` package (comes with template)
- `accvabbrv` package
- `splncs04.bst` bibliography style
- Python 3 + matplotlib (for figure generation, already done)

### Optional
- `hyperref` with `pagebackref` — helps reviewers navigate
- `orcidlink` — for author OrcID links (camera-ready)
- `axessibility` — improves PDF accessibility

### Troubleshooting
- If figures don't appear: Check `images/` path is relative and PDFs exist
- If references missing: Run `bibtex main` before second `pdflatex`
- If hyperref conflicts: Use `pagebackref` only in review version (comment in camera-ready)

---

## Content Checklist

- ✅ 6 sections (Intro, Related Work, Methods, Results, Discussion, Conclusion)
- ✅ 3 main figures (TDS ranking, aliasing curves, spatial resolution)
- ✅ 2 main tables (datasets summary, E1 aliasing results)
- ✅ Abstract with keywords
- ✅ 45 references with DOIs
- ✅ Supplementary material with 13 additional figures
- ✅ Supplementary material with 6 additional tables
- ✅ All implementation details documented
- ✅ Anonymous submission format

---

**Last updated:** 2026-06-01  
**Status:** Ready for ACCV 2026 submission
