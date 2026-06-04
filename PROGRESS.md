# InfoRates — Research Progress

**ACCV 2026** · Mi3 Lab · Wesley Maia · PI: Ross Greer (UC Merced)
Last updated: 2026-06-03

---

## Paper Status

**Title:** Temporal Aliasing in Video Action Recognition: A Cross-Architecture Analysis at Scale

**Target:** ACCV 2026 (deadline: late June 2026)

**Draft:** `paper/main.tex` + `paper/supplementary.tex` — estrutura completa, pronta para compilar.

---

## Contributions

| # | Contribution | Status | Dados |
|---|-------------|--------|-------|
| **C1** | Temporal Demand Score (TDS) | ✅ Completo | 1,400 configs · 8 modelos · 7 datasets |
| **C2** | Cross-architecture temporal aliasing | ✅ Completo | 1,400 configs · Spearman ρ=0.97 |
| **C3** | Spatial resolution robustness | ✅ Completo | 40 configs · 8 modelos × 5 resoluções · SSv2 |
| **C4** | Entropy-based adaptive routing | ✅ Completo | +4.2pp vs FrameExit · 77% routed cheaply |

---

## Experiment Status

| Experimento | Status | Resultado principal |
|-------------|--------|---------------------|
| Temporal sweep (coverage × stride) | ✅ **Completo** — 1,400/1,400 | VMamba+TSF avg 8pp; SlowFast 42pp; ViViT anomalia |
| Spectral validation (optical flow ↔ aliasing) | ✅ **Completo** | r=0.03–0.33; AUTSL r=0.285, UCF r=0.031 |
| ANOVA η² effect sizes | ✅ **Completo** | η²(stride): 0.08 SSM/TSF vs 0.35 SlowFast |
| Levene variance inflation | ✅ **Completo** | 67% pares inflam variância; VideoMAE/HMDB 2.0× |
| Action sensitivity taxonomy | ✅ **Completo** | UCF Low: −0.3pp; AUTSL all tiers >38pp |
| Spatial resolution sweep (cross-res eval) | ✅ **Completo** — SSv2, 8 modelos × 5 res | CNNs −37pp; Transformers/SSM ±6pp |
| Entropy routing (E7) | ✅ **Completo** | 42.5% SSv2 · 7.7f · +4.2pp vs FrameExit |
| Baseline comparison vs AdaFocus/AR-Net/FrameExit | ✅ **Completo** | Tabela comparação pronta |
| Clip duration analysis | ✅ **Completo** | Clips curtos aliam mais (r=−0.3 a −0.8) |
| Resolution retraining (P3) | ❌ **Removido do paper** | Dados com bug de batch_size; aguarda cluster liberar |

---

## Dados Verificados (2026-06-03)

### Temporal (sweep_summary.csv)
- 1,400 configs completas (8 × 7 × 25), nenhuma faltando
- TDS values verificados:
  - AUTSL: 58.3pp (n=7, VideoMamba excluído por colapso)
  - SSv2: 27.6pp · DriveAct: 21.9pp · Diving-48: 19.2pp
  - HMDB-51: 16.6pp · EPIC-Kitchens: 9.7pp · UCF-101: 4.9pp

### Spatial (spatial_eval.csv)
- 40 configs (8 modelos × 5 resoluções, SSv2 apenas, sem retraining)
- CNNs: queda máx −37pp (R2+1D@336px)
- Transformers+SSM: queda máx −6pp (VideoMamba@96px)
- ViViT: robusto espacialmente (−1.9pp máx) mas frágil temporalmente (+34pp)
  → fato novo e interessante para reviewers

### Routing
- TimeSformer/SSv2: 42.5% @ 7.7f médios (oracle: 47.3%)
- Média cross-architecture: 77% dos vídeos roteados para 4-frame

---

## Paper — Decisões Importantes

| Decisão | Motivo |
|---------|--------|
| Remover P3 retraining do paper | Dados com bug de batch_size (336px colapsou: MC3@autsl −43pp). Dados 96-160px OK mas incompletos sem 336px e sem todos os datasets. |
| Manter seção espacial como cross-resolution eval | 40 configs completas, limpas, padrão claro. Honesto: "on SSv2". |
| Remover nomenclatura E1/E6/E7 | Nomenclatura interna; não faz sentido para leitores. |
| Título: "Spatiotemporal" → "Temporal" | Paper é primariamente sobre aliasing temporal; espacial é uma seção secundária. |
| ViViT anomaly: frágil temporal, robusto espacial | Achado genuíno — patch tokenization + attention = invariância espacial, mas fatorização temporal = fragilidade. |

---

## Estrutura do Paper

```
main.tex (~820 linhas)
├── Abstract
├── 1. Introduction (Nyquist framing, 4 contributions)
├── 2. Related Work
├── 3. Methodology
│   ├── Datasets & Architectures (Table 1: TDS por dataset)
│   ├── Temporal Sampling Protocol (coverage × stride)
│   ├── Spatial Resolution Protocol (cross-res, sem retraining)
│   ├── TDS metric (Eq. 1)
│   ├── Statistical Analysis (ANOVA, Levene)
│   └── Entropy-Based Routing
├── 4. Results
│   ├── TDS + Spectral Validation (Fig 3, Table 2)
│   ├── Cross-Architecture Temporal Aliasing (Fig 1, Fig 2, Table 3, Table 4, Table 5)
│   ├── Statistical Validation (ANOVA + Levene)
│   ├── Action Sensitivity Taxonomy (Table 6)
│   ├── Spatial Resolution Robustness (Fig 5, Table 7) ← SSv2, 8 modelos × 5 res
│   └── Entropy Routing (Fig 4, Table 8, Table 9)
├── 5. Discussion
│   ├── Por que attention type determina aliasing?
│   ├── Spatial robustness mirrors temporal robustness
│   ├── Practical deployment guidance (3 bullets)
│   ├── The ViViT anomaly
│   ├── SSM as new robustness paradigm
│   └── Limitations
└── 6. Conclusion

supplementary.tex (~450 linhas)
├── S1: Full coverage×stride heatmaps (8 modelos × 7 datasets)
├── S2: Levene variance inflation
├── S3: Full ANOVA tables
├── S4: Action sensitivity taxonomy detail
├── S5: Entropy routing curves — all models
├── S6: Clip duration analysis
├── S7: Spectral correlation detail
└── S8: Implementation details
```

---

## Infraestrutura

- **Cluster:** Pinnacles HPC (UC Merced)
- **GPUs:** A100 (40GB, partition `gpu`) · L40s/H200 (48-141GB, partition `cenvalarc.gpu`)
- **Envs:** `.venv` (CNNs + Transformers) · `.venv_mamba` (VideoMamba)
- **W&B:** `inforates-accv2026`
- **Dashboard:** [inforates.streamlit.app](https://inforates.streamlit.app)
- **Data:** `dashboard/data/` — 8 CSVs bundled (332KB total + spatial_eval.csv)

---

## Próximos Passos

1. **Compilar PDF** — via Overleaf ou `pdflatex` local (não há compilador no cluster)
2. **Verificar figuras** — `paper/images/main_fig5_spatial_resolution.pdf` precisa refletir dados corretos (8 modelos, SSv2, sem retraining)
3. **P3 retraining** (quando cluster liberar) — rodar com `--partition=cenvalarc.gpu`, batch=64, para dados limpos; se terminar antes da deadline, pode adicionar tabela ao supplementary
4. **Polimento final** — abstract word count, page count, references check
