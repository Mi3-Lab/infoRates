# InfoRates — Research Progress

**ACCV 2026** · Mi3 Lab · Wesley Maia · PI: Ross Greer (UC Merced)
Last updated: 2026-06-19

---

## Paper Status

**Title:** Temporal Aliasing in Video Action Recognition: A Cross-Architecture Analysis at Scale

**Target:** ACCV 2026 (deadline: late June 2026)

**Draft:** `paper/main.tex` + `paper/supplementary.tex` — estrutura completa, pronta para compilar.

---

## Contributions

| # | Contribution | Status | Dados |
|---|-------------|--------|-------|
| **C1** | Temporal Demand Score (TDS) | ✅ Completo | 1,600 configs · 8 modelos · 8 datasets (+ FineGym) |
| **C2** | Cross-architecture temporal aliasing | ✅ Completo | 1,600 configs · Spearman ρ=0.97 |
| **C3** | Spatial resolution robustness | ✅ Completo | 280 configs · 8 modelos × 5 res × 7 datasets (checkpoints nativos corretos) |
| **C4** | Entropy-based adaptive routing | ✅ Completo | +4.2pp vs FrameExit · 77% routed cheaply (8 datasets) |

---

## Experiment Status

| Experimento | Status | Resultado principal |
|-------------|--------|---------------------|
| Temporal sweep (coverage × stride) | ✅ **Completo** — 1,600/1,600 | VMamba+TSF avg 8pp; SlowFast 42pp; ViViT anomalia; FineGym 58.1pp |
| Spectral validation v1 (7 datasets) | ✅ **Completo** | r=0.03–0.33; AUTSL r=0.285, UCF r=0.031 (underpowered, n=7) |
| Spectral validation v2 (7 datasets × 5 res = 35 pts) | ✅ **Completo** | Spearman ρ=−0.549, p=0.0006 — inversion: higher flow freq → lower TDS |
| ANOVA η² effect sizes | ✅ **Completo** | η²(stride): 0.08 SSM/TSF vs 0.35 SlowFast |
| Levene variance inflation | ✅ **Completo** | 67% pares inflam variância; VideoMAE/HMDB 2.0× |
| Action sensitivity taxonomy | ✅ **Completo** | UCF Low: −0.3pp; AUTSL all tiers >38pp |
| Spatial resolution sweep (cross-res eval) | ✅ **Completo** — 308 configs · 56/56 jobs (incl. 48px + FineGym) | CNNs: −73pp (R3D-18/AUTSL); Transformers ≤7.7pp; 48px adicionado |
| Entropy routing (E7) | ✅ **Completo** | 42.5% SSv2 · 7.7f · +4.2pp vs FrameExit |
| Baseline comparison vs AdaFocus/AR-Net/FrameExit | ✅ **Completo** | Tabela comparação pronta |
| Clip duration analysis | ✅ **Completo** | Clips curtos aliam mais (r=−0.3 a −0.8) |
| Resolution retraining (96–224px, 7 datasets) | ✅ **256/280 done** | 256 checkpoints 96–224px prontos; 24 faltando = todos 336px |
| Stride×Coverage×Resolution sweep (C3b) | ✅ **Completo** | VideoMamba/EK fix aplicado (checkpoint leakage removido); job 148039 OK |
| Anomaly retraining campaign | ⏳ **Em progresso** | Ver tabela abaixo |
| TDS robustness analysis | ✅ **Completo** | Family ablation CNN-only ρ=0.976, Transformer-only ρ=1.000; bootstrap CI FineGym−AUTSL gap [−4.6, +11.7]pp |
| VideoMAE PE bug fix (model_factory.py) | ✅ **Corrigido** | PE sinusoidal era tensor não-parameter → não salvo; fix: carrega HF base + interpola bicubic |
| Resolution retraining (P3/336px investigation) | ❌ **Pausado** | Padrão anômalo (batch_size bug); investiga após sweep C3b terminar |

---

## Anomalias Identificadas e Corrigidas (2026-06-14)

### Root Cause: VideoMAE PE Bug
`position_embeddings` no VideoMAE é stored como tensor simples (não `nn.Parameter`) →
não salvo no `model.safetensors`. Antes do fix, training e inference usavam PE aleatória
nas resoluções sub-nativas. Fix: carrega base HF model, interpolação bicúbica do PE
224px → target res, depois overlay os pesos do checkpoint. Aplicado em:
- `scripts/accv2026/sweep_coverage_stride.py` (inference)
- `src/info_rates/models/model_factory.py` (training)

### Checkpoints com retraining em andamento:
| Modelo | Dataset | Res | Problema | Status |
|--------|---------|-----|----------|--------|
| slowfast_r50 | autsl | 96px | Collapse (LR 1e-4 > 5e-5 ideal) | RUNNING |
| slowfast_r50 | autsl | 112px | Collapse | RUNNING |
| slowfast_r50 | ssv2 | 96px | Collapse | RUNNING |
| slowfast_r50 | epic_kitchens | 96px | Collapse | RUNNING |
| slowfast_r50 | epic_kitchens | 48px | Collapse | daemon queue |
| mc3_18 | driveact | 48px | Checkpoint curto | daemon queue |
| mc3_18 | epic_kitchens | 160px | Checkpoint curto | daemon queue |
| timesformer | autsl | 48px | PE mismatch (9 tokens, collapse) | daemon queue |
| timesformer | diving48 | 48px | PE mismatch (9 tokens, collapse) | daemon queue |
| vivit | autsl | 48px | PE mismatch (9 tokens, collapse) | daemon queue |
| videomae | ssv2 | 96px | PE mismatch durante training | daemon queue |
| videomae | ssv2 | 112px | PE mismatch durante training | daemon queue |
| videomae | ssv2 | 160px | PE mismatch durante training | daemon queue |

### Anomalias menores (não corrigindo):
- VideoMAE driveact@160px (75.9%) > @224px (73.0%): checkpoint v2 em 160px tem val_acc superior ao v1-only em 224px
- VideoMAE EK@160px (42.6%) > @224px (38.6%): mesmo motivo
- ViViT SSV2@96-160px plateau (~23%): SSV2 genuinamente difícil para ViViT < 224px
- VideoMamba autsl/diving48@48px collapse: limite genuíno (9 SSM tokens p/ 226 classes)

---

## Dados Verificados (2026-06-10)

### Temporal (sweep_summary.csv)
- 1,600 configs completas (8 × 8 × 25), nenhuma faltando
- TDS values verificados:
  - FineGym: **58.1pp** (n=8, mean; alta demanda temporal — ginástica) ← NOVO
  - AUTSL: 58.3pp (n=7, VideoMamba excluído por colapso)
  - SSv2: 27.6pp · DriveAct: 21.9pp · Diving-48: 19.2pp
  - HMDB-51: 16.6pp · EPIC-Kitchens: 9.7pp · UCF-101: 4.9pp
- FineGym sweep: 200/200 configs, 8 modelos × 25 configs
  - R3D-18: 71.9%→11.2% (-60.7pp) · MC3: 74.6%→11.2% (-63.4pp)
  - R(2+1)D: 78.4%→12.3% (-66.1pp) · SlowFast: 79.9%→7.6% (-72.2pp)
  - TimeSformer: 66.1%→29.7% (-36.4pp) · ViViT: 69.1%→13.6% (-55.6pp)
  - VideoMAE: 77.7%→9.6% (-68.1pp) · VideoMamba: 72.7%→30.6% (-42.1pp)

### Spatial (p3_results.csv — 280 linhas, checkpoints nativos corretos)
- 48/48 jobs completos (VideoMamba/AUTSL = colapso confirmado: 0.4% em todas as resoluções)
- **SSv2 (motion-heavy):** CNNs −37pp (R2+1D@336px); Transformers ≤6pp
- **Cross-dataset — padrão MAIS SEVERO em datasets appearance-heavy:**
  - R3D-18/AUTSL: 75.0% → 2.0% (−73.0pp) — pior drop de todos
  - MC3-18/DriveAct: 69.0% → 4.5% (−64.5pp)
  - R2+1D/AUTSL: 75.9% → 7.3% (−68.7pp)
  - UCF-101 CNNs: peak ~81-88% @ 112px, colapso a 20-25% @ 336px (−60-64pp)
- **Transformers:** VideoMAE ≤7.7pp de variação em qualquer dataset (EPIC-Kitchens pior)
- Bug corrigido: script estava priorizando checkpoint 224px-retreinado para CNNs; corrigido para usar `full_e10_a100` (nativo 112px)
- Dados em `dashboard/data/p3_results.csv`; tabelas completas em supplementary S9
- ViViT: robusto espacialmente (−1.9pp SSv2) mas frágil temporalmente (+34pp)

### Routing
- TimeSformer/SSv2: 42.5% @ 7.7f médios (oracle: 47.3%)
- Média cross-architecture: 77% dos vídeos roteados para 4-frame

### Dataset Expansion (Status 2026-06-10)
- **FLAME:** ✅ Downloaded and unzipped to `data/FLAME_data/`
- **UCF-Crime:** ✅ Downloaded and unzipped to `data/UCFCrime_data/`
- **Ego4D:** ⏳ Downloading — Slurm job `long` partition, **12 GB** (288 clips subset, top 50 verbs)
  - FHO-LTA: 50 verb classes, balanced subset (~50 train + 20 val per class)
  - After download: `preprocess_ego4d.py` extracts 3–8s segments → `action_clips/{verb}/{clip_uid}_a{idx}.mp4`
  - Training daemon: `submit_ego4d_retrain.sh` (waits for manifest, then 8×4=32 jobs)

---

## Paper — Decisões Importantes

| Decisão | Motivo |
|---------|--------|
| Remover P3 retraining do paper | Dados com bug de batch_size (336px colapsou: MC3@autsl −43pp). Dados 96-160px OK mas incompletos sem 336px e sem todos os datasets. |
| Manter seção espacial como cross-resolution eval | 280 configs completas (8×5×7), padrão claro e universal. CNNs: até −73pp; Transformers: ≤8pp. |
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
│   ├── Spatial Resolution Robustness (Fig 5, Table 7) ← SSv2 + cross-dataset validation (S9)
│   └── Entropy Routing (Fig 4, Table 8, Table 9)
├── 5. Discussion
│   ├── Por que attention type determina aliasing?
│   ├── Spatial robustness mirrors temporal (+ amplified on appearance-heavy domains)
│   ├── Practical deployment guidance (4 bullets)
│   ├── The ViViT anomaly
│   ├── SSM as new robustness paradigm
│   └── Limitations
└── 6. Conclusion

supplementary.tex (~630 linhas)
├── S1: Full coverage×stride heatmaps (8 modelos × 7 datasets)
├── S2: Levene variance inflation
├── S3: Full ANOVA tables
├── S4: Action sensitivity taxonomy detail
├── S5: Entropy routing curves — all models
├── S6: Clip duration analysis
├── S7: Spectral correlation detail
├── S8: Implementation details
├── S9: Multi-dataset spatial resolution results (7 datasets × 8 modelos × 5 res)
├── S10: Best model per dataset breakdown
├── S11: TDS robustness — leave-one-out + family ablation + bootstrap CI
└── S12: Direct spectral test of Nyquist framing (n=35, ρ=−0.549, p=0.0006)
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

## Dataset Expansion — 3 adicionais recomendados (2026-06-03)

**Decisão:** Expandir de 7 para 10 datasets para validar **TDS como propriedade universal** e testar **aplicações práticas** (autonomous evacuation systems).

| Dataset | Domínio | TDS esperado | Aplicação | Tamanho |
|---------|---------|--------------|-----------|---------|
| **FLAME** | Emergency (fire/smoke) | ~70pp | Safety-critical: automatic vehicle evacuation | ~1.9K vídeos |
| **Ego4D** | Egocentric long-form | ~15pp | Episodic memory: SSM recurrence under stress | ~1.4M vídeos |
| **UCF-Crime** | Surveillance anomalies | ~50pp | Real-world deployment: edge cameras | ~1.9K vídeos |

**Por quê estes 3:**
- **FLAME:** Stress-test condições extremas (visibilidade <10%); valida se CNN/Transformer split mantém em degradação severa
- **Ego4D:** Vídeos muito longos (30+ min) força VideoMamba (SSM) a provar vantagem episódic memory real
- **UCF-Crime:** Vigilância real com câmeras de baixo FPS — valida TDS em deployment crítico

**Narrativa unificada do paper:**
- **C1 (TDS):** Validado cross 10 datasets, confirmando que é propriedade dataset, não arquitetura
- **C2 (Temporal aliasing):** Atenção-type governs aliasing até em condições extremas (FLAME)
- **C3 (Spatial robustness):** Complementa validação cross-domain
- **C4 (Routing):** Adaptativo routing em cenários críticos (surveillance, emergency)

**Timeline:** ~2 meses de processamento (paralelo com escrita + compilation)

**Impact:** 10 datasets + 8 architectures = narrativa muito mais forte para ACCV

---

## Próximos Passos (Prioridade)

### Imediato (pré-submissão):
1. **Compilar PDF** — via Overleaf ou `pdflatex` no PC com LaTeX instalado (sem compilador no cluster)
2. **FineGym spectral (outro PC)** — rodar `scripts/accv2026/analyze_nyquist_finegym_flowonly.py` onde os vídeos FineGym estão acessíveis; copiar `evaluations/accv2026/e3_spectral/finegym_cutoff_freq.csv` de volta; re-merge → atualiza S12 com n=40 e p-value recalculado
3. **Polimento final** — contagem de palavras abstract, page count, verificar referências

### Médio prazo (quando cluster liberar):
4. **P3 retraining 336px** — daemon `scripts/accv2026/submit_p3_336px_2gpu.sh`, fix de path `/scratch/` → já planejado em `slurm_336px_2gpu.sbatch`
5. **Verificar figuras** — `paper/images/` confirmar que todas as figuras refletem dados finais (sem retraining p3, sem 336px)
