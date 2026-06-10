# InfoRates — Research Progress

**ACCV 2026** · Mi3 Lab · Wesley Maia · PI: Ross Greer (UC Merced)
Last updated: 2026-06-09

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
| **C3** | Spatial resolution robustness | ✅ Completo | 280 configs · 8 modelos × 5 res × 7 datasets (checkpoints nativos corretos) |
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
| Spatial resolution sweep (cross-res eval) | ✅ **Completo** — 280 configs · 48/48 jobs | CNNs: −73pp (R3D-18/AUTSL@336px); SSv2 −37pp; Transformers ≤7.7pp |
| Entropy routing (E7) | ✅ **Completo** | 42.5% SSv2 · 7.7f · +4.2pp vs FrameExit |
| Baseline comparison vs AdaFocus/AR-Net/FrameExit | ✅ **Completo** | Tabela comparação pronta |
| Clip duration analysis | ✅ **Completo** | Clips curtos aliam mais (r=−0.3 a −0.8) |
| Resolution retraining (96–224px, 7 datasets) | ⏳ **197/224 done** | 8 jobs running (H200+A100 daemons); ~27 remaining |
| Resolution retraining (P3/336px investigation) | ❌ **Pausado** | Padrão anômalo (batch_size bug); investiga após 96–224px terminar |

---

## Dados Verificados (2026-06-03)

### Temporal (sweep_summary.csv)
- 1,400 configs completas (8 × 7 × 25), nenhuma faltando
- TDS values verificados:
  - AUTSL: 58.3pp (n=7, VideoMamba excluído por colapso)
  - SSv2: 27.6pp · DriveAct: 21.9pp · Diving-48: 19.2pp
  - HMDB-51: 16.6pp · EPIC-Kitchens: 9.7pp · UCF-101: 4.9pp

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
└── S9: Multi-dataset spatial resolution results (7 datasets × 8 modelos × 5 res)
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

### Curto prazo (próx. 1-2 semanas):
1. **Compilar PDF** — via Overleaf ou `pdflatex` local (não há compilador no cluster)
2. **Verificar figuras** — `paper/images/main_fig5_spatial_resolution.pdf` reflete dados corretos (8 modelos, SSv2, sem retraining)
3. **Polimento draft** — abstract word count, page count, references check

### Médio prazo (quando cluster liberar):
4. **P3 retraining 336px** — rodar com `--partition=cenvalarc.gpu`, batch=64, daemon pronto em `scripts/accv2026/submit_p3_336px_2gpu.sh`
5. **Começar processamento FLAME** — dataset menor, prioridade 1 para autonomous evacuation validation

### Longo prazo (pós-deadline ACCV):
6. **Ego4D + UCF-Crime** — processar em paralelo enquanto paper está em review
7. **Atualizar paper com 10 datasets** — versão estendida para journal (aceitação esperada ACCV → versão journal com datasets extras)
