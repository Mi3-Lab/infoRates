# InfoRates — Research Progress & Roadmap

**ACCV 2026** · Mi3 Lab · Wesley Maia  
Last updated: 2026-05-26 — **43/49 runs complete** (7 modelos × 7 datasets) + VideoMamba (8th model, SSM, em setup)

---

## Research Purpose (não perder o fio)

**Pergunta central:** Quantos frames um modelo de vídeo realmente precisa para classificar corretamente um vídeo?

**Hipótese:** A demanda temporal varia por dataset, por classe, e por modelo. Um budget fixo desperdiça compute em vídeos simples e falha em vídeos complexos. Nosso método aloca frames de forma adaptativa por vídeo, reduzindo FLOPs sem perder acurácia.

### Três Contribuições do Paper

1. **TDS Score (Temporal Demand Score)** — métrica que quantifica o quanto um dataset depende de informação temporal. Calculado como a queda de acurácia ao reduzir frames de 32→4. Permite comparar datasets em uma escala comum.

2. **InfoRates Adaptive Router** — dado um vídeo de entrada e um budget computacional global (ex: 8 frames por vídeo em média), aloca mais frames para vídeos difíceis e menos para vídeos fáceis. Três variantes testadas:
   - **FDE Router** (Feature Diversity Estimation)
   - **Spectral Router** (frequência temporal do vídeo)
   - **Confidence Cascade** (early-exit por confiança)
   - **Knapsack Allocator** (otimização combinatorial por budget global)

3. **Cross-dataset temporal analysis** — primeiro estudo sistemático com 7 datasets × 5 arquiteturas sob budgets fixos. Mostra que a demanda temporal é uma propriedade do dataset, não do modelo.

### Claim Principal para o Paper

> *"Not all videos need 32 frames. Our method identifies per-video temporal demand and allocates frames accordingly, matching full-budget accuracy at X% of the compute on Y% of videos."*

---

## Status Atual (2026-05-26)

### Fase 1: Fine-tuning + Fixed-Budget Evaluation

**Objetivo:** Treinar todos os 7 modelos em todos os 7 datasets para ter a baseline de acurácia por budget.
Target: **7 modelos × 7 datasets = 49 runs** — **38/49 completos**

| Dataset | R3D-18 | MC3-18 | R2Plus1D | SlowFast | TSF | ViViT | VideoMAE | Status |
|---------|--------|--------|----------|----------|-----|-------|----------|--------|
| SSv2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETO** |
| UCF-101 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETO** |
| HMDB-51 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETO** |
| DriveAct | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETO** |
| Diving-48 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETO** |
| AUTSL | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🔄 | 6/7 |
| EPIC-Kitchens | 🔄* | 🔄* | 🔄 | 🔄 | ✅ | ✅ | ✅ | 3/7+2retry |

### Resultados — Top-1 por Budget

| Dataset | Modelo | 4f | 8f | 16f | 32f |
|---------|--------|---:|---:|----:|----:|
| SSv2 | R3D-18 | 9.8% | 19.7% | 37.1% | 36.9% |
| SSv2 | MC3-18 | 8.2% | 18.8% | 33.6% | 34.5% |
| SSv2 | R2Plus1D-18 | 12.6% | 24.3% | 42.6% | 42.1% |
| SSv2 | SlowFast-R50 | 6.6% | 15.2% | 33.3% | 49.5% |
| SSv2 | TimeSformer | 31.8% | 42.3% | 41.3% | 41.7% |
| SSv2 | ViViT | 8.4% | 17.5% | 30.5% | 38.3% |
| SSv2 | VideoMAE | 21.0% | 39.5% | 52.3% | 51.9% |
| UCF-101 | R3D-18 | 59.5% | 72.6% | 81.2% | 81.4% |
| UCF-101 | MC3-18 | 72.9% | 80.9% | 85.4% | 85.1% |
| UCF-101 | R2Plus1D-18 | 70.0% | 81.6% | 88.6% | 89.0% |
| UCF-101 | SlowFast-R50 | 50.1% | 66.2% | 81.3% | 87.6% |
| UCF-101 | TimeSformer | 90.0% | 91.0% | 91.2% | 90.9% |
| UCF-101 | ViViT | 75.3% | 86.9% | 92.5% | 94.3% |
| UCF-101 | VideoMAE | 81.4% | 91.4% | 95.4% | 95.5% |
| HMDB-51 | R3D-18 | 49.2% | 67.1% | 80.3% | 80.1% |
| HMDB-51 | MC3-18 | 63.5% | 71.2% | 78.6% | 78.2% |
| HMDB-51 | R2Plus1D-18 | 46.2% | 63.2% | 73.1% | 74.6% |
| HMDB-51 | SlowFast-R50 | 35.1% | 44.7% | 65.1% | 79.3% |
| HMDB-51 | TimeSformer | 73.0% | 79.9% | 80.0% | 79.8% |
| HMDB-51 | ViViT | 52.4% | 66.1% | 75.4% | 80.2% |
| HMDB-51 | VideoMAE | 51.5% | 73.6% | 84.0% | 84.4% |
| DriveAct | R3D-18 | 47.8% | 56.2% | 68.3% | 67.2% |
| DriveAct | MC3-18 | 55.1% | 65.8% | 69.0% | 68.5% |
| DriveAct | R2Plus1D-18 | 37.7% | 49.8% | 62.5% | 61.8% |
| DriveAct | SlowFast-R50 | 42.6% | 53.3% | 66.7% | 72.5% |
| DriveAct | TimeSformer | 64.7% | 67.6% | 68.8% | 66.5% |
| DriveAct | ViViT | 48.9% | 55.8% | 62.5% | 67.4% |
| DriveAct | VideoMAE | 40.2% | 56.0% | 74.1% | 72.5% |
| Diving-48 | R3D-18 | 5.9% | 14.4% | 28.8% | 28.8% |
| Diving-48 | MC3-18 | 8.2% | 19.5% | 31.6% | 33.4% |
| Diving-48 | R2Plus1D-18 | 8.6% | 16.8% | 35.3% | 34.7% |
| Diving-48 | SlowFast-R50 | 5.8% | 14.5% | 26.4% | 50.5% |
| Diving-48 | TimeSformer | 23.6% | 38.0% | 36.9% | 38.0% |
| Diving-48 | ViViT | 7.9% | 19.9% | 35.1% | 53.0% |
| Diving-48 | VideoMAE | 8.6% | 27.6% | 48.6% | 49.9% |
| AUTSL | R3D-18 | 4.7% | 24.5% | 75.0% | 74.4% |
| AUTSL | MC3-18 | 4.1% | 37.5% | 63.7% | 63.7% |
| AUTSL | R2Plus1D-18 | 8.4% | 30.2% | 75.9% | 75.0% |
| AUTSL | SlowFast-R50 | 1.6% | 12.7% | 41.8% | 82.3% |
| AUTSL | TimeSformer | 52.0% | 66.8% | 66.2% | 67.0% |
| AUTSL | ViViT | 8.4% | 25.5% | 61.2% | 74.6% |
| AUTSL | VideoMAE | — | — | — | — |
| EPIC-Kitchens | R3D-18 | — | — | — | — |
| EPIC-Kitchens | MC3-18 | — | — | — | — |
| EPIC-Kitchens | R2Plus1D-18 | — | — | — | — |
| EPIC-Kitchens | SlowFast-R50 | — | — | — | — |
| EPIC-Kitchens | TimeSformer | 22.5% | 37.4% | 37.4% | 36.1% |
| EPIC-Kitchens | ViViT | 11.7% | 23.2% | 36.2% | 40.2% |
| EPIC-Kitchens | VideoMAE | 20.4% | 40.0% | 78.9% | 74.2% |

### Achados Relevantes

- **TimeSformer satura rápido:** HMDB-51 73%→80% em só 8f. UCF-101 já 90% com 4f. critical_frame_budget = 4–8f.
- **SlowFast precisa de muitos frames:** HMDB-51 35%→79% (4→32f), Diving-48 5.8%→50.5%. Alta demanda temporal.
- **AUTSL (sign language):** jump brutal 24.5% (8f) → 75.0% (16f) → plateau. Precisa de janela temporal mínima de 16f.
- **Diving-48 é o dataset mais exigente:** ViViT 7.9%→53.0%, SlowFast 5.8%→50.5%. Enorme ganho com mais frames.
- **SSv2 vs UCF-101:** SSv2 TDS muito maior — motions sutis vs. ações grosseiras — confirma hipótese.
- **EPIC-Kitchens (resultados limpos após fix de data leakage):** TimeSformer 22.5%→37.4% (plateau imediato), VideoMAE 20.4%→78.9% (16f), ViViT 11.7%→40.2%. Os valores anteriores (TSF 83.6%) eram inflados por data leakage.
- **AUTSL (sign language, 226 classes):** Demanda temporal extrema. SlowFast 1.6%→82.3% (4→32f). R3D-18 4.7%→75.0% apenas com 16f. TimeSformer excepcionalmente bom a 4f (52%), mas plateau em 67% — provavelmente aprende padrões de postura, não sequência temporal.
- **Latência:** CNN (R3D/MC3) são 6-25x mais rápidas que Transformers. MC3-18: ~1.6ms/sample; ViViT: ~41ms/sample. TimeSformer domina eficiência em budget baixo (4-8f), CNN dominam em budget alto (16-32f). Arquivo: `evaluations/accv2026/paper_results/latency_summary.csv`.

### Jobs Rodando no Cluster (2026-05-26 ~22:00)

| Job ID | Partição | Dataset | Modelos | Status |
|--------|----------|---------|---------|--------|
| 71704 | A100 (gpu) | epic_kitchens | slowfast_r50 (epoch 6/10) + r2plus1d_18 pendente | running |
| 71788 | A100 (gpu) | epic_kitchens | r3d_18 (resume ep4→5+) + mc3_18 + r2plus1d_18 | running (retry com prefetch_factor=2 fix) |
| 71753 | H200 (cenvalarc.gpu) | autsl | videomae (epoch 1/10) | running |
| 71789 | H200 (cenvalarc.gpu) | — | build mamba-ssm para VideoMamba | pending |

**Completos nesta sessão:**
- ✅ EPIC-Kitchens: TimeSformer + ViViT + VideoMAE (todos H200, resultados limpos após fix de data leakage)

**VideoMamba (8th model — SSM):**
- `.venv_mamba` criado com PyTorch 2.8.0+cu128
- Código: `third_party/videomamba_repo/` + pesos K400 em `fine_tuned_models/videomamba_pretrained/`
- Scripts: `train_videomamba.py`, `run_h200_multidata_videomamba.sh`, `slurm_h200_videomamba_all_datasets.sbatch`
- Build job 71789: instalando `causal-conv1d` + `mamba-ssm` com fake-nvcc (workaround CUDA 13.x vs 12.8)
- Smoke test → treinamento completo pendente (aguarda build bem-sucedido)

**Bug fix nesta sessão:** `train_torchvision.py` prefetch_factor 4→2 (fix DataLoader worker OOM em EPIC-Kitchens)

---

## Fase 2: Análises Pós-Treinamento (após todos os jobs terminarem)

```bash
bash scripts/accv2026/run_post_completion_analyses.sh
```

### Checklist

- [ ] `02_make_manifests.py` — manifests de avaliação (20 amostras/classe)
- [ ] `04_compute_temporal_demand.py` — TDS score por dataset
- [ ] `05_compute_temporal_metrics.py` — AUC da curva budget×acurácia, critical_frame_budget
- [ ] `07_dataset_temporal_demand.py` — ranking de datasets por demanda temporal
- [ ] `08_compile_paper_results.py` — tabelas do paper (Table 1, Table 2)
- [ ] `09_plot_paper_figures.py` — figuras 1–9 do paper
- [ ] `10_per_class_temporal_analysis.py` — análise por classe (quais classes precisam de mais frames?)
- [ ] `06_fde_adaptive_routing.py` — avalia o FDE router vs. fixed budget
- [ ] `11_spectral_router.py` — router baseado em frequência temporal
- [ ] `12_confidence_cascade.py` — early-exit por confiança do modelo
- [ ] `13_knapsack_confidence.py` — alocação ótima de frames por budget global
- [ ] `14_plot_routing_comparison.py` — comparação entre os 4 routers
- [ ] `15_baseline_comparison.py` — tabela multi-dataset vs. literatura

---

## Fase 3: Paper Writing

### Estrutura do Paper (ACCV 2026 format — 14 páginas)

```
1. Introduction          — motivação, claim, preview do resultado principal
2. Related Work          — adaptive inference, early-exit, temporal modeling
3. Temporal Demand Score — definição formal do TDS, propriedades
4. InfoRates Method      — os 4 routers + unified framework
5. Experiments
   5.1 Setup             — 7 datasets, 7 modelos, budgets 4/8/16/32
   5.2 Fixed-budget baseline — Table 1 (a preencher com resultados finais)
   5.3 TDS analysis      — Fig 1: budget curves, Fig 2: TDS ranking
   5.4 Router comparison — Fig 3: FDE vs Spectral vs Cascade vs Knapsack
   5.5 Per-class analysis — Fig 4: which classes benefit most
   5.6 Cross-model analysis — é a demanda temporal do dataset ou do modelo?
6. Conclusion
```

### Figuras Prioritárias (já geradas com dados parciais)

- `fig1`: curvas budget×acurácia por dataset (confirmar com dados finais)
- `fig2`: cross-dataset TDS ranking
- `fig9`: comparação principal — InfoRates vs. fixed 32f

### Tabelas do Paper

- **Table 1** — Fixed-budget baseline: 5 modelos × 7 datasets × 4 budgets ← **em construção**
- **Table 2** — Multi-dataset comparison vs. SOTA methods
- **Table 3** — Router comparison: FDE vs Spectral vs Cascade vs Knapsack

---

## Próximas Etapas Imediatas

1. **Aguardar jobs terminarem** (~10-15h) — feeder automático, não precisa de ação
2. **Rodar análises pós-treinamento** — `bash scripts/accv2026/run_post_completion_analyses.sh`
3. **Verificar Table 1 completa** — `evaluations/accv2026/paper_results/paper_table_main_comparison.csv`
4. **Checar AUTSL/DriveAct/EPIC nas análises de router** — scripts 12, 13, 15 têm entradas comentadas para esses datasets, descomentar após os results chegarem
5. **Analisar TDS ranking** — confirmar hipótese: Diving-48 > AUTSL > HMDB-51 > UCF-101 > SSv2 (invertido — SSv2 é mais difícil temporalmente)

---

## Perguntas em Aberto (para seção de experiments)

- O TDS score é consistente entre modelos diferentes no mesmo dataset?
- O router de confiança generaliza para datasets fora da distribuição de treino?
- Qual o tradeoff compute-vs-acurácia dos 4 routers em budget fixo global?
- A análise por classe revela alguma categoria semântica com alta demanda temporal?

---

## Infraestrutura

- **Cluster:** Mi3 Lab HPC — A100 (gpu), H200 (cenvalarc.gpu), máx 4 jobs/partição
- **Storage:** `data/` e `fine_tuned_models/` → symlinks para `/scratch/wesleyferreiramaia/infoRates/`
- **HF cache:** `/scratch/wesleyferreiramaia/hf_unified/` (consolidado, ~205GB)
- **W&B:** projeto `inforates-accv2026`
- **Deadline ACCV 2026:** verificar datas oficiais em accv2026.org
