# InfoRates — Research Progress & Roadmap

**ACCV 2026** · Mi3 Lab · Wesley Maia  
Last updated: 2026-05-26

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

**Objetivo:** Treinar todos os 5 modelos em todos os 7 datasets para ter a baseline de acurácia por budget.

| Dataset | R3D-18 | MC3-18 | SlowFast | TimeSformer | ViViT | Status |
|---------|--------|--------|----------|-------------|-------|--------|
| HMDB-51 | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETO** |
| Diving-48 | 🔄 ep2 | 🔄 | ✅ | 🔄 ep2 | 🔄 | treinando |
| EPIC-Kitchens | 🔄 | 🔄 | 🔄 | 🔄 | 🔄 | treinando |
| AUTSL | ✅ | 🔄 ep6 | 🔄 | 🔄 ep2 | 🔄 | treinando |
| DriveAct | 🔄 | 🔄 | 🔄 | 🔄 vivit | 🔄 | treinando |
| SSv2 | ✅ (legacy) | ✅ (legacy) | ✅ (legacy) | ✅ (legacy) | ✅ (legacy) | disponível |
| UCF-101 | ✅ (legacy) | ✅ (legacy) | ✅ (legacy) | ✅ (legacy) | — | disponível |

### Resultados Disponíveis — Top-1 por Budget

| Dataset | Modelo | 4f | 8f | 16f | 32f |
|---------|--------|---:|---:|----:|----:|
| HMDB-51 | R3D-18 | 49.2% | 67.1% | 80.3% | 80.1% |
| HMDB-51 | MC3-18 | 63.5% | 71.2% | 78.6% | 78.2% |
| HMDB-51 | SlowFast-R50 | 35.1% | 44.7% | 65.1% | 79.3% |
| HMDB-51 | TimeSformer | 73.0% | 79.9% | 80.0% | 79.8% |
| HMDB-51 | ViViT | 52.4% | 66.1% | 75.4% | 80.2% |
| HMDB-51 | VideoMAE | 51.5% | 73.6% | 84.0% | 84.4% |
| UCF-101 | R3D-18 | 59.5% | 72.6% | 81.2% | 81.4% |
| UCF-101 | MC3-18 | 72.9% | 80.9% | 85.4% | 85.1% |
| UCF-101 | SlowFast-R50 | 50.1% | 66.2% | 81.3% | 87.6% |
| UCF-101 | TimeSformer | 90.0% | 91.0% | 91.2% | 90.9% |
| UCF-101 | VideoMAE | 81.4% | 91.4% | 95.4% | 95.5% |
| SSv2 | R3D-18 | 9.8% | 19.7% | 37.1% | 36.9% |
| SSv2 | MC3-18 | 8.2% | 18.8% | 33.6% | 34.5% |
| SSv2 | SlowFast-R50 | 6.6% | 15.2% | 33.3% | 49.5% |
| SSv2 | TimeSformer | 31.8% | 42.3% | 41.3% | 41.7% |
| SSv2 | ViViT | 8.4% | 17.5% | 30.5% | 38.3% |
| SSv2 | VideoMAE | 21.0% | 39.5% | 52.3% | 51.9% |
| Kinetics-400 | VideoMAE (pretrained) | 62.4% | 69.1% | 75.7% | 75.5% |
| AUTSL | R3D-18 | 4.7% | 24.5% | 75.0% | 74.4% |
| Diving-48 | SlowFast-R50 | 5.8% | 14.5% | 26.4% | 50.5% |
| Diving-48 | VideoMAE | 8.6% | 27.6% | 48.6% | 49.9% |
| EPIC-Kitchens | VideoMAE | 20.1% | 35.7% | 60.5% | 57.9% |
| DriveAct | VideoMAE | 40.2% | 56.0% | 74.1% | 72.5% |

### Achados Parciais Relevantes

- **TimeSformer satura rápido:** em HMDB-51, vai de 73% (4f) → 80% (8f) → plateau. Crítico em 8 frames.
- **SlowFast precisa de muitos frames:** 35% (4f) → 79% (32f). Alta demanda temporal.
- **AUTSL (sign language):** jump brutal de 24.5% (8f) → 75.0% (16f) → plateau. Precisa de janela temporal mínima.
- **Diving-48:** SlowFast vai de 5.8% (4f) → 50.5% (32f). Dataset mais temporalmente exigente até agora.
- **SSv2 vs UCF-101:** SSv2 tem TDS muito maior — resultado esperado (motions sutis vs. ações grosseiras).

### Jobs Rodando no Cluster

- **A100 (gpu):** diving48, epic_kitchens, autsl, driveact — todos os 3 CNNs sequenciais por job
- **H200 (cenvalarc.gpu):** diving48, epic_kitchens, autsl, driveact — TimeSformer + ViViT por job
- **Feeder daemon:** `nohup bash scripts/accv2026/feeder_submit_jobs.sh &` — submete automaticamente

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
   5.1 Setup             — 7 datasets, 5 modelos, budgets 4/8/16/32
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
