# InfoRates — Research Progress & Roadmap

**ACCV 2026** · Mi3 Lab · Wesley Maia  
Last updated: 2026-05-27 (tarde) — **46/56 runs complete** (7 modelos × 7 datasets + VideoMamba 8 datasets)

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

## Status Atual (2026-05-27 tarde)

### Fase 1: Fine-tuning + Fixed-Budget Evaluation

**Objetivo:** Treinar todos os 7 modelos em todos os 7 datasets (+ VideoMamba em 8 datasets).
Target: **7 modelos × 7 datasets = 49 runs** + **VideoMamba × 8 = 8 runs**

| Dataset | R3D-18 | MC3-18 | R2Plus1D | SlowFast | TSF | ViViT | VideoMAE | VideoMamba | Status |
|---------|--------|--------|----------|----------|-----|-------|----------|------------|--------|
| SSv2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🔄 eval (job 72616) | 7/8 |
| UCF-101 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETO** |
| HMDB-51 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETO** |
| DriveAct | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETO** |
| Diving-48 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETO** |
| AUTSL | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 7/8 (VideoMamba não converge — feature collapse K400→ASL) |
| EPIC-Kitchens | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETO** |

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
| AUTSL | VideoMAE | 17.6% | 43.2% | 79.5% | 78.9% |
| EPIC-Kitchens | R3D-18 | 13.6% | 22.3% | 37.2% | 37.0% |
| EPIC-Kitchens | MC3-18 | 11.3% | 27.1% | 36.2% | 37.2% |
| EPIC-Kitchens | R2Plus1D-18 | 13.0% | 20.2% | 35.5% | 35.2% |
| EPIC-Kitchens | SlowFast-R50 | 9.2% | 15.8% | 27.2% | 39.4% |
| EPIC-Kitchens | TimeSformer | 19.5% | 32.3% | 31.5% | 31.0% |
| EPIC-Kitchens | ViViT | 10.3% | 21.1% | 26.9% | 32.9% |
| EPIC-Kitchens | VideoMAE | 13.4% | 28.3% | 37.7% | 37.5% |
| EPIC-Kitchens | VideoMamba | 23.2% | 28.3% | 28.2% | 28.4% |

### Achados Relevantes

- **TimeSformer satura rápido:** HMDB-51 73%→80% em só 8f. UCF-101 já 90% com 4f. critical_frame_budget = 4–8f.
- **SlowFast precisa de muitos frames:** HMDB-51 35%→79% (4→32f), Diving-48 5.8%→50.5%. Alta demanda temporal — comportamento arquitetural (model_frames=32).
- **AUTSL (sign language):** jump brutal 24.5% (8f) → 75.0% (16f) → plateau. Janela temporal mínima de 16f obrigatória.
- **Diving-48 é o dataset mais exigente:** ViViT 7.9%→53.0%, SlowFast 5.8%→50.5%. Enorme ganho com mais frames.
- **SSv2 vs UCF-101:** SSv2 TDS muito maior — motions sutis vs. ações grosseiras — confirma hipótese.
- **EPIC-Kitchens (split limpo):** Todos os modelos convergem para ~35-37% em 16f (R3D-18 37.2%, VideoMAE 37.7%, TimeSformer 31.5%). VideoMAE NÃO é superior em EPIC — resultado anterior de 78% era inflado por data leakage. Dataset uniformemente difícil para todos os modelos.
- **VideoMamba plateau após 8f:** comportamento arquitetural — model_frames=8, budgets maiores são subsampled de volta a 8f de posições temporais diferentes.
- **Latência:** CNN (R3D/MC3) são 6-25x mais rápidas que Transformers. MC3-18: ~1.6ms/sample; ViViT: ~41ms/sample. TimeSformer domina eficiência em budget baixo (4-8f), CNN dominam em budget alto (16-32f). Arquivo: `evaluations/accv2026/paper_results/latency_summary.csv`.

### Jobs Rodando no Cluster (2026-05-28)

| Job ID | Partição | Dataset/Modelo | Status |
|--------|----------|----------------|--------|
| 72616 | H200 (cenvalarc.gpu) | VideoMamba SSV2 eval (fix dataset-name bug) | rodando |

**Completos (2026-05-27 a 2026-05-28):**
- ✅ R3D-18 EPIC (split limpo): 13.6/22.3/37.2/37.0%
- ✅ MC3-18 EPIC (split limpo): 11.3/27.1/36.2/37.2%
- ✅ R2Plus1D-18 EPIC (split limpo, resize fix 224→112): 13.0/20.2/35.5/35.2%
- ✅ TimeSformer EPIC (retrain clean): 19.5/32.3/31.5/31.0%
- ✅ ViViT EPIC (retrain clean): 10.3/21.1/26.9/32.9%
- ✅ SlowFast-R50 EPIC (retrain clean): 9.2/15.8/27.2/39.4%
- ✅ VideoMAE EPIC (retrain clean): 13.4/28.3/37.7/37.5%
- ✅ VideoMamba SSV2 treino: best val_acc=52.2% (ep5), eval com bug (job 72616 corrige)
- ❌ VideoMamba AUTSL: não converge — feature collapse K400→língua de sinais (raw pixel std 10× menor que UCF-101)

**Bugs encontrados e corrigidos:**
- EPIC split contaminado (train/val leak) → corrigido 2026-05-27 01:09; modelos contaminados retreinados
- R2Plus1D eval com model-frames=8 (deveria ser 16) → eval refeita
- R2Plus1D eval com resize=224 (treinado em 112px) → eval refeita com resize=112
- VideoMAE EPIC resultado ~78% era inflado por split contaminado → resultado real: ~37.7%

**Anomalia aberta:**
- VideoMamba AUTSL: loss=ln(226) em todos os epochs com qualquer LR. Hipótese: `decord` retorna frames incorretos para vídeos AUTSL (codec incompatível). Todos os outros 6 datasets funcionam normalmente.

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

1. **Aguardar SlowFast + ViViT EPIC** (jobs 72182, 72220) — ~4h restantes
2. **Investigar VideoMamba AUTSL** — verificar se `decord` decodifica os vídeos AUTSL corretamente (testar carregar 1 clip manualmente em `.venv_mamba`)
3. **Rodar análises pós-treinamento** assim que EPIC estiver completo — `bash scripts/accv2026/run_post_completion_analyses.sh`
4. **Analisar TDS ranking** — confirmar hipótese: Diving-48 > AUTSL > SSv2 > HMDB-51 > UCF-101

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
