# E3 — Nyquist Temporal Aliasing Validation
## Relatório Completo de Experimentos e Resultados

**Data de execução:** 21 de junho de 2026  
**Cluster:** cenvalarc.gpu (A100/H200), Slurm  
**Objetivo:** Validar empiricamente que classes com evidência temporal localizada sofrem mais aliasing sob amostragem esparsa (framing Nyquist-Shannon para reconhecimento de ações em vídeo).

---

## 1. Hipótese Central

Dado que um modelo treinado com stride=1 (16 frames densamente amostrados) tem acurácia `acc_s1` e o mesmo modelo com stride=16 (amostrado esparso) tem `acc_s16`, definimos:

```
aliasing_drop(c) = acc_s1(c) - acc_s16(c)   [mean_abs_drop na taxonomy]
```

A hipótese é: **classes onde o modelo fica confiante apenas em janelas específicas do vídeo (evidência temporalmente localizada) devem ter maior aliasing**, porque o subsampling esparso tem maior probabilidade de perder a janela crítica.

**Preditor proposto:** concentração da confiança do modelo ao deslizar uma janela de T frames pelo vídeo completo.

---

## 2. Dados de Entrada

### 2.1 Taxonomy (aliasing por classe)

**Localização:** `evaluations/accv2026/e5_taxonomy/{dataset}_class_taxonomy.csv`

Coluna-chave: `mean_abs_drop` — drop médio de acurácia entre stride=1 e stride=16, calculado sobre 8 modelos. Disponível para 8 datasets:

| Dataset | Classes | Domínio |
|---------|---------|---------|
| UCF-101 | 101 | Ações genéricas (câmera fixa) |
| SSv2 | 174 | Ações mão-objeto (câmera fixa) |
| HMDB-51 | 51 | Ações humanas mistas |
| Diving-48 | 47 | Mergulhos olímpicos |
| AUTSL | 226 | Língua de sinais turca |
| DriveAct | 33 | Ações de condução |
| EPIC-Kitchens | 89 | Cozinha em egocentric view |
| FineGym | 97 | Ginástica atlética |

### 2.2 Manifests (vídeos de validação)

**Localização:** `evaluations/accv2026/manifests/{dataset}_val_20_per_class.csv`

Cada manifest tem no máximo 20 vídeos por classe do split de validação, com colunas: `video_path`, `label_id`, `split`, `exists`.

### 2.3 Checkpoints fine-tuned

**Localização:** `/scratch/wesleyferreiramaia/infoRates/fine_tuned_models/`

Padrão de nome: `accv2026_{model}_{dataset}_full_e10_{gpu}` onde `gpu` é `h200` ou `a100`.

Modelos disponíveis: timesformer, videomae, vivit, r3d_18, mc3_18, r2plus1d_18, slowfast_r50, videomamba.

**Nota FineGym:** FineGym tem taxonomy mas **não tem checkpoints fine-tuned**. Para rodar E3 no FineGym é preciso primeiro treinar os 8 modelos nele (ver Seção 7).

---

## 3. Experimentos Realizados

Foram testadas três abordagens para medir localização temporal da evidência:

| Opção | Nome | Descrição | Resultado |
|-------|------|-----------|-----------|
| A | Optical Flow Burstiness | CV da magnitude do fluxo óptico inter-frames | ❌ Não funciona |
| B | Attention Entropy (TimeSformer) | Entropia dos pesos de atenção temporal | Parcial (4/7) |
| C | Sliding Window Concentration | Concentração da confiança em janela deslizante | ✅ Resultado principal |
| — | Temporal GradCAM | Gradiente de entrada por frame | ❌ Não funciona |

---

## 4. Opção A — Optical Flow Burstiness

**Ideia:** Classes com alta variância do fluxo óptico inter-frames têm mais movimento localizado → mais aliasing.

**Script:** `scripts/accv2026/e3_nyquist_robust_all_datasets.py` (seção Option A)  
**Sbatch:** `scripts/accv2026/slurm_nyquist_robust.sbatch`  
**Saída:** `evaluations/accv2026/e3_spectral/{dataset}_flow_stats_v2.csv`

**Métrica:** `partial_r(flow_burstiness, aliasing | mean_flow)` — correlação parcial controlando velocidade média.

### Resultados Option A (partial r controlando mean_flow)

| Dataset | n | partial r | p |
|---------|---|-----------|---|
| UCF-101 | 101 | −0.048 | 0.633 |
| SSv2 | 174 | +0.084 | 0.272 |
| HMDB-51 | 51 | +0.285 | 0.043 |
| Diving-48 | 47 | +0.008 | 0.958 |
| AUTSL | 226 | −0.038 | 0.571 |
| DriveAct | 33 | −0.067 | 0.713 |
| EPIC-Kitchens | 89 | −0.094 | 0.382 |
| **Média pooled** | | **+0.019** | |

**Conclusão:** Fluxo óptico não prediz aliasing. Apenas HMDB-51 é marginalmente significativo. **Abandonado como contribuição principal.**

---

## 5. Opção B — Attention Entropy (TimeSformer only)

**Ideia:** Se o TimeSformer concentra sua atenção temporal em poucos frames, a evidência é localizada → mais aliasing.

**Script:** `scripts/accv2026/e3_nyquist_robust_all_datasets.py` (seção Option B)  
**Mecanismo:** Hook no `attn_drop` de todos os 12 blocos do TimeSformer, captura tensores `[patches × heads × T × T]`, calcula `1 - H(mean_attn_per_frame) / log(T)` por vídeo.

**Limitação:** Funciona apenas para TimeSformer (requer mecanismo de atenção). Não generalizável.

### Resultados Option B (r de Pearson, TimeSformer)

| Dataset | n | r | p |
|---------|---|---|---|
| UCF-101 | 101 | +0.192 | 0.055 |
| SSv2 | 174 | +0.273 | 0.0003 |
| HMDB-51 | 51 | +0.474 | 0.0004 |
| Diving-48 | 47 | −0.177 | 0.234 |
| AUTSL | 226 | +0.189 | 0.004 |
| DriveAct | 33 | +0.373 | 0.033 |
| EPIC-Kitchens | 89 | +0.070 | 0.512 |
| **Média pooled** | | **+0.199** | |

---

## 6. Temporal GradCAM (experimento negativo)

**Ideia:** Gradiente `d(logit_correto)/d(pixel[t])` mede quais frames mais impactam a predição.

**Script:** `scripts/accv2026/e3_temporal_gradcam_all_models.py`  
**Sbatch:** `scripts/accv2026/slurm_temporal_gradcam.sbatch`  
**Saída:** `evaluations/accv2026/e3_spectral/{model}_{dataset}_temporal_saliency.csv` e `nyquist_temporal_saliency_master.csv`

**Modelos testados:** timesformer, videomae, vivit, r3d_18, mc3_18, r2plus1d_18, slowfast_r50 (7 modelos).

### Resultados GradCAM (Spearman ρ ensemble de 7 modelos)

| Dataset | n | ρ | p |
|---------|---|---|---|
| UCF-101 | 101 | +0.034 | 0.739 |
| SSv2 | 174 | +0.050 | 0.513 |
| HMDB-51 | 51 | +0.452 | 0.0009 |
| Diving-48 | 47 | −0.187 | 0.207 |
| AUTSL | 226 | −0.092 | 0.170 |
| DriveAct | 33 | +0.252 | 0.157 |
| EPIC-Kitchens | 89 | −0.024 | 0.821 |
| **Média pooled** | | **+0.069** | |

**Conclusão:** GradCAM captura sensibilidade a perturbações nos pixels, mas não onde a evidência discriminativa está concentrada. **Não funciona para este propósito.**

---

## 7. Opção C — Sliding Window Concentration ← RESULTADO PRINCIPAL

**Ideia:** Decodificar 48 frames de cada vídeo, deslizar uma janela de T frames com stride adaptado para gerar ~10 janelas, registrar `P(classe correta)` em cada posição. A concentração mede "em quantos momentos o modelo fica confiante":

```
concentration = 1 - H(P_normalizado) / log(n_janelas)
```

Onde H é a entropia de Shannon. Concentração alta → o modelo só fica confiante em janelas específicas → evidência temporalmente localizada → mais aliasing esperado.

### 7.1 Parâmetros por modelo

| Modelo | Família | T (frames clip) | stride | n_janelas |
|--------|---------|-----------------|--------|-----------|
| TimeSformer | HF Transformer | 8 | 4 | 11 |
| VideoMAE | HF Transformer | 16 | 3 | 11 |
| ViViT | HF Transformer | 32 | 1 | 17 |
| R3D-18 | CNN TorchVision | 16 | 3 | 11 |
| MC3-18 | CNN TorchVision | 16 | 3 | 11 |
| R(2+1)D-18 | CNN TorchVision | 16 | 3 | 11 |
| SlowFast-R50 | SlowFast | 32 | 1 | 17 |
| VideoMamba | SSM (Mamba) | 8 | 4 | 11 |

### 7.2 Scripts e como rodar

#### Passo 1 — Rodar cada modelo em paralelo (7 modelos, sem VideoMamba)

```bash
cd /data/wesleyferreiramaia/infoRates

# Submeter 3 de uma vez (limite QOS = 3 jobs simultâneos)
for MODEL in r3d_18 mc3_18 r2plus1d_18; do
  MODEL_KEY=$MODEL sbatch --job-name=$MODEL scripts/accv2026/slurm_sw_model_worker.sbatch
done

# Aguardar e submeter próxima rodada
for MODEL in timesformer videomae vivit; do
  # aguarda slot livre manualmente ou com daemon abaixo
  MODEL_KEY=$MODEL sbatch --job-name=$MODEL scripts/accv2026/slurm_sw_model_worker.sbatch
done

# Última rodada
MODEL_KEY=slowfast_r50 sbatch --job-name=slowfast_r50 scripts/accv2026/slurm_sw_model_worker.sbatch
```

**Ou usar o daemon automático que submete assim que abrir slot (max 3 simultâneos):**

```bash
PENDING=(r3d_18 mc3_18 r2plus1d_18 timesformer videomae vivit slowfast_r50)
MAX=3
for MODEL in "${PENDING[@]}"; do
  while [ $(squeue -u $USER -h | wc -l) -ge $MAX ]; do sleep 30; done
  MODEL_KEY=$MODEL sbatch --job-name=$MODEL scripts/accv2026/slurm_sw_model_worker.sbatch
  echo "submetido $MODEL"
  sleep 5
done
```

**Script do worker:** `scripts/accv2026/e3_sw_model_worker.py`
- Aceita `MODEL_KEY` como argumento ou env var
- Pula CSVs que já existem (seguro para retomar)
- Processa os 7 datasets na ordem: ucf101, ssv2, hmdb51, diving48, autsl, driveact, epic_kitchens
- Usa `.venv` (não `.venv_mamba`)

**Saída por modelo:** `evaluations/accv2026/e3_spectral/sw_{model}_{dataset}.csv`  
Colunas: `label_id, n_videos, mean_concentration, std_concentration`

#### Passo 2 — Rodar VideoMamba separadamente (requer .venv_mamba)

```bash
sbatch scripts/accv2026/slurm_sw_videomamba.sbatch
```

**Script:** `scripts/accv2026/e3_sw_videomamba.py`  
**IMPORTANTE:** Usa `.venv_mamba` (não `.venv`) por causa do pacote `mamba_ssm`.  
**Bug conhecido e resolvido:** O modelo retorna logits em BFloat16. A linha crítica é `logits.float().softmax(-1)` — sem o `.float()` todos os vídeos falham silenciosamente com `TypeError: Got unsupported ScalarType BFloat16`.

**Saída:** `evaluations/accv2026/e3_spectral/sw_videomamba_{dataset}.csv`

#### Passo 3 — Calcular ensemble final

Após todos os modelos terminarem:

```bash
source .venv/bin/activate
python3 -u scripts/accv2026/e3_sw_recompute_ensemble.py
```

**Script:** `scripts/accv2026/e3_sw_recompute_ensemble.py`
- Lê todos os `sw_{model}_{dataset}.csv` disponíveis
- Calcula média de concentração por classe sobre os modelos disponíveis
- Salva `evaluations/accv2026/e3_spectral/sw_ensemble_{dataset}.csv`
- Correlaciona ensemble_concentration com `mean_abs_drop` da taxonomy
- Salva tabela master em `evaluations/accv2026/e3_spectral/sw_all_models_master.csv`
- Roda em `.venv` normal (sem mamba)

### 7.3 Tempo estimado por modelo (cluster A100/H200, 1 GPU)

| Modelo | Tempo estimado × 7 datasets |
|--------|---------------------------|
| r3d_18, mc3_18, r2plus1d_18 | ~20-25 min |
| timesformer | ~30-40 min |
| videomae | ~35-45 min |
| videomamba | ~25-35 min |
| slowfast_r50 | ~45-60 min |
| vivit | ~60-90 min (T=32, janela a janela) |

Com 3 jobs em paralelo: **~1.5-2 horas total** para todos os 8 modelos × 7 datasets.

---

## 8. Resultados Finais — Option C Ensemble 8 Modelos

### 8.1 TimeSformer isolado (partial r controlando mean_flow)

| Dataset | n | partial r | p | |
|---------|---|-----------|---|---|
| UCF-101 | 101 | +0.234 | 0.019 | ✅ |
| SSv2 | 174 | +0.281 | 0.0002 | ✅✅ |
| HMDB-51 | 51 | +0.491 | 0.0003 | ✅✅ |
| Diving-48 | 47 | +0.077 | 0.606 | ❌ |
| AUTSL | 226 | +0.142 | 0.033 | ✅ |
| DriveAct | 33 | +0.186 | 0.301 | ~ |
| EPIC-Kitchens | 89 | +0.181 | 0.091 | ~ |
| **Média pooled** | | **+0.227** | | positivo em 7/7 |

### 8.2 Ensemble 8 modelos (Spearman ρ)

Saída: `evaluations/accv2026/e3_spectral/sw_all_models_master.csv`

| Dataset | n classes | n modelos | ρ Spearman | p | Pearson r | p | |
|---------|-----------|-----------|------------|---|-----------|---|---|
| UCF-101 | 101 | 8 | **+0.456** | <0.001 | +0.250 | 0.012 | ✅✅✅ |
| SSv2 | 174 | 6 | **+0.304** | <0.001 | +0.323 | <0.001 | ✅✅✅ |
| HMDB-51 | 51 | 8 | **+0.605** | <0.001 | +0.534 | <0.001 | ✅✅✅ |
| DriveAct | 33 | 8 | **+0.422** | 0.014 | +0.410 | 0.018 | ✅ |
| EPIC-Kitchens | 89 | 8 | +0.143 | 0.182 | +0.071 | 0.507 | ~ |
| AUTSL | 226 | 8 | +0.074 | 0.265 | +0.104 | 0.120 | ❌ |
| Diving-48 | 47 | 8 | −0.062 | 0.679 | −0.101 | 0.501 | ❌ |
| **Média pooled** | | | **+0.277** | | **+0.227** | | positivo em 6/7 |

**SSv2 usa 6 modelos** (VideoMAE e R3D-18 não têm checkpoint SSv2 — sem `accv2026_videomae_ssv2_*` nem `accv2026_r3d_18_ssv2_*` no scratch).

### 8.3 Comparação entre abordagens

| Abordagem | Pooled ρ | Sig. datasets | |
|-----------|----------|--------------|---|
| Opt. A — Optical Flow | +0.019 | 1/7 | ❌ Descartada |
| GradCAM | +0.069 | 1/7 | ❌ Descartada |
| Opt. B — Attention (TimeSformer) | +0.199 | 4/7 | Parcial |
| Opt. C — SW TimeSformer | +0.227 | 4/7 | ✅ Positivo em 7/7 |
| **Opt. C — SW Ensemble 8 modelos** | **+0.277** | **4/7** | ✅✅ **Resultado final** |

---

## 9. Interpretação dos Resultados

### Datasets significativos (p < 0.05)

- **HMDB-51 (ρ=+0.605):** Dataset diverso com ações que vão de "escovar cabelo" a "saltar". Alta variância de localização temporal entre classes → correlação mais forte.
- **UCF-101 (ρ=+0.456):** Dataset clássico com muitas ações de atividade física. Ações como "punch" e "diving" têm evidência em momentos específicos.
- **DriveAct (ρ=+0.422):** Ações de condução têm diferenças claras de localização (ex: "acionar pisca" vs "conduzir reto").
- **SSv2 (ρ=+0.304):** Ações mão-objeto — "jogando X para longe" vs "passando X de mão em mão" têm localização temporal bem diferente.

### Datasets não-significativos

- **AUTSL (ρ=+0.074):** Língua de sinais — o aliasing pode ser regido por frequência espacial dos gestos, não por localização temporal.
- **Diving-48 (ρ=−0.062):** Classes muito similares temporalmente (todas são mergulhos) → variância de `mean_abs_drop` é pequena, pouca variância para predizer.
- **EPIC-Kitchens (ρ=+0.143):** Câmera egocentric com muito movimento — o fluxo background contamina a análise de localização temporal.

### Sinal consistente

O fato mais importante: **Option C é positiva em 6/7 datasets** (o único negativo é Diving-48 com ρ=−0.062, praticamente zero). Isso demonstra que o preditor é consistente entre domínios, não um artefato de um dataset específico.

---

## 10. Arquivos de Saída Gerados

```
evaluations/accv2026/e3_spectral/
├── sw_{model}_{dataset}.csv          # por modelo × dataset (55 CSVs = 8 modelos × 7 datasets, menos 2 missing)
├── sw_ensemble_{dataset}.csv         # ensemble médio por dataset (7 CSVs)
├── sw_all_models_master.csv          # tabela master de correlações (RESULTADO FINAL)
├── nyquist_temporal_saliency_master.csv  # GradCAM ensemble (experimento negativo)
├── nyquist_robust_all_datasets.csv   # Options A+B+C com TimeSformer
└── {dataset}_flow_stats_v2.csv       # fluxo óptico por dataset
```

---

## 11. O que Falta — FineGym

FineGym tem:
- ✅ Taxonomy: `evaluations/accv2026/e5_taxonomy/finegym_class_taxonomy.csv` (97 classes, com `mean_abs_drop`)
- ✅ Manifest: `evaluations/accv2026/manifests/finegym_val_20_per_class.csv`
- ❌ **Checkpoints fine-tuned: nenhum**

### Para adicionar FineGym:

**Passo 1 — Treinar os 8 modelos no FineGym**

Usar os scripts de treino existentes com dataset=finegym. Exemplo:
```bash
sbatch scripts/accv2026/slurm_train.sbatch timesformer finegym
# ... repetir para todos os 8 modelos
```

Tempo estimado: ~4-8h por modelo × 8 modelos ÷ 3 paralelo = **~1.5-2 dias de cluster**.

**Passo 2 — Rodar sliding window no FineGym**

Após treino, adicionar `"finegym": "finegym_val_20_per_class.csv"` no dict `DATASETS` de `e3_sw_model_worker.py` e `e3_sw_videomamba.py`, e rodar novamente:

```bash
# Adicionar finegym ao dict DATASETS nos scripts, então:
for MODEL in r3d_18 mc3_18 r2plus1d_18 timesformer videomae vivit slowfast_r50; do
  MODEL_KEY=$MODEL sbatch --job-name=${MODEL}_fg scripts/accv2026/slurm_sw_model_worker.sbatch
done
sbatch scripts/accv2026/slurm_sw_videomamba.sbatch
```

**Passo 3 — Recomputar ensemble**

```bash
python3 -u scripts/accv2026/e3_sw_recompute_ensemble.py
```

Isso vai incluir FineGym na tabela master automaticamente.

---

## 12. Para o Paper

### Claim principal (Seção de Análise / Supplementary)

> We validate our temporal aliasing hypothesis by measuring the correlation between per-class sliding-window confidence concentration and per-class accuracy drop under temporal subsampling. Across 7 diverse action recognition benchmarks and an ensemble of 8 video understanding models spanning CNNs, Transformers, and State-Space Models, we find consistent positive correlation (Spearman ρ pooled = +0.277), with 4 out of 7 datasets reaching statistical significance (p < 0.05). Notably, the correlation is positive in 6 out of 7 datasets, demonstrating that classes whose discriminative evidence is temporally concentrated suffer disproportionately from sparse frame sampling — directly supporting the Nyquist-Shannon framing of adaptive temporal allocation.

### Tabela para o paper (LaTeX)

```latex
\begin{table}[h]
\centering
\caption{Correlation between sliding-window confidence concentration and per-class aliasing drop
         ($\uparrow$ means concentrated evidence predicts higher aliasing, as hypothesized).
         Ensemble of 8 models; $n_\text{models}$ indicates models with available checkpoints.}
\begin{tabular}{lrrccc}
\toprule
Dataset & Classes & Models & $\rho$ & $r$ & $p$ \\
\midrule
HMDB-51       &  51 & 8 & \textbf{+0.605} & +0.534 & $<$0.001 \\
UCF-101       & 101 & 8 & \textbf{+0.456} & +0.250 & $<$0.001 \\
DriveAct      &  33 & 8 & \textbf{+0.422} & +0.410 &  0.014   \\
SSv2          & 174 & 6 & \textbf{+0.304} & +0.323 & $<$0.001 \\
EPIC-Kitchens &  89 & 8 & +0.143          & +0.071 &  0.182   \\
AUTSL         & 226 & 8 & +0.074          & +0.104 &  0.265   \\
Diving-48     &  47 & 8 & $-$0.062        & $-$0.101 & 0.679  \\
\midrule
Pooled mean   &     &   & \textbf{+0.277} & +0.227 &          \\
\bottomrule
\end{tabular}
\label{tab:aliasing-correlation}
\end{table}
```
