# VideoMamba3 — Experimento

Baseado em: **VideoMamba** (OpenGVLab, ECCV 2024) + inovações do **Mamba-3** (ICLR 2026, arXiv 2603.15569).

## Motivação

VideoMamba usa **Mamba-1** como SSM base (mamba_ssm ≤2.x). O Mamba-3 introduz três melhorias ortogonais que são particularmente relevantes para compreensão de vídeo:

| Inovação | Por que importa para vídeo |
|----------|--------------------------|
| **Discretização Trapezoidal** | Acurácia de 2ª ordem na integração do estado → melhor tracking de objetos ao longo do tempo |
| **Estado Complexo via RoPE** | SSMs reais não conseguem resolver tarefas de paridade/posição; estados complexos recuperam expressividade sem overhead 4x — útil para rastreamento temporal |
| **MIMO (rank-R)** | Atualização de estado via produto matricial ao invés de produto externo escalar → mais capacidade expressiva por timestep |

Adicionalmente: **Hydra** (arXiv 2407.09941) propõe bidirecionalidade com projeção compartilhada, economizando ~50% de parâmetros vs nosso BiMamba atual (dois módulos independentes).

## Arquitetura Atual (VideoMamba + Mamba-1)

```
Video (B, T, H, W, C)
    → Patch Embed → (B, N, D)      # N = T × (H/patch) × (W/patch)
    → [Block × L]:
         BiMamba-1: forward(x) + backward(flip(x))   # dois Mamba-1 separados
    → Temporal pool → Head → logits
```

**Atualização de estado em Mamba-1 (Euler):**
```
h_t = exp(Δ·A) · h_{t-1}  +  Δ·B · x_t          # Euler 1ª ordem
y_t = C · h_t
```

## Arquitetura Proposta (VideoMamba3)

```
Video (B, T, H, W, C)
    → Patch Embed → (B, N, D)
    → [Block × L]:
         BiMamba3: forward(x) + backward(flip(x))   # dois Mamba-3 separados
    → Temporal pool → Head → logits
```

**Atualização de estado em Mamba-3 (Trapezoidal + Complexo + MIMO):**
```
# 1. Trapezoidal (2ª ordem):
h_t = exp(Δ·A) · h_{t-1}  +  (Δ/2) · B · (x_t + x_{t-1})

# 2. Estados complexos via RoPE data-dependent:
θ_t = f(Δ_t, A_t)                                  # ângulo de rotação
h_t = R(θ_t) · (exp(|Δ·A|) · h_{t-1})  +  (Δ/2) · B · (x_t + x_{t-1})

# 3. MIMO (rank-R, substitui produto externo):
X_t ∈ ℝ^{D×R}  (low-rank input projection)
h_t = A_disc · h_{t-1}  +  X_t @ W_O^T             # matrix × matrix
y_t = (h_t @ C_out^T).mean(dim=-1)
```

## Estado Atual

### Fase 1 — Núcleo SSM Mamba-3 em PyTorch puro ✅
- [x] `mamba3_core.py`: `TrapezoidalSSM` + `ComplexRoPESSM` + `MIMOSSMBlock` + `BiMamba3`
  - Implementado e testado: todas as 3 variantes passam forward pass
  - MIMO: `mimo_in: d_inner → d_inner×R`, `mimo_out ∈ R^{rank×d_state}`, `X @ W_O → (d_inner, d_state)`
- [x] Implementação legível, sem dependência de kernels CUDA novos
- [x] Observação importante: o scan atual é explícito em PyTorch e fica lento em sequências longas

### Fase 2 — BiMamba3 + Block ✅
- [x] `videomamba3.py`: `create_block()` aceita `mamba3_variant` → usa `BiMamba3` ou `BiMamba`
- [x] `VisionMamba.__init__()` aceita `mamba3_variant` e propaga para todos os blocos
- [x] `mamba_ssm` import é opcional (try/except) — VideoMamba3 roda sem CUDA 13
- [x] `videomamba3_tiny/small/middle()` registrados como `@register_model`
- [x] Forward pass end-to-end verificado com vídeo sintético

### Fase 3 — Fine-tuning e ablação ✅
- [x] `scripts/accv2026/train_videomamba3.py`
- [x] Treino em UCF-101 com `trapezoidal`, `complex` e `mimo`
- [x] Suporte a `--depth` para checkpoints pequenos e rápidos de comunidade
- [x] Histórico CSV por run e metadados completos no checkpoint
- [x] Benchmark sintético de latência/VRAM
- [x] Compilador de tabelas para o draft do paper

### Fase 4 — Validação de sistema
- [x] Benchmark sintético simples: `scripts/accv2026/benchmark_videomamba3.py`
- [x] Validador completo de sistema: `scripts/accv2026/validate_videomamba3_system.py`
- [x] Mede inferência, train-step, VRAM alocada/reservada, videos/s e tokens/s
- [ ] Rodar validação em H200 para `complex` vs `trapezoidal`, `depth=4/8/12/24`
- [ ] Rodar treino rápido `complex`, `depth=8`, `112x112`, `4` frames
- [ ] Fixed-budget eval do checkpoint VideoMamba3 contra VideoMamba normal

### Fase 5 — Release público
- [x] Infra de export existe, mas fica congelada até validação real
- [ ] Publicar pesos somente depois de latência/VRAM/eval real

### Fase 6 — Avaliação e Paper
- [ ] Eval fixed-budget (4/8/16/32f) em todos os 7 datasets
- [ ] Comparar curvas temporal accuracy vs budget com VideoMamba baseline
- [ ] Análise de eficiência (FLOPs, latência, VRAM)
- [ ] Kernel fused/chunked para substituir o scan explícito antes de treinos grandes

## Instalação

A versão atual `mamba_ssm 2.3.1` tem Mamba-1 e Mamba-2 mas NÃO tem Mamba-3.
Para usar a implementação oficial (quando disponível):
```bash
# Atualizar mamba_ssm para versão com Mamba3
source .venv_mamba/bin/activate
pip install --upgrade mamba-ssm
# ou instalar da source:
pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation
```

Nossa implementação PyTorch puro (Fase 1) não precisa de versão mais nova.

## Estrutura de Arquivos

```
experiments/videomamba3/
├── README.md              ← este arquivo
├── mamba3_core.py         ← ✅ BiMamba3 com 3 variantes (trapezoidal/complex/mimo)
└── videomamba3.py         ← ✅ VisionMamba + BiMamba3 (videomamba3_tiny/small/middle)
```

Scripts principais:

```bash
# Treino/ablação em H200. O default atual usa DEPTH=8 para ser publicável e rápido.
bash scripts/accv2026/submit_videomamba3_ablation_h200.sh

# Benchmark simples de latência/VRAM.
sbatch scripts/accv2026/slurm_h200_videomamba3_benchmark.sbatch

# Validação de sistema: inferência, train-step, VRAM e throughput.
sbatch scripts/accv2026/slurm_h200_videomamba3_system_validation.sbatch

# Treino rápido do candidato principal.
sbatch scripts/accv2026/slurm_h200_videomamba3_fast_ucf101.sbatch
```
