# EVA-Mamba3 — Implementation Plan

**Goal:** CVPR submission with full experimental results on FineGym, AUTSL, SSv2,
Diving-48, HMDB-51, UCF-101.

**Code lives in:** `src/eva_mamba3/` (new package inside the InfoRates repo)

---

## Phase 0 — Environment & Dependencies

### 0.1 Install Mamba-3

Mamba-3 landed in the official `state-spaces/mamba` repository (`modules/mamba3.py`).
Install from source to get the latest:

```bash
# Force rebuild so CUDA kernels match the system (CUDA 13.2 / PyTorch 2.x)
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
    git+https://github.com/state-spaces/mamba.git \
    --no-build-isolation
```

Verify:

```python
from mamba_ssm.modules.mamba3 import Mamba3   # should import without error
```

If `mamba3.py` is not yet in the pip release, install `mamba3-ssm` as fallback:

```bash
pip install mamba3-ssm
```

### 0.2 Additional dependencies

```bash
pip install einops rotary-embedding-torch timm
```

### 0.3 Package skeleton

```
src/eva_mamba3/
├── __init__.py
├── encoder.py          # spatial encoder (ResNet-50 → per-frame features)
├── eva.py              # EVA dual-band tokenizer
├── backbone.py         # bidirectional Mamba-3 + classification head
├── model.py            # full EVA-Mamba3 assembled
├── losses.py           # composite training objective
├── datasets.py         # dense clip loaders for all 6 datasets
├── train.py            # training loop
└── evaluate.py         # TDS, AAG, phase robustness, Pareto
```

---

## Phase 1 — Spatial Encoder

**File:** `src/eva_mamba3/encoder.py`

**Input:** raw frames `(B, T, H, W, 3)` uint8
**Output:** per-frame features `(B, T, C)` with `C = 512`

### Design

Use ResNet-50 from torchvision, remove the final FC layer, and apply global
average pooling after layer4:

```python
class SpatialEncoder(nn.Module):
    def __init__(self, C_out=512, freeze=False):
        # torchvision.models.resnet50(pretrained=True)
        # truncate at avgpool output → (B*T, 2048, 1, 1)
        # project 2048 → C_out with a linear + LayerNorm
        ...

    def forward(self, frames: Tensor) -> Tensor:
        # frames: (B, T, H, W, 3) uint8
        # return: (B, T, C_out)
        B, T = frames.shape[:2]
        x = frames.view(B*T, H, W, 3).permute(0,3,1,2).float() / 255.
        x = normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        feat = self.backbone(x)          # (B*T, 2048)
        feat = self.proj(feat)           # (B*T, C_out)
        return feat.view(B, T, -1)
```

**Notes:**
- Process frames in chunks of 16 to avoid OOM at T=64
- `freeze=True` in early training, unfreeze after 10 epochs
- Native resolution: 224×224 (resize in dataloader)

---

## Phase 2 — EVA Dual-Band Tokenizer

**File:** `src/eva_mamba3/eva.py`

**Input:** `f` of shape `(B, T, C)`, stride `s ∈ {1,2,4,8,16}`
**Output:** `Z` of shape `(B, T//s, D)` (or `(B, T//s * M, D)` for M=2 variant)

### 2.1 Gaussian low-pass filter

```python
class LearnableGaussian(nn.Module):
    def __init__(self, K=3, sigma_init=1.5):
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma_init)))
        self.K = K

    def forward(self, f: Tensor) -> Tensor:
        # f: (B, T, C)
        sigma = self.log_sigma.exp().clamp(0.3, 5.0)
        k = torch.arange(-self.K, self.K+1, device=f.device).float()
        h = torch.exp(-k**2 / (2 * sigma**2))
        h = h / h.sum()                        # normalise
        # Apply as 1D depthwise conv over time dim
        # pad edges with replicate
        f_pad = F.pad(f.permute(0,2,1), (self.K, self.K), mode='replicate')
        ell = F.conv1d(f_pad, h.view(1,1,-1).expand(C,1,-1),
                       groups=C).permute(0,2,1)  # (B, T, C)
        return ell
```

### 2.2 Event score MLP

```python
class EventScorer(nn.Module):
    def __init__(self, C):
        self.mlp = nn.Sequential(
            nn.Linear(4*C, C//4),
            nn.GELU(),
            nn.Linear(C//4, 1),
        )

    def forward(self, f, r) -> Tensor:
        # f: (B, T, C), r: (B, T, C)
        df_fwd = (f[:, 1:] - f[:, :-1]).abs()   # (B, T-1, C)
        df_bwd = (f[:, :-1] - f[:, 1:]).abs()
        # pad to T
        df_fwd = F.pad(df_fwd, (0,0,0,1))
        df_bwd = F.pad(df_bwd, (0,0,1,0))
        inp = torch.cat([f, df_fwd, df_bwd, r.abs()], dim=-1)  # (B, T, 4C)
        return torch.sigmoid(self.mlp(inp)).squeeze(-1)          # (B, T)
```

### 2.3 Window compression

```python
class EVATokenizer(nn.Module):
    def __init__(self, C, D, K=3, sigma_init=1.5, tau=0.1, M=1):
        self.gaussian = LearnableGaussian(K, sigma_init)
        self.scorer   = EventScorer(C)
        self.q_ctx    = nn.Parameter(torch.randn(C) / C**0.5)
        self.W_ctx    = nn.Linear(C, D, bias=False)
        self.W_evt    = nn.Linear(C, D, bias=False)
        # 5 stride values: {1,2,4,8,16} → indices {0,1,2,3,4}
        self.pos_embed = nn.Embedding(5, D)
        self.STRIDE_IDX = {1:0, 2:1, 4:2, 8:3, 16:4}
        self.ln  = nn.LayerNorm(D)
        self.tau = tau
        self.M   = M   # 1 = single fused token; 2 = ctx and evt as separate tokens

    def forward(self, f: Tensor, s: int) -> Tensor:
        # f: (B, T, C)
        B, T, C = f.shape
        assert T % s == 0, f"T={T} must be divisible by stride s={s}"
        n_win = T // s

        ell = self.gaussian(f)          # (B, T, C)  low-frequency context
        r   = f - ell                   # (B, T, C)  high-frequency residual
        e   = self.scorer(f, r)         # (B, T)     event scores ∈ [0,1]

        # Reshape into windows
        ell_w = ell.view(B, n_win, s, C)   # (B, n_win, s, C)
        r_w   = r.view(B, n_win, s, C)
        e_w   = e.view(B, n_win, s)

        # Context attention: softmax(q^T ell_t / sqrt(C))
        a_ctx = torch.einsum('c,bwsc->bws', self.q_ctx, ell_w) / C**0.5
        a_ctx = a_ctx.softmax(dim=-1)                           # (B, n_win, s)

        # Event attention: softmax(e_t / tau)
        a_evt = (e_w / self.tau).softmax(dim=-1)                # (B, n_win, s)

        # Weighted aggregation
        z_ctx = torch.einsum('bws,bwsc->bwc', a_ctx, ell_w)    # (B, n_win, C)
        z_evt = torch.einsum('bws,bwsc->bwc', a_evt, r_w)      # (B, n_win, C)

        # Positional embedding
        pos_idx = self.STRIDE_IDX[s]
        p = self.pos_embed(torch.tensor(pos_idx, device=f.device))  # (D,)

        if self.M == 1:
            z = self.ln(self.W_ctx(z_ctx) + self.W_evt(z_evt) + p)  # (B, n_win, D)
            return z
        else:
            # M=2: return ctx and evt as separate tokens → (B, 2*n_win, D)
            z_c = self.W_ctx(z_ctx) + p
            z_e = self.W_evt(z_evt) + p
            # Interleave: [ctx_0, evt_0, ctx_1, evt_1, ...]
            z = torch.stack([z_c, z_e], dim=2).view(B, 2*n_win, -1)
            return self.ln(z)
```

**Key design decisions:**
- `M=1` for strides 1–4; `M=2` for strides 8–16 (controlled in model.forward)
- `tau=0.1` is fixed (not learned) — sharpens event attention to near-argmax

---

## Phase 3 — Mamba-3 Backbone

**File:** `src/eva_mamba3/backbone.py`

**Input:** `Z` of shape `(B, L, D)` where `L = T//s` (or `2*T//s` for M=2)
**Output:** class logits `(B, n_classes)`

### 3.1 Bidirectional Mamba-3

```python
from mamba_ssm.modules.mamba3 import Mamba3

class BidirectionalMamba3Block(nn.Module):
    def __init__(self, D, d_state=64, d_conv=4, expand=2):
        # d_state=64 complex states = 128 real-equivalent (Mamba-3 halves state dim)
        self.fwd = Mamba3(d_model=D, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba3(d_model=D, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm_fwd = nn.LayerNorm(D)
        self.norm_bwd = nn.LayerNorm(D)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, D)
        h_fwd = self.fwd(self.norm_fwd(x)) + x
        h_bwd = self.bwd(self.norm_bwd(x.flip(1))).flip(1) + x
        return torch.cat([h_fwd, h_bwd], dim=-1)  # (B, L, 2D)
```

### 3.2 Full backbone

```python
class Mamba3Backbone(nn.Module):
    def __init__(self, D_in, D, n_layers=12, n_classes=97):
        self.input_proj = nn.Linear(D_in, D)
        self.blocks = nn.ModuleList([
            BidirectionalMamba3Block(D) for _ in range(n_layers)
        ])
        # Attention pooling: learns to weight tokens for clip-level repr
        self.attn_pool = nn.Linear(2*D, 1)
        self.head = nn.Linear(2*D, n_classes)

    def forward(self, Z: Tensor) -> Tensor:
        # Z: (B, L, D_in)
        x = self.input_proj(Z)        # (B, L, D)
        for blk in self.blocks:
            x = blk(x)                # (B, L, 2D)
        # Attention pooling over sequence
        w = self.attn_pool(x).softmax(dim=1)   # (B, L, 1)
        v = (w * x).sum(dim=1)                  # (B, 2D)
        return self.head(v)                      # (B, n_classes)
```

**Config (base model):**
- `D = 384`, `n_layers = 12`, `d_state = 64` (complex, ≡ 128 real), `expand = 2`
- Total params (backbone only): ~40M

---

## Phase 4 — Full EVA-Mamba3 Model

**File:** `src/eva_mamba3/model.py`

```python
class EVAMamba3(nn.Module):
    def __init__(self, C=512, D=384, n_classes=97, n_layers=12):
        self.encoder  = SpatialEncoder(C_out=C)
        self.eva      = EVATokenizer(C=C, D=D, M=1)   # M toggled at forward
        self.backbone = Mamba3Backbone(D_in=D, D=D, n_layers=n_layers,
                                       n_classes=n_classes)

    def forward(self, frames: Tensor, s: int) -> dict:
        # frames: (B, T, H, W, 3)
        f = self.encoder(frames)       # (B, T, C)

        # Dense forward (always needed for distillation loss at s>1)
        logits_dense = None
        if self.training and s > 1:
            z_dense = self.eva(f, s=1)
            logits_dense = self.backbone(z_dense)

        # Sparse forward
        M = 2 if s >= 8 else 1
        self.eva.M = M
        z = self.eva(f, s=s)           # (B, T//s * M, D)
        logits = self.backbone(z)      # (B, n_classes)

        return {
            'logits': logits,
            'logits_dense': logits_dense,
            'event_weights': self.eva._last_a_evt,   # cached in eva.forward
            'dense_event_weights': self.eva._last_a_evt_dense,
        }
```

**Note:** `_last_a_evt` is a cached tensor set inside `EVATokenizer.forward` so
that the evidence-alignment loss can access it without a second forward pass.

---

## Phase 5 — Composite Training Loss

**File:** `src/eva_mamba3/losses.py`

```python
class EVALoss(nn.Module):
    def __init__(self, lambda_kl=1.0, lambda_evid=0.5,
                 lambda_phase=0.5, lambda_sp=0.1, T_distill=3.0):
        ...

    def forward(self, out: dict, labels: Tensor, s: int,
                out_phase2: dict = None) -> dict:
        losses = {}

        # L_cls: cross-entropy at stride s
        losses['cls'] = F.cross_entropy(out['logits'], labels)

        # L_KL: dense-to-sparse distillation (only when s > 1)
        if s > 1 and out['logits_dense'] is not None:
            T = self.T_distill
            p_dense  = F.softmax(out['logits_dense'].detach() / T, dim=-1)
            log_p_sp = F.log_softmax(out['logits'] / T, dim=-1)
            losses['kl'] = self.lambda_kl * T**2 * \
                           F.kl_div(log_p_sp, p_dense, reduction='batchmean')

        # L_evid: align sparse event weights to dense evidence map
        if s > 1 and out['event_weights'] is not None:
            a_dense  = out['dense_event_weights'].detach()  # (B, T, s) → rescaled
            a_sparse = out['event_weights']                 # (B, T//s, s)
            # Flatten and normalise to same grid
            a_dense_flat  = a_dense.view(a_dense.shape[0], -1)
            a_dense_flat  = a_dense_flat / a_dense_flat.sum(-1, keepdim=True)
            a_sparse_flat = a_sparse.view(a_sparse.shape[0], -1)
            a_sparse_flat = a_sparse_flat / a_sparse_flat.sum(-1, keepdim=True)
            losses['evid'] = self.lambda_evid * \
                             (a_dense_flat - a_sparse_flat).abs().mean()

        # L_phase: sampling-phase consistency
        if out_phase2 is not None:
            p1 = F.softmax(out['logits'], dim=-1)
            p2 = F.softmax(out_phase2['logits'], dim=-1)
            m  = 0.5 * (p1 + p2)
            js = 0.5 * (F.kl_div(p1.log(), m, reduction='batchmean') +
                         F.kl_div(p2.log(), m, reduction='batchmean'))
            losses['phase'] = self.lambda_phase * js

        # L_sp: entropy penalty on event attention (encourage sparsity)
        if out['event_weights'] is not None:
            a = out['event_weights']   # (B, n_win, s)
            H = -(a * (a + 1e-8).log()).sum(-1).mean()
            losses['sp'] = self.lambda_sp * H

        losses['total'] = sum(losses.values())
        return losses
```

---

## Phase 6 — Dataset Loaders

**File:** `src/eva_mamba3/datasets.py`

**Key requirement:** return **dense** clips of exactly `T=64` frames in native
order (no subsampling). EVA handles all subsampling internally.

```python
class DenseVideoDataset(Dataset):
    """Returns T consecutive frames from a random window of the clip."""
    T = 64          # number of dense frames to decode
    SIZE = 224      # spatial resolution

    def __init__(self, dataset_name, split='val', root=None):
        # dataset_name ∈ {finegym, autsl, ssv2, diving48, hmdb51, ucf101}
        # Reuse existing clip lists from evaluations/accv2026/e2_variance/
        # or build from annotation files
        ...

    def __getitem__(self, idx):
        # Decode T consecutive frames starting from a random offset
        # within the annotated clip boundary
        # Return: frames (T, H, W, 3) uint8, label int
        ...
```

**Reuse from InfoRates:** the clip lists and annotation parsing in
`src/info_rates/datasets/` can be adapted. The difference is that here we need
dense T=64 frames, not the already-subsampled tensors.

**Padding:** if the clip has fewer than T frames, repeat the last frame.

---

## Phase 7 — Training Loop

**File:** `src/eva_mamba3/train.py`

```python
# Pseudo-code for the training loop

STRIDES = [1, 2, 4, 8, 16]

for epoch in range(N_EPOCHS):
    for frames, labels in dataloader:
        s = random.choice(STRIDES)     # sample stride uniformly per step
        phi = random.randint(0, s-1)   # random temporal offset for phase loss

        out = model(frames, s=s)

        # Phase-consistency: second forward with offset phi
        frames_shifted = frames[:, phi::1][:, :T]   # shift by phi
        # ... pad to T if needed
        out_phase2 = model(frames_shifted, s=s) if s > 1 else None

        loss_dict = criterion(out, labels, s, out_phase2)
        loss_dict['total'].backward()
        optimizer.step()
        ...
```

**Hyperparameters:**
- Optimizer: AdamW, lr=2e-4, weight decay=0.05
- Schedule: cosine with 5-epoch linear warmup
- Epochs: 50
- Batch size: 64 (4 × A100, 16 per GPU)
- Unfreeze spatial encoder at epoch 10

---

## Phase 8 — Evaluation Suite

**File:** `src/eva_mamba3/evaluate.py`

### 8.1 TDS and per-stride accuracy

```python
def eval_tds(model, dataloader, strides=[1,2,4,8,16]):
    acc = {}
    for s in strides:
        correct, total = 0, 0
        for frames, labels in dataloader:
            out = model(frames, s=s)
            pred = out['logits'].argmax(-1)
            correct += (pred == labels).sum().item()
            total += len(labels)
        acc[s] = correct / total
    tds = max(0, acc[1] - acc[16])
    return acc, tds
```

### 8.2 Phase robustness

```python
def eval_phase_robustness(model, dataloader, s, n_phases=8):
    """Variance of predictions across n_phases random temporal offsets."""
    var_per_clip = []
    for frames, _ in dataloader:
        probs = []
        for phi in torch.linspace(0, s-1, n_phases).long():
            f_shifted = frames[:, phi:phi+T_DENSE]   # T_DENSE = 64
            # pad if needed
            out = model(f_shifted, s=s)
            probs.append(out['logits'].softmax(-1))
        probs = torch.stack(probs, dim=0)            # (n_phases, B, C)
        var_per_clip.append(probs.var(dim=0).mean().item())
    return sum(var_per_clip) / len(var_per_clip)
```

### 8.3 Anti-Aliasing Gain (AAG)

```python
def aag(acc_ours: dict, acc_baseline: dict,
        strides=[2,4,8,16]) -> float:
    return sum(acc_ours[s] - acc_baseline[s] for s in strides) / len(strides)
```

### 8.4 Pareto curve

Save `(latency_ms, accuracy)` pairs for each `(method, stride)` combination.
Latency measured with CUDA events, batch size 1, bfloat16, on A100.

---

## Phase 9 — Experiment Schedule

### Step 1: Verify EVA module standalone (no training)

```python
# Smoke test
f = torch.randn(2, 64, 512)
eva = EVATokenizer(C=512, D=384, M=1)
for s in [1, 2, 4, 8, 16]:
    z = eva(f, s=s)
    assert z.shape == (2, 64//s, 384), f"wrong shape at s={s}: {z.shape}"
```

### Step 2: Verify full model forward pass + loss

```python
model = EVAMamba3(n_classes=97)
frames = torch.randint(0, 255, (2, 64, 224, 224, 3), dtype=torch.uint8)
out = model(frames, s=4)
loss = criterion(out, torch.zeros(2, dtype=torch.long), s=4)
loss['total'].backward()   # must not error
```

### Step 3: Establish baseline numbers (VideoMamba, uniform stride)

Before training EVA-Mamba3, run VideoMamba on all 6 datasets at all strides
to get the TDS baseline numbers that will fill Table 1.
Use the existing InfoRates checkpoints from `fine_tuned_models/`.

### Step 4: Train EVA-Mamba3 on FineGym (primary benchmark)

FineGym first because it has the highest TDS (55.9 pp) and clearest signal.
Expected training time: ~12h on 4 × A100.

```bash
python src/eva_mamba3/train.py \
    --dataset finegym \
    --n_classes 97 \
    --epochs 50 \
    --batch_size 64 \
    --lr 2e-4 \
    --output checkpoints/eva_mamba3_finegym/
```

### Step 5: Train on AUTSL, SSv2

After FineGym converges (validate at s=16 every 5 epochs), train on AUTSL and
SSv2. These three are the high-TDS datasets central to the CVPR contribution.

### Step 6: Train on control datasets (Diving-48, HMDB-51, UCF-101)

Critical for showing EVA does not degrade low-TDS performance.

### Step 7: Ablation runs on FineGym

One checkpoint per ablation row in Table 2 (ablation table):

| Variant | Change from full model |
|---------|----------------------|
| Mamba-3 only | Remove EVA; use naive stride |
| EVA ctx only | Set `W_evt = 0`, `lambda_evid = 0` |
| No L_KL | `lambda_kl = 0` |
| No L_phase | `lambda_phase = 0` |
| No L_evid | `lambda_evid = 0` |
| Mamba-2 backbone | Replace Mamba3 blocks with Mamba2 |
| EVA-Mamba3 full | — |

### Step 8: Backbone generality (EVA + VideoMamba, EVA + TimeSformer)

Plug EVA tokenizer in front of existing VideoMamba and TimeSformer checkpoints
from InfoRates. Fine-tune EVA tokenizer only (freeze backbone) to show
architecture-agnostic gains.

### Step 9: Efficiency measurements

Measure latency on A100 (bfloat16, batch=1, 20 warmup, 100 iters) for all
methods at all strides. Build Pareto curves.

---

## Phase 10 — Paper Result Tables

Once all experiments are complete, fill in the placeholder tables in `main.tex`:

- **Table 1 (main results):** acc at s∈{1,2,4,8,16} + TDS for all 6 datasets × 7 methods
- **Table 2 (ablation):** TDS + a@s=1 + a@s=16 for 7 ablation variants on FineGym
- **Table 3 (efficiency):** GFLOPs, params, latency@s=4, acc@s=4
- **Figure 1:** architecture diagram (EVA tokenizer + Mamba-3 backbone)
- **Figure 2:** Pareto curves on FineGym and UCF-101
- **Figure 3:** TDS bar chart (motivation; reuse from InfoRates analysis)

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Mamba-3 CUDA kernel incompatibility with CUDA 13.2 | Build from source; fallback to pure-Python Mamba-3 for debugging |
| EVA adds no benefit on FineGym (TDS unchanged) | Check event weights: if `a_evt` is uniform, the event branch is not learning — increase `lambda_sp` |
| Phase loss destabilises training | Start with `lambda_phase=0`, add at epoch 10 |
| OOM at T=64 with ResNet-50 on A100 | Process in chunks of 16 frames in the spatial encoder |
| No improvement on UCF-101 (expected, not a failure) | Explicitly flag this as validation in the paper |
| Mamba-3 complex states not yet in `mamba_ssm` pip | Use `mamba3-ssm` package or stub with Mamba-2 for ablation comparison |

---

## File Checklist

- [ ] `src/eva_mamba3/__init__.py`
- [ ] `src/eva_mamba3/encoder.py` — SpatialEncoder
- [ ] `src/eva_mamba3/eva.py` — LearnableGaussian, EventScorer, EVATokenizer
- [ ] `src/eva_mamba3/backbone.py` — BidirectionalMamba3Block, Mamba3Backbone
- [ ] `src/eva_mamba3/model.py` — EVAMamba3
- [ ] `src/eva_mamba3/losses.py` — EVALoss
- [ ] `src/eva_mamba3/datasets.py` — DenseVideoDataset × 6 datasets
- [ ] `src/eva_mamba3/train.py` — training loop
- [ ] `src/eva_mamba3/evaluate.py` — TDS, AAG, phase robustness, Pareto
- [ ] `scripts/eva_mamba3/smoke_test.py` — shape and loss sanity checks
- [ ] `scripts/eva_mamba3/train_finegym.sh`
- [ ] `scripts/eva_mamba3/train_all.sh`
- [ ] `scripts/eva_mamba3/eval_baselines.sh`
