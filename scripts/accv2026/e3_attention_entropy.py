"""Option B — Temporal attention entropy as Nyquist test.

For each FineGym class, extracts temporal attention weights from all 12
TimeSformer blocks via attn_drop hooks, computes the entropy of the
attention distribution over frames, and correlates with per-class aliasing.

Hypothesis:
  Low temporal attention entropy = attention concentrated on few frames
  = discriminative evidence is temporally localised
  = class should alias MORE under sparse sampling  →  negative r(entropy, aliasing)
  Equivalently: temporal_concentration = 1 - H/log(T)  →  positive r with aliasing

This is the label-space Nyquist test: we directly measure how concentrated
the model's temporal attention is, rather than using pixel motion as a proxy.

Output: evaluations/accv2026/e3_spectral/timesformer_attention_entropy.csv
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
from transformers import AutoModelForVideoClassification, AutoImageProcessor

ROOT      = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "evaluations/accv2026/manifests"
TAXONOMY  = ROOT / "evaluations/accv2026/e5_taxonomy"
OUT       = ROOT / "evaluations/accv2026/e3_spectral"
CKPT      = ROOT / "fine_tuned_models/accv2026_timesformer_finegym"

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BATCH     = 32
N_FRAMES  = 8   # TimeSformer default
MAX_VID   = 5   # videos per class


# ── load model ───────────────────────────────────────────────────────────────
print(f"Loading TimeSformer from {CKPT} on {DEVICE} …")
proc  = AutoImageProcessor.from_pretrained(CKPT)
model = AutoModelForVideoClassification.from_pretrained(CKPT).to(DEVICE)
model.eval()

T = model.config.num_frames   # = 8
log_T = float(np.log(T))


# ── register attn_drop hooks on all blocks ───────────────────────────────────
# Each hook captures temporal attention weights [N_patches, heads, T, T]
# from one block. We store them in a list-of-lists indexed by block.
n_blocks    = len(model.timesformer.encoder.layer)
block_attns = [[] for _ in range(n_blocks)]
hooks       = []

for blk_idx, blk in enumerate(model.timesformer.encoder.layer):
    def make_hook(idx):
        def hook(m, inp, out):
            block_attns[idx].append(inp[0].detach().cpu())
        return hook
    hooks.append(
        blk.temporal_attention.attention.attn_drop.register_forward_hook(
            make_hook(blk_idx)
        )
    )


def clear_attns():
    for lst in block_attns:
        lst.clear()


def compute_entropy_from_hooks() -> float:
    """Average temporal attention entropy (nats) over all captured blocks."""
    entropies = []
    for lst in block_attns:
        for attn in lst:   # attn: [N, heads, T, T]
            eps = 1e-8
            H = -(attn * torch.log(attn + eps)).sum(-1)   # [N, heads, T]
            entropies.append(H.mean().item())
    return float(np.mean(entropies)) if entropies else float("nan")


# ── decode videos ─────────────────────────────────────────────────────────────
def decode_clip(path: str) -> list | None:
    """Return a list of N_FRAMES RGB numpy arrays (H×W×3), or None on error."""
    try:
        from decord import VideoReader, cpu as dec_cpu
        vr = VideoReader(str(path), ctx=dec_cpu(0))
        total = len(vr)
        if total < 2:
            return None
        idxs = np.linspace(0, total - 1, N_FRAMES).astype(int)
        frames = vr.get_batch(idxs).asnumpy()
        return [frames[i] for i in range(len(frames))]
    except Exception:
        pass
    # fallback: PyAV
    try:
        import av
        all_frames = []
        with av.open(str(path)) as container:
            for frame in container.decode(container.streams.video[0]):
                all_frames.append(frame.to_ndarray(format="rgb24"))
        if len(all_frames) < 2:
            return None
        idxs = np.linspace(0, len(all_frames) - 1, N_FRAMES).astype(int)
        return [all_frames[i] for i in idxs]
    except Exception:
        return None


# ── load data ────────────────────────────────────────────────────────────────
manifest_path = MANIFESTS / "finegym_val_20_per_class.csv"
taxonomy_path = TAXONOMY / "finegym_class_taxonomy.csv"

manifest = pd.read_csv(manifest_path)
manifest = manifest[manifest["split"].isin(["val", "validation", "test"])].copy()
if "exists" in manifest.columns:
    manifest = manifest[manifest["exists"] == True].copy()
taxonomy = pd.read_csv(taxonomy_path)

classes = sorted(manifest["label_id"].unique())
print(f"FineGym: {len(classes)} classes, {len(manifest)} val clips")

# ── per-class loop ────────────────────────────────────────────────────────────
rows = []
for cls_idx, label_id in enumerate(classes):
    paths = manifest[manifest["label_id"] == label_id]["video_path"].tolist()
    sample = paths[:MAX_VID]

    clip_entropies = []
    clip_concentrations = []

    # Decode all clips for this class and batch them
    batch_frames = []
    for p in sample:
        frames = decode_clip(str(p))
        if frames is not None:
            batch_frames.append(frames)

    if not batch_frames:
        continue

    # Process in mini-batches
    for i in range(0, len(batch_frames), BATCH):
        mini = batch_frames[i : i + BATCH]
        clear_attns()
        inputs = proc(mini, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        H = compute_entropy_from_hooks()
        clip_entropies.append(H)
        clip_concentrations.append(1.0 - H / log_T)

    if clip_entropies:
        rows.append({
            "label_id":              label_id,
            "n_videos":              len(clip_entropies),
            "mean_entropy":          float(np.mean(clip_entropies)),
            "mean_concentration":    float(np.mean(clip_concentrations)),
            "std_entropy":           float(np.std(clip_entropies)),
        })

    if (cls_idx + 1) % 20 == 0 or cls_idx == len(classes) - 1:
        print(f"  [{cls_idx+1}/{len(classes)}] processed {len(rows)} classes so far")

for h in hooks:
    h.remove()

# ── save and correlate ────────────────────────────────────────────────────────
attn_df = pd.DataFrame(rows)
attn_df.to_csv(OUT / "timesformer_attention_entropy.csv", index=False)

merged = attn_df.merge(taxonomy[["label_id", "mean_abs_drop"]], on="label_id", how="inner")
y = merged["mean_abs_drop"].values

print(f"\nn = {len(merged)} classes merged with taxonomy")
print()
print("Temporal attention entropy vs aliasing (Pearson r):")
print("  Prediction: concentrated attention → more aliasing")
print()

for col, desc in [
    ("mean_entropy",       "entropy         (lower = more concentrated)"),
    ("mean_concentration", "concentration   (higher = more concentrated)"),
]:
    r, p = stats.pearsonr(merged[col].values, y)
    sig = "✅" if p < 0.05 else "  "
    print(f"  {desc:50s}  r={r:+.3f}  p={p:.4f}  {sig}")

print()
print(f"Max possible entropy (uniform over {T} frames): {log_T:.4f}")
print(f"Mean entropy observed: {merged['mean_entropy'].mean():.4f}")
print(f"Mean concentration observed: {merged['mean_concentration'].mean():.4f}")
print(f"\nSaved: {OUT}/timesformer_attention_entropy.csv")
