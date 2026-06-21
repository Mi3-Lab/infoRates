"""Option C — Sliding-window confidence profile as Nyquist test.

For each FineGym class, slides a temporal window of width W across a densely
decoded clip, runs the fine-tuned TimeSformer at each window position, and
records P(correct class). The temporal profile of model confidence tells us
WHERE in the clip the discriminative evidence resides.

If confidence is concentrated in a few windows → high temporal demand → more
aliasing. If confidence is uniform across windows → action recognisable from
any time point → less aliasing.

Feature: temporal confidence entropy (nats). Lower entropy = more concentrated
= should alias more.  Equivalently, temporal_concentration = 1 - H/log(W_count).

Output: evaluations/accv2026/e3_spectral/sliding_window_confidence.csv
"""
import warnings; warnings.filterwarnings("ignore")
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

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
T          = 8      # frames per window (TimeSformer input size)
N_DECODE   = 48     # frames decoded per clip (temporal resolution)
WIN_STRIDE = 4      # window hop (N_DECODE=48, T=8, stride=4 → 11 windows)
MAX_VID    = 5      # videos per class


# ── load model ───────────────────────────────────────────────────────────────
print(f"Loading TimeSformer from {CKPT} on {DEVICE} …")
proc  = AutoImageProcessor.from_pretrained(CKPT)
model = AutoModelForVideoClassification.from_pretrained(CKPT).to(DEVICE)
model.eval()

n_windows = (N_DECODE - T) // WIN_STRIDE + 1
log_W = float(np.log(n_windows))
print(f"Window config: N_DECODE={N_DECODE}, T={T}, stride={WIN_STRIDE} → {n_windows} windows/clip")


# ── decode ────────────────────────────────────────────────────────────────────
def decode_clip(path: str, n_frames: int) -> list | None:
    """Return list of n_frames RGB arrays decoded uniformly, or None."""
    try:
        from decord import VideoReader, cpu as dec_cpu
        vr = VideoReader(str(path), ctx=dec_cpu(0))
        total = len(vr)
        if total < 2:
            return None
        idxs = np.linspace(0, total - 1, n_frames).astype(int)
        frames = vr.get_batch(idxs).asnumpy()
        return [frames[i] for i in range(len(frames))]
    except Exception:
        pass
    try:
        import av
        all_frames = []
        with av.open(str(path)) as container:
            for frame in container.decode(container.streams.video[0]):
                all_frames.append(frame.to_ndarray(format="rgb24"))
        if len(all_frames) < 2:
            return None
        idxs = np.linspace(0, len(all_frames) - 1, n_frames).astype(int)
        return [all_frames[i] for i in idxs]
    except Exception:
        return None


def window_confidence(all_frames: list, true_label: int) -> list[float]:
    """Run model on each sliding window and return P(true_label) per window.

    all_frames: list of N_DECODE RGB arrays
    Returns: list of n_windows probabilities.
    """
    # Build all windows as a single batch
    windows = []
    for w in range(n_windows):
        start = w * WIN_STRIDE
        windows.append(all_frames[start : start + T])

    # batch: list of (list of T frames)
    inputs = proc(windows, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits             # [n_windows, n_classes]
        probs  = logits.softmax(dim=-1).cpu().numpy()

    return [float(probs[w, true_label]) for w in range(n_windows)]


def temporal_entropy(conf: list[float]) -> float:
    """Shannon entropy of a confidence profile (normalised to sum-to-1)."""
    c = np.array(conf, dtype=float)
    c = np.clip(c, 1e-8, None)
    c /= c.sum()
    return float(-(c * np.log(c)).sum())


# ── load data ────────────────────────────────────────────────────────────────
manifest_path = MANIFESTS / "finegym_val_20_per_class.csv"
taxonomy_path = TAXONOMY / "finegym_class_taxonomy.csv"

manifest = pd.read_csv(manifest_path)
manifest = manifest[manifest["split"].isin(["val", "validation", "test"])].copy()
if "exists" in manifest.columns:
    manifest = manifest[manifest["exists"] == True].copy()
taxonomy = pd.read_csv(taxonomy_path)

classes = sorted(manifest["label_id"].unique())
print(f"FineGym: {len(classes)} classes, {len(manifest)} val clips\n")

# ── per-class loop ────────────────────────────────────────────────────────────
rows = []
for cls_idx, label_id in enumerate(classes):
    paths = manifest[manifest["label_id"] == label_id]["video_path"].tolist()
    sample = paths[:MAX_VID]

    entropies     = []
    concentrations = []
    peak_windows  = []   # window index with highest confidence

    for p in sample:
        frames = decode_clip(str(p), N_DECODE)
        if frames is None or len(frames) < T:
            continue

        conf = window_confidence(frames, true_label=label_id)
        H    = temporal_entropy(conf)
        entropies.append(H)
        concentrations.append(1.0 - H / log_W)
        peak_windows.append(int(np.argmax(conf)))

    if not entropies:
        continue

    rows.append({
        "label_id":           label_id,
        "n_videos":           len(entropies),
        "mean_entropy":       float(np.mean(entropies)),
        "mean_concentration": float(np.mean(concentrations)),
        "mean_peak_window":   float(np.mean(peak_windows)),
        "std_entropy":        float(np.std(entropies)),
    })

    if (cls_idx + 1) % 20 == 0 or cls_idx == len(classes) - 1:
        print(f"  [{cls_idx+1}/{len(classes)}] {len(rows)} classes done")

# ── save and correlate ────────────────────────────────────────────────────────
sw_df = pd.DataFrame(rows)
sw_df.to_csv(OUT / "sliding_window_confidence.csv", index=False)

merged = sw_df.merge(taxonomy[["label_id", "mean_abs_drop"]], on="label_id", how="inner")
y = merged["mean_abs_drop"].values

print(f"\nn = {len(merged)} classes merged with taxonomy")
print()
print("Sliding-window temporal confidence vs aliasing (Pearson r):")
print("  Prediction: concentrated confidence → more aliasing")
print()

for col, desc in [
    ("mean_entropy",       "confidence entropy       (lower = more concentrated)"),
    ("mean_concentration", "confidence concentration (higher = more concentrated)"),
    ("mean_peak_window",   "mean peak window index   (earlier peak → ?)"),
]:
    r, p = stats.pearsonr(merged[col].values, y)
    sig = "✅" if p < 0.05 else "  "
    print(f"  {desc:52s}  r={r:+.3f}  p={p:.4f}  {sig}")

print()
print(f"Max possible entropy (uniform over {n_windows} windows): {log_W:.4f}")
print(f"Mean entropy observed: {merged['mean_entropy'].mean():.4f}")
print(f"Mean concentration observed: {merged['mean_concentration'].mean():.4f}")
print(f"\nSaved: {OUT}/sliding_window_confidence.csv")
