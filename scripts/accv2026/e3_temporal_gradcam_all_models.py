"""Temporal GradCAM saliency — all 7 models, all 7 datasets.

Why all models?  Temporal attention entropy (Option B) is specific to
Transformers.  Gradient-based temporal saliency (GradCAM on the time axis)
works for ALL architectures — CNNs, Transformers, and SlowFast — using the
same mathematical quantity:

    saliency[t] = mean_over(C,H,W) |d(logit_correct)/d(pixel_values[t])|

The normalized saliency vector is treated as a probability distribution.
Its concentration (1 - H/log(T)) measures how temporally localised the
discriminative evidence is.  Averaging concentration across ALL 7 models
per class gives a model-agnostic temporal demand signal that we then
correlate with aliasing (mean_abs_drop from taxonomy).

Models handled:
  Transformers (HF format): timesformer, videomae, vivit
      pixel_values shape: [B, T, C, H, W]  → grad axes (0,2,3,4) → [T]
  CNNs (TorchVision format): r3d_18, mc3_18, r2plus1d_18
      pixel_values shape: [B, C, T, H, W]  → grad axes (0,1,3,4) → [T]
  SlowFast (PyTorchVideo format): slowfast_r50
      slow_frames: [B,C,8,H,W]  fast_frames: [B,C,32,H,W]
      → interpolate slow to 32 frames + average with fast → [32]
  VideoMamba: SKIPPED (needs separate .venv_mamba environment)

Outputs (evaluations/accv2026/e3_spectral/):
  {model}_{dataset}_temporal_saliency.csv      per-model per-class
  {dataset}_temporal_saliency_ensemble.csv     mean across available models
  nyquist_temporal_saliency_master.csv         cross-dataset correlation table
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF
from scipy import stats
from transformers import AutoImageProcessor, AutoModelForVideoClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from info_rates.models.torchvision_video import (
    load_torchvision_video_checkpoint, TorchvisionVideoProcessor,
)
from info_rates.models.slowfast_video import load_slowfast_checkpoint

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/scratch/wesleyferreiramaia/infoRates")
CKPT_ROOT = DATA_ROOT / "fine_tuned_models"
MANIFESTS = ROOT / "evaluations/accv2026/manifests"
TAXONOMY  = ROOT / "evaluations/accv2026/e5_taxonomy"
OUT       = ROOT / "evaluations/accv2026/e3_spectral"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── model config ──────────────────────────────────────────────────────────────
# family: "hf" | "cnn" | "slowfast"
# T:      frames fed to model in one forward pass
MODELS = {
    "timesformer":  {"family": "hf",       "prefix": "accv2026_timesformer",  "T": 8},
    "videomae":     {"family": "hf",       "prefix": "accv2026_videomae",     "T": 16},
    "vivit":        {"family": "hf",       "prefix": "accv2026_vivit",        "T": 32},
    "r3d_18":       {"family": "cnn",      "prefix": "accv2026_r3d_18",       "T": 16},
    "mc3_18":       {"family": "cnn",      "prefix": "accv2026_mc3_18",       "T": 16},
    "r2plus1d_18":  {"family": "cnn",      "prefix": "accv2026_r2plus1d_18",  "T": 16},
    "slowfast_r50": {"family": "slowfast", "prefix": "accv2026_slowfast_r50", "T": 32},
}

DATASETS = {
    "ucf101":        "ucf101_val_20_per_class.csv",
    "ssv2":          "somethingv2_val_20_per_class.csv",
    "hmdb51":        "hmdb51_val_20_per_class.csv",
    "diving48":      "diving48_val_20_per_class.csv",
    "autsl":         "autsl_val_20_per_class.csv",
    "driveact":      "driveact_val_20_per_class.csv",
    "epic_kitchens": "epic_kitchens_val_20_per_class.csv",
}

CKPT_SUFFIXES = ["_full_e10_h200", "_full_e10_a100", "_full_e10_h200_v2"]
MAX_VID = 5
N_DECODE = 32  # decode this many frames; each model sub-samples to its T


# ── checkpoint discovery ──────────────────────────────────────────────────────
def find_ckpt(prefix, ds):
    for sfx in CKPT_SUFFIXES:
        p = CKPT_ROOT / f"{prefix}_{ds}{sfx}"
        if p.exists() and (p / "config.json").exists():
            return p
    return None


# ── video decode ──────────────────────────────────────────────────────────────
def decode_frames(path, n):
    try:
        from decord import VideoReader, cpu as dec_cpu
        vr = VideoReader(str(path), ctx=dec_cpu(0))
        total = len(vr)
        if total < 2: return None
        idxs = np.linspace(0, total-1, n).astype(int)
        arr = vr.get_batch(idxs).asnumpy()
        return [arr[i] for i in range(n)]
    except Exception:
        pass
    try:
        import av
        frames = []
        with av.open(str(path)) as c:
            for f in c.decode(c.streams.video[0]):
                frames.append(f.to_ndarray(format="rgb24"))
        if len(frames) < 2: return None
        idxs = np.linspace(0, len(frames)-1, n).astype(int)
        return [frames[i] for i in idxs]
    except Exception:
        return None


def subsample(frames, T):
    idxs = np.linspace(0, len(frames)-1, T).astype(int)
    return [frames[i] for i in idxs]


# ── concentration from saliency ───────────────────────────────────────────────
def saliency_concentration(sal):
    """1 - H(normalize(|sal|)) / log(T)."""
    s = np.abs(sal)
    s = np.clip(s, 1e-8, None)
    s /= s.sum()
    H = float(-(s * np.log(s)).sum())
    log_T = float(np.log(len(s)))
    return 1.0 - H / log_T


# ── GradCAM: HF Transformers ─────────────────────────────────────────────────
def gradcam_hf(model, proc, frames, T, label_id):
    """Temporal GradCAM for timesformer / videomae / vivit.
    pixel_values shape: [B, T, C, H, W]
    """
    frames_T = subsample(frames, T)
    try:
        inputs = proc(frames_T, return_tensors="pt")
        pv = inputs["pixel_values"].to(DEVICE)  # [1, T, C, H, W]
        pv.requires_grad_(True)
        model.zero_grad()
        out = model(pixel_values=pv)
        out.logits[0, label_id].backward()
        if pv.grad is None:
            return None
        sal = pv.grad.abs().mean(dim=(0, 2, 3, 4))  # [T]
        return sal.detach().cpu().numpy()
    except Exception as e:
        return None


# ── GradCAM: TorchVision CNNs ────────────────────────────────────────────────
def gradcam_cnn(model, proc, frames, T, label_id):
    """Temporal GradCAM for r3d_18 / mc3_18 / r2plus1d_18.
    pixel_values shape: [B, C, T, H, W]
    """
    frames_T = subsample(frames, T)
    try:
        inputs = proc(frames_T)
        pv = inputs["pixel_values"].to(DEVICE)  # [1, C, T, H, W]
        pv.requires_grad_(True)
        model.zero_grad()
        out = model(pv)
        out.logits[0, label_id].backward()
        if pv.grad is None:
            return None
        sal = pv.grad.abs().mean(dim=(0, 1, 3, 4))  # [T]
        return sal.detach().cpu().numpy()
    except Exception as e:
        return None


# ── GradCAM: SlowFast ─────────────────────────────────────────────────────────
def gradcam_slowfast(model, proc, frames, label_id):
    """Temporal GradCAM for slowfast_r50.
    slow: [B,C,8,H,W]  fast: [B,C,32,H,W]
    Returns unified [32]-length saliency (interpolated slow + fast).
    """
    try:
        inputs = proc(frames)
        slow = inputs["slow_frames"].to(DEVICE)   # [1, C, 8, H, W]
        fast = inputs["fast_frames"].to(DEVICE)   # [1, C, 32, H, W]
        slow.requires_grad_(True)
        fast.requires_grad_(True)
        model.zero_grad()
        out = model(slow_frames=slow, fast_frames=fast)
        out.logits[0, label_id].backward()
        if slow.grad is None or fast.grad is None:
            return None
        sal_slow = slow.grad.abs().mean(dim=(0, 1, 3, 4))  # [8]
        sal_fast = fast.grad.abs().mean(dim=(0, 1, 3, 4))  # [32]
        sal_slow_up = TF.interpolate(
            sal_slow[None, None].float(), size=32, mode="linear", align_corners=False
        )[0, 0]
        sal = (sal_slow_up + sal_fast).detach().cpu().numpy() / 2.0  # [32]
        return sal
    except Exception as e:
        return None


# ── load model by family ──────────────────────────────────────────────────────
def load_model(model_key: str, ckpt: Path):
    cfg = MODELS[model_key]
    family = cfg["family"]
    if family == "hf":
        proc  = AutoImageProcessor.from_pretrained(str(ckpt))
        model = AutoModelForVideoClassification.from_pretrained(str(ckpt)).to(DEVICE)
        model.eval()
        return model, proc
    elif family == "cnn":
        model, _, config = load_torchvision_video_checkpoint(ckpt, DEVICE)
        model.eval()
        proc = TorchvisionVideoProcessor(size=int(config.get("input_size", 112)))
        return model, proc
    elif family == "slowfast":
        model, proc, _ = load_slowfast_checkpoint(ckpt, DEVICE)
        model.eval()
        return model, proc
    else:
        raise ValueError(f"Unknown family: {family}")


# ── partial correlation ────────────────────────────────────────────────────────
def partial_r(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    bx = np.polyfit(z, x, 1); rx = x - np.polyval(bx, z)
    by = np.polyfit(z, y, 1); ry = y - np.polyval(by, z)
    return stats.pearsonr(rx, ry)


# ── main: iterate datasets → models → classes → videos ───────────────────────
master_rows = []

for ds, manifest_fn in DATASETS.items():
    print(f"\n{'='*65}")
    print(f"  Dataset: {ds}")
    print(f"{'='*65}")

    manifest_path = MANIFESTS / manifest_fn
    taxonomy_path = TAXONOMY  / f"{ds}_class_taxonomy.csv"
    if not manifest_path.exists() or not taxonomy_path.exists():
        print("  manifest or taxonomy missing — skip"); continue

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["split"].isin(["val","validation","test"])].copy()
    if "exists" in manifest.columns:
        manifest = manifest[manifest["exists"]==True].copy()
    taxonomy = pd.read_csv(taxonomy_path)

    classes = sorted(manifest["label_id"].unique())
    print(f"  {len(classes)} classes")

    # per-class accumulator: {label_id: [concentration_model1, ...]}
    class_conc: dict[int, list[float]] = {c: [] for c in classes}

    for model_key, mcfg in MODELS.items():
        ckpt = find_ckpt(mcfg["prefix"], ds)
        if ckpt is None:
            print(f"  [{model_key}] checkpoint not found — skip")
            continue

        print(f"  [{model_key}] loading {ckpt.name} …")
        try:
            model, proc = load_model(model_key, ckpt)
        except Exception as e:
            print(f"  [{model_key}] load error: {e} — skip")
            continue

        T      = mcfg["T"]
        family = mcfg["family"]
        rows_model = []

        for cls_idx, label_id in enumerate(classes):
            paths = [DATA_ROOT / p
                     for p in manifest[manifest["label_id"]==label_id]["video_path"].tolist()]
            paths = [p for p in paths if p.exists()][:MAX_VID]
            if not paths:
                continue

            concs = []
            for p in paths:
                frames = decode_frames(p, N_DECODE)
                if frames is None: continue

                if family == "hf":
                    sal = gradcam_hf(model, proc, frames, T, label_id)
                elif family == "cnn":
                    sal = gradcam_cnn(model, proc, frames, T, label_id)
                else:
                    sal = gradcam_slowfast(model, proc, frames, label_id)

                if sal is not None and np.isfinite(sal).all() and sal.sum() > 0:
                    concs.append(saliency_concentration(sal))

            if concs:
                mean_conc = float(np.mean(concs))
                rows_model.append({"label_id": label_id, "n_videos": len(concs),
                                   "mean_concentration": mean_conc,
                                   "std_concentration": float(np.std(concs))})
                class_conc[label_id].append(mean_conc)

            if (cls_idx+1) % 30 == 0 or cls_idx == len(classes)-1:
                print(f"    [{cls_idx+1}/{len(classes)}] {len(rows_model)} classes done")

        # save per-model CSV
        if rows_model:
            df_m = pd.DataFrame(rows_model)
            df_m.to_csv(OUT / f"{model_key}_{ds}_temporal_saliency.csv", index=False)
            print(f"    saved {len(df_m)} classes → {model_key}_{ds}_temporal_saliency.csv")

        # free GPU memory
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ensemble: mean across all models that have data per class
    ens_rows = []
    for label_id in classes:
        cs = class_conc[label_id]
        if cs:
            ens_rows.append({"label_id": label_id,
                             "n_models": len(cs),
                             "ensemble_concentration": float(np.mean(cs)),
                             "std_across_models": float(np.std(cs))})
    df_ens = pd.DataFrame(ens_rows)
    df_ens.to_csv(OUT / f"{ds}_temporal_saliency_ensemble.csv", index=False)
    print(f"\n  Ensemble: {len(df_ens)} classes across {df_ens['n_models'].mean():.1f} models avg")

    # correlate ensemble with aliasing
    if "mean_abs_drop" not in taxonomy.columns:
        print("  no mean_abs_drop column in taxonomy — skip corr"); continue

    merged = df_ens.merge(taxonomy[["label_id", "mean_abs_drop"]], on="label_id")
    if len(merged) < 10:
        print("  too few classes for correlation"); continue

    y = merged["mean_abs_drop"].values
    x = merged["ensemble_concentration"].values
    r_sp, p_sp = stats.spearmanr(x, y)
    r_pe, p_pe = stats.pearsonr(x, y)

    print(f"  Spearman ρ={r_sp:+.3f} p={p_sp:.4f}")
    print(f"  Pearson  r={r_pe:+.3f} p={p_pe:.4f}  n={len(merged)}")

    master_rows.append({
        "dataset":        ds,
        "n_classes":      len(merged),
        "n_models_avg":   round(df_ens["n_models"].mean(), 1),
        "spearman_rho":   round(r_sp, 3),
        "spearman_p":     round(p_sp, 4),
        "pearson_r":      round(r_pe, 3),
        "pearson_p":      round(p_pe, 4),
    })

# ── master table ──────────────────────────────────────────────────────────────
master = pd.DataFrame(master_rows)
master_path = OUT / "nyquist_temporal_saliency_master.csv"
master.to_csv(master_path, index=False)

print(f"\n{'='*70}")
print("TEMPORAL GRADCAM — MASTER CORRELATION TABLE (all models, all datasets)")
print(f"{'='*70}")
print(master.to_string(index=False))

if not master.empty:
    print(f"\nPooled (unweighted mean across datasets):")
    print(f"  Spearman ρ = {master['spearman_rho'].mean():+.3f}")
    print(f"  Pearson  r = {master['pearson_r'].mean():+.3f}")
    print(f"  Datasets   = {len(master)}")

print(f"\nSaved: {master_path}")
