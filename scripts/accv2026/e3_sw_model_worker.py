"""Sliding-window confidence concentration — 1 modelo × todos os datasets.

Aceita MODEL_KEY via argumento ou env var MODEL_KEY.
Pula CSVs que já existem (para retomar sem reprocessar).
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
from transformers import AutoImageProcessor, AutoModelForVideoClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from info_rates.models.torchvision_video import (
    load_torchvision_video_checkpoint, TorchvisionVideoProcessor,
)
from info_rates.models.slowfast_video import (
    load_slowfast_checkpoint, SLOWFAST_SLOW_FRAMES, SLOWFAST_FAST_FRAMES,
)

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/scratch/wesleyferreiramaia/infoRates")
CKPT_ROOT = DATA_ROOT / "fine_tuned_models"
MANIFESTS = ROOT / "evaluations/accv2026/manifests"
TAXONOMY  = ROOT / "evaluations/accv2026/e5_taxonomy"
OUT       = ROOT / "evaluations/accv2026/e3_spectral"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

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
N_DECODE  = 48
N_WINDOWS = 10
MAX_VID   = 5


def find_ckpt(prefix, ds):
    for sfx in CKPT_SUFFIXES:
        p = CKPT_ROOT / f"{prefix}_{ds}{sfx}"
        if p.exists() and (p / "config.json").exists():
            return p
    return None


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


def concentration(conf):
    c = np.clip(np.array(conf, dtype=float), 1e-8, None)
    c /= c.sum()
    H = float(-(c * np.log(c)).sum())
    return 1.0 - H / np.log(len(c))


def sliding_windows_hf(model, proc, all_frames, T, stride, n_windows, label_id):
    if T >= 16:
        conf = []
        for w in range(n_windows):
            window = all_frames[w*stride : w*stride + T]
            inputs = proc([window], return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                prob = model(**inputs).logits.float().softmax(-1).cpu().numpy()
            conf.append(float(prob[0, label_id]))
        return conf
    windows = [all_frames[w*stride : w*stride + T] for w in range(n_windows)]
    inputs  = proc(windows, return_tensors="pt")
    inputs  = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        probs = model(**inputs).logits.float().softmax(-1).cpu().numpy()
    return [float(probs[w, label_id]) for w in range(n_windows)]


def sliding_windows_cnn(model, proc, all_frames, T, stride, n_windows, label_id):
    confs = []
    for w in range(n_windows):
        window = all_frames[w*stride : w*stride + T]
        inputs = proc(window)
        pv = inputs["pixel_values"].to(DEVICE)
        with torch.no_grad():
            prob = model(pv).logits.float().softmax(-1).cpu().numpy()
        confs.append(float(prob[0, label_id]))
    return confs


def sliding_windows_slowfast(model, proc, all_frames, T, stride, n_windows, label_id):
    confs = []
    for w in range(n_windows):
        window = all_frames[w*stride : w*stride + T]
        inputs = proc(window).to(DEVICE)
        with torch.no_grad():
            prob = model(
                slow_frames=inputs["slow_frames"],
                fast_frames=inputs["fast_frames"],
            ).logits.float().softmax(-1).cpu().numpy()
        confs.append(float(prob[0, label_id]))
    return confs


def load_model(model_key, ckpt):
    family = MODELS[model_key]["family"]
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
    raise ValueError(f"Unknown family: {family}")


# ── main ─────────────────────────────────────────────────────────────────────
model_key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_KEY", "")
if not model_key or model_key not in MODELS:
    print(f"Uso: python e3_sw_model_worker.py <model_key>")
    print(f"Modelos disponíveis: {list(MODELS.keys())}")
    sys.exit(1)

mcfg   = MODELS[model_key]
family = mcfg["family"]
T      = mcfg["T"]
stride    = max(1, (N_DECODE - T) // (N_WINDOWS - 1))
n_windows = (N_DECODE - T) // stride + 1

print(f"\n{'='*65}")
print(f"Modelo: {model_key}  |  T={T}, stride={stride}, janelas={n_windows}")
print(f"{'='*65}")

for ds, manifest_fn in DATASETS.items():
    out_csv = OUT / f"sw_{model_key}_{ds}.csv"
    if out_csv.exists():
        df_existing = pd.read_csv(out_csv)
        if len(df_existing) > 0:
            print(f"\n[{ds}] já existe ({len(df_existing)} classes) — skip")
            continue

    print(f"\n[{ds}]")
    ckpt = find_ckpt(mcfg["prefix"], ds)
    if ckpt is None:
        print(f"  checkpoint não encontrado — skip"); continue

    manifest_path = MANIFESTS / manifest_fn
    if not manifest_path.exists():
        print(f"  manifest ausente — skip"); continue

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["split"].isin(["val","validation","test"])].copy()
    if "exists" in manifest.columns:
        manifest = manifest[manifest["exists"]==True].copy()
    classes = sorted(manifest["label_id"].unique())
    print(f"  {len(classes)} classes | ckpt: {ckpt.name}")

    try:
        model, proc = load_model(model_key, ckpt)
    except Exception as e:
        print(f"  erro ao carregar: {e} — skip"); continue

    rows = []
    for cls_idx, label_id in enumerate(classes):
        paths = [DATA_ROOT / p
                 for p in manifest[manifest["label_id"]==label_id]["video_path"].tolist()]
        paths = [p for p in paths if p.exists()][:MAX_VID]
        if not paths: continue

        concs = []
        for p in paths:
            frames = decode_frames(p, N_DECODE)
            if frames is None or len(frames) < T: continue
            try:
                if family == "hf":
                    conf = sliding_windows_hf(model, proc, frames, T, stride, n_windows, label_id)
                elif family == "cnn":
                    conf = sliding_windows_cnn(model, proc, frames, T, stride, n_windows, label_id)
                else:
                    conf = sliding_windows_slowfast(model, proc, frames, T, stride, n_windows, label_id)
                if conf:
                    concs.append(concentration(conf))
            except Exception as e:
                print(f"    WARN {type(e).__name__}: {str(e)[:100]}")
                continue

        if concs:
            rows.append({"label_id": label_id,
                         "n_videos": len(concs),
                         "mean_concentration": float(np.mean(concs)),
                         "std_concentration": float(np.std(concs))})

        if (cls_idx+1) % 30 == 0 or cls_idx == len(classes)-1:
            print(f"  [{cls_idx+1}/{len(classes)}] {len(rows)} classes")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  salvo {out_csv.name} ({len(df)} classes)")

    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()

print(f"\nDone: {model_key} completo.")
