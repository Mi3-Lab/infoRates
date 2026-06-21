"""Sliding-window confidence concentration — todos os 8 modelos × FineGym (local).

Variante do e3_sw_model_worker.py adaptada para este PC:
  - Checkpoints em fine_tuned_models/ (local, não /scratch/)
  - Vídeos em data/FineGym_data/videos/ (local)
  - Pula sw_{model}_finegym.csv que já existem
  - Após todos os modelos, recomputa ensemble incluindo FineGym

Saída: evaluations/accv2026/e3_spectral/sw_{model}_finegym.csv
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT      = Path(__file__).resolve().parents[2]
CKPT_ROOT = ROOT / "fine_tuned_models"
MANIFESTS = ROOT / "evaluations/accv2026/manifests"
TAXONOMY  = ROOT / "evaluations/accv2026/e5_taxonomy"
OUT       = ROOT / "evaluations/accv2026/e3_spectral"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from info_rates.models.torchvision_video import (
    load_torchvision_video_checkpoint, TorchvisionVideoProcessor,
)
from info_rates.models.slowfast_video import (
    load_slowfast_checkpoint,
)
from transformers import AutoImageProcessor, AutoModelForVideoClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Checkpoint map: model → (family, ckpt_dir_name, T_frames) ────────────────
MODELS = {
    "timesformer":  ("hf",        "accv2026_timesformer_finegym",            8),
    "videomae":     ("hf",        "accv2026_videomae_finegym",              16),
    "vivit":        ("hf",        "accv2026_vivit_finegym",                 32),
    "r3d_18":       ("cnn",       "accv2026_r3d_18_finegym_full_e10_a100",  16),
    "mc3_18":       ("cnn",       "accv2026_mc3_18_finegym_full_e10_a100",  16),
    "r2plus1d_18":  ("cnn",       "accv2026_r2plus1d_18_finegym_full_e10_a100", 16),
    "slowfast_r50": ("slowfast",  "accv2026_slowfast_r50_finegym",          32),
    "videomamba":   ("videomamba","accv2026_videomamba_finegym",              8),
}

N_DECODE  = 48
N_WINDOWS = 10
MAX_VID   = 5


# ── helpers ───────────────────────────────────────────────────────────────────
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


def sliding_windows_hf(model, proc, frames, T, stride, n_win, label_id):
    if T >= 16:
        conf = []
        for w in range(n_win):
            window = frames[w*stride : w*stride + T]
            inp = proc([window], return_tensors="pt")
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            with torch.no_grad():
                p = model(**inp).logits.float().softmax(-1).cpu().numpy()
            conf.append(float(p[0, label_id]))
        return conf
    windows = [frames[w*stride : w*stride + T] for w in range(n_win)]
    inp = proc(windows, return_tensors="pt")
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        probs = model(**inp).logits.float().softmax(-1).cpu().numpy()
    return [float(probs[w, label_id]) for w in range(n_win)]


def sliding_windows_cnn(model, proc, frames, T, stride, n_win, label_id):
    conf = []
    for w in range(n_win):
        window = frames[w*stride : w*stride + T]
        inp = proc(window)
        pv = inp["pixel_values"].to(DEVICE)
        with torch.no_grad():
            p = model(pv).logits.float().softmax(-1).cpu().numpy()
        conf.append(float(p[0, label_id]))
    return conf


def sliding_windows_slowfast(model, proc, frames, T, stride, n_win, label_id):
    conf = []
    for w in range(n_win):
        window = frames[w*stride : w*stride + T]
        inp = proc(window).to(DEVICE)
        with torch.no_grad():
            p = model(
                slow_frames=inp["slow_frames"],
                fast_frames=inp["fast_frames"],
            ).logits.float().softmax(-1).cpu().numpy()
        conf.append(float(p[0, label_id]))
    return conf


def sliding_windows_videomamba(model, proc, frames, T, stride, n_win, label_id):
    conf = []
    for w in range(n_win):
        window = frames[w*stride : w*stride + T]
        inp = proc(window)
        pv = inp["pixel_values"].to(DEVICE)
        with torch.no_grad():
            out = model(pixel_values=pv)
            logits = out.logits if hasattr(out, "logits") else out
            p = logits.float().softmax(-1).cpu().numpy()
        conf.append(float(p[0, label_id]))
    return conf


def load_model(family, ckpt_path):
    if family == "hf":
        proc  = AutoImageProcessor.from_pretrained(str(ckpt_path))
        model = AutoModelForVideoClassification.from_pretrained(str(ckpt_path)).to(DEVICE)
        model.eval()
        return model, proc
    elif family == "cnn":
        model, _, config = load_torchvision_video_checkpoint(ckpt_path, DEVICE)
        model.eval()
        proc = TorchvisionVideoProcessor(size=int(config.get("input_size", 112)))
        return model, proc
    elif family == "slowfast":
        model, proc, _ = load_slowfast_checkpoint(ckpt_path, DEVICE)
        model.eval()
        return model, proc
    elif family == "videomamba":
        from info_rates.models.videomamba_model import load_videomamba_checkpoint
        model, proc, _ = load_videomamba_checkpoint(ckpt_path, DEVICE)
        model.eval()
        return model, proc
    raise ValueError(f"Unknown family: {family}")


# ── load manifest + taxonomy ──────────────────────────────────────────────────
manifest = pd.read_csv(MANIFESTS / "finegym_val_20_per_class.csv")
manifest = manifest[manifest["split"].isin(["val","validation","test"])].copy()
if "exists" in manifest.columns:
    manifest = manifest[manifest["exists"] == True].copy()
taxonomy = pd.read_csv(TAXONOMY / "finegym_class_taxonomy.csv")
classes  = sorted(manifest["label_id"].unique())
print(f"\nFineGym: {len(classes)} classes, {len(manifest)} val clips\n")

# ── run each model ────────────────────────────────────────────────────────────
for model_key, (family, ckpt_name, T) in MODELS.items():
    out_csv = OUT / f"sw_{model_key}_finegym.csv"
    if out_csv.exists():
        df_ex = pd.read_csv(out_csv)
        if len(df_ex) > 0:
            print(f"[{model_key}] já existe ({len(df_ex)} classes) — skip")
            continue

    ckpt_path = CKPT_ROOT / ckpt_name
    if not ckpt_path.exists():
        print(f"[{model_key}] checkpoint não encontrado: {ckpt_path} — skip")
        continue

    stride    = max(1, (N_DECODE - T) // (N_WINDOWS - 1))
    n_windows = (N_DECODE - T) // stride + 1
    print(f"\n{'='*65}")
    print(f"[{model_key}]  T={T}, stride={stride}, janelas={n_windows}")
    print(f"{'='*65}")

    try:
        model, proc = load_model(family, ckpt_path)
    except Exception as e:
        print(f"  ERRO ao carregar: {e} — skip")
        continue

    rows = []
    for cls_idx, label_id in enumerate(classes):
        paths = [ROOT / p
                 for p in manifest[manifest["label_id"] == label_id]["video_path"].tolist()]
        paths = [p for p in paths if p.exists()][:MAX_VID]
        if not paths:
            continue

        concs = []
        for p in paths:
            frames = decode_frames(p, N_DECODE)
            if frames is None or len(frames) < T:
                continue
            try:
                if family == "hf":
                    conf = sliding_windows_hf(model, proc, frames, T, stride, n_windows, label_id)
                elif family == "cnn":
                    conf = sliding_windows_cnn(model, proc, frames, T, stride, n_windows, label_id)
                elif family == "slowfast":
                    conf = sliding_windows_slowfast(model, proc, frames, T, stride, n_windows, label_id)
                else:
                    conf = sliding_windows_videomamba(model, proc, frames, T, stride, n_windows, label_id)
                if conf:
                    concs.append(concentration(conf))
            except Exception as e:
                print(f"    WARN {type(e).__name__}: {str(e)[:80]}")

        if concs:
            rows.append({
                "label_id":           label_id,
                "n_videos":           len(concs),
                "mean_concentration": float(np.mean(concs)),
                "std_concentration":  float(np.std(concs)),
            })

        if (cls_idx + 1) % 25 == 0 or cls_idx == len(classes) - 1:
            print(f"  [{cls_idx+1}/{len(classes)}] {len(rows)} classes")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  salvo {out_csv.name}  ({len(df)} classes)")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ── ensemble recompute ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("Recomputando ensemble incluindo FineGym …")
print("="*65)

import subprocess
result = subprocess.run(
    [sys.executable, str(ROOT / "scripts/accv2026/e3_sw_recompute_ensemble.py")],
    capture_output=False,
)
if result.returncode != 0:
    print("WARN: ensemble recompute retornou erro — verifique manualmente.")
