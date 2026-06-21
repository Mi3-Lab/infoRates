"""Sliding-window confidence — VideoMamba (SSM), todos os 7 datasets.

Roda em .venv_mamba (necessário para mamba_ssm).
Salva sw_videomamba_{dataset}.csv no mesmo diretório dos outros modelos.
Após completar, rodar e3_sw_recompute_ensemble.py para atualizar o ensemble com 8 modelos.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/scratch/wesleyferreiramaia/infoRates")
CKPT_ROOT = DATA_ROOT / "fine_tuned_models"
MANIFESTS = ROOT / "evaluations/accv2026/manifests"
TAXONOMY  = ROOT / "evaluations/accv2026/e5_taxonomy"
OUT       = ROOT / "evaluations/accv2026/e3_spectral"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from info_rates.models.videomamba_model import load_videomamba_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

DATASETS = {
    "ucf101":        "ucf101_val_20_per_class.csv",
    "ssv2":          "somethingv2_val_20_per_class.csv",
    "hmdb51":        "hmdb51_val_20_per_class.csv",
    "diving48":      "diving48_val_20_per_class.csv",
    "autsl":         "autsl_val_20_per_class.csv",
    "driveact":      "driveact_val_20_per_class.csv",
    "epic_kitchens": "epic_kitchens_val_20_per_class.csv",
}

N_DECODE  = 48
N_WINDOWS = 10
MAX_VID   = 5


def find_ckpt(ds):
    p = CKPT_ROOT / f"accv2026_videomamba_{ds}_full_e10_h200"
    return p if p.exists() and (p / "accv_meta.json").exists() else None


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


for ds, manifest_fn in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {ds}")
    print(f"{'='*60}")

    ckpt = find_ckpt(ds)
    if ckpt is None:
        print("  checkpoint não encontrado — skip"); continue

    manifest_path = MANIFESTS / manifest_fn
    taxonomy_path = TAXONOMY  / f"{ds}_class_taxonomy.csv"
    if not manifest_path.exists():
        print("  manifest ausente — skip"); continue

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["split"].isin(["val","validation","test"])].copy()
    if "exists" in manifest.columns:
        manifest = manifest[manifest["exists"]==True].copy()
    classes = sorted(manifest["label_id"].unique())
    print(f"  {len(classes)} classes")

    print(f"  Carregando {ckpt.name} …")
    model, proc, meta = load_videomamba_checkpoint(ckpt, DEVICE)
    model.eval()

    T      = meta.get("num_frames", 8)
    stride = max(1, (N_DECODE - T) // (N_WINDOWS - 1))
    n_win  = (N_DECODE - T) // stride + 1
    print(f"  T={T}, stride={stride}, janelas={n_win}")

    rows = []
    for cls_idx, label_id in enumerate(classes):
        paths = [DATA_ROOT / p
                 for p in manifest[manifest["label_id"]==label_id]["video_path"].tolist()]
        paths = [p for p in paths if p.exists()][:MAX_VID]
        if not paths: continue

        concs = []
        _first_err_printed = False
        for p in paths:
            frames = decode_frames(p, N_DECODE)
            if frames is None or len(frames) < T: continue
            try:
                # processa janelas uma a uma para evitar OOM e obter logits corretos
                conf = []
                for w in range(n_win):
                    window = frames[w*stride : w*stride + T]
                    inputs = proc(window)          # single video → [1, C, T, H, W]
                    pv     = inputs["pixel_values"].to(DEVICE)
                    with torch.no_grad():
                        logits = model(pixel_values=pv).logits   # [1, num_classes]
                    prob = float(logits.float().softmax(-1).cpu().numpy()[0, label_id])
                    conf.append(prob)
                concs.append(concentration(conf))
            except Exception as e:
                if not _first_err_printed:
                    print(f"    ERRO ({type(e).__name__}): {str(e)[:150]}")
                    _first_err_printed = True
                continue

        if concs:
            rows.append({"label_id": label_id,
                         "n_videos": len(concs),
                         "mean_concentration": float(np.mean(concs)),
                         "std_concentration": float(np.std(concs))})

        if (cls_idx+1) % 30 == 0 or cls_idx == len(classes)-1:
            print(f"  [{cls_idx+1}/{len(classes)}] {len(rows)} classes")

    df = pd.DataFrame(rows)
    out_path = OUT / f"sw_videomamba_{ds}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Salvo: {out_path}  ({len(df)} classes)")

    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()

print("\nVideoMamba sliding window completo.")
print("Rode agora: python3 scripts/accv2026/e3_sw_recompute_ensemble.py")
