"""Sliding-window confidence concentration — todos os 7 modelos, todos os 7 datasets.

Para cada modelo fine-tunado num dataset:
  1. Decodifica N_DECODE=48 frames de cada vídeo
  2. Desliza uma janela de T frames (T = tamanho de clip do modelo)
     com stride adaptado para dar ~10 janelas por vídeo
  3. Roda o modelo em cada posição → P(classe correta) por janela
  4. Concentração = 1 - H(P) / log(n_janelas)

A concentração mede QUANDO o modelo fica confiante no vídeo.
Alta concentração → evidência em momentos específicos → mais aliasing.

Outputs (evaluations/accv2026/e3_spectral/):
  sw_{model}_{dataset}.csv                  por modelo × dataset
  sw_ensemble_{dataset}.csv                 média entre modelos
  sw_all_models_master.csv                  tabela de correlações final
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import json
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
N_DECODE  = 48   # frames decodificados por vídeo
N_WINDOWS = 10   # janelas alvo por vídeo
MAX_VID   = 5    # vídeos por classe


# ── helpers ───────────────────────────────────────────────────────────────────
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
    """1 - H(normalized conf) / log(n_windows)."""
    c = np.clip(np.array(conf, dtype=float), 1e-8, None)
    c /= c.sum()
    H = float(-(c * np.log(c)).sum())
    return 1.0 - H / np.log(len(c))


def partial_r(x, y, z):
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return stats.pearsonr(rx, ry)


# ── sliding window per model family ───────────────────────────────────────────
def sliding_windows_hf(model, proc, all_frames, T, stride, n_windows, label_id):
    """Retorna lista de P(label_id) para cada janela — modelos HF.
    Para T>=16 processa janelas individualmente para evitar OOM (e.g. ViViT T=32).
    """
    if T >= 16:
        # processa janela a janela — mais lento mas seguro para modelos com T grande
        conf = []
        for w in range(n_windows):
            window = all_frames[w*stride : w*stride + T]
            inputs = proc([window], return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                prob = model(**inputs).logits.softmax(-1).cpu().numpy()
            conf.append(float(prob[0, label_id]))
        return conf
    # T<16: batch de todas as janelas de uma vez
    windows = [all_frames[w*stride : w*stride + T] for w in range(n_windows)]
    inputs  = proc(windows, return_tensors="pt")
    inputs  = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(-1).cpu().numpy()
    return [float(probs[w, label_id]) for w in range(n_windows)]


def sliding_windows_cnn(model, proc, all_frames, T, stride, n_windows, label_id):
    """Retorna lista de P(label_id) para cada janela — CNNs TorchVision."""
    confs = []
    for w in range(n_windows):
        window = all_frames[w*stride : w*stride + T]
        inputs = proc(window)
        pv = inputs["pixel_values"].to(DEVICE)
        with torch.no_grad():
            prob = model(pv).logits.softmax(-1).cpu().numpy()
        confs.append(float(prob[0, label_id]))
    return confs


def sliding_windows_slowfast(model, proc, all_frames, T, stride, n_windows, label_id):
    """Retorna lista de P(label_id) para cada janela — SlowFast."""
    confs = []
    for w in range(n_windows):
        window = all_frames[w*stride : w*stride + T]
        inputs = proc(window).to(DEVICE)
        with torch.no_grad():
            prob = model(
                slow_frames=inputs["slow_frames"],
                fast_frames=inputs["fast_frames"],
            ).logits.softmax(-1).cpu().numpy()
        confs.append(float(prob[0, label_id]))
    return confs


# ── load model ────────────────────────────────────────────────────────────────
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


# ── main ──────────────────────────────────────────────────────────────────────
master_rows = []

for ds, manifest_fn in DATASETS.items():
    print(f"\n{'='*65}")
    print(f"Dataset: {ds}")
    print(f"{'='*65}")

    manifest_path = MANIFESTS / manifest_fn
    taxonomy_path = TAXONOMY  / f"{ds}_class_taxonomy.csv"
    if not manifest_path.exists() or not taxonomy_path.exists():
        print("  manifest ou taxonomy ausente — skip"); continue

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["split"].isin(["val","validation","test"])].copy()
    if "exists" in manifest.columns:
        manifest = manifest[manifest["exists"]==True].copy()
    taxonomy = pd.read_csv(taxonomy_path)
    classes  = sorted(manifest["label_id"].unique())
    print(f"  {len(classes)} classes")

    # acumulador: label_id → lista de concentrações (uma por modelo)
    class_conc = {c: [] for c in classes}

    for model_key, mcfg in MODELS.items():
        ckpt = find_ckpt(mcfg["prefix"], ds)
        if ckpt is None:
            print(f"  [{model_key}] checkpoint não encontrado — skip")
            continue

        print(f"  [{model_key}] {ckpt.name} …")
        try:
            model, proc = load_model(model_key, ckpt)
        except Exception as e:
            print(f"  [{model_key}] erro ao carregar: {e} — skip"); continue

        T      = mcfg["T"]
        family = mcfg["family"]
        # stride para dar ~N_WINDOWS janelas com N_DECODE frames
        stride    = max(1, (N_DECODE - T) // (N_WINDOWS - 1))
        n_windows = (N_DECODE - T) // stride + 1
        print(f"    T={T}, stride={stride}, janelas={n_windows}")

        rows_model = []
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
                except Exception:
                    continue

            if concs:
                mean_conc = float(np.mean(concs))
                rows_model.append({"label_id": label_id, "n_videos": len(concs),
                                   "mean_concentration": mean_conc,
                                   "std_concentration": float(np.std(concs))})
                class_conc[label_id].append(mean_conc)

            if (cls_idx+1) % 30 == 0 or cls_idx == len(classes)-1:
                print(f"    [{cls_idx+1}/{len(classes)}] {len(rows_model)} classes")

        if rows_model:
            df_m = pd.DataFrame(rows_model)
            df_m.to_csv(OUT / f"sw_{model_key}_{ds}.csv", index=False)
            print(f"    salvo sw_{model_key}_{ds}.csv  ({len(df_m)} classes)")

        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()

    # ensemble por dataset
    ens_rows = []
    for label_id in classes:
        cs = class_conc[label_id]
        if cs:
            ens_rows.append({"label_id": label_id,
                             "n_models": len(cs),
                             "ensemble_concentration": float(np.mean(cs)),
                             "std_across_models": float(np.std(cs))})

    df_ens = pd.DataFrame(ens_rows)
    df_ens.to_csv(OUT / f"sw_ensemble_{ds}.csv", index=False)
    n_m_avg = df_ens["n_models"].mean() if not df_ens.empty else 0
    print(f"\n  Ensemble: {len(df_ens)} classes, {n_m_avg:.1f} modelos avg")

    # correlação com aliasing
    y_col = "mean_abs_drop"
    if y_col not in taxonomy.columns:
        print("  sem mean_abs_drop na taxonomy — skip corr"); continue

    merged = df_ens.merge(taxonomy[["label_id", y_col]], on="label_id")
    if len(merged) < 10:
        print("  classes insuficientes para correlação"); continue

    x = merged["ensemble_concentration"].values
    y = merged[y_col].values
    r_sp, p_sp = stats.spearmanr(x, y)
    r_pe, p_pe = stats.pearsonr(x, y)
    print(f"  Spearman ρ = {r_sp:+.3f}  p={p_sp:.4f}")
    print(f"  Pearson  r = {r_pe:+.3f}  p={p_pe:.4f}  n={len(merged)}")

    master_rows.append({
        "dataset":     ds,
        "n_classes":   len(merged),
        "n_models_avg": round(n_m_avg, 1),
        "spearman_rho": round(r_sp, 3),
        "spearman_p":   round(p_sp, 4),
        "pearson_r":    round(r_pe, 3),
        "pearson_p":    round(p_pe, 4),
    })

# ── tabela master ─────────────────────────────────────────────────────────────
master = pd.DataFrame(master_rows)
master_path = OUT / "sw_all_models_master.csv"
master.to_csv(master_path, index=False)

print(f"\n{'='*70}")
print("SLIDING WINDOW — TODOS OS MODELOS — TABELA FINAL")
print(f"{'='*70}")
print(master.to_string(index=False))

if not master.empty:
    print(f"\nPooled (média não-ponderada entre datasets):")
    print(f"  Spearman ρ = {master['spearman_rho'].mean():+.3f}")
    print(f"  Pearson  r = {master['pearson_r'].mean():+.3f}")
    print(f"  Datasets   = {len(master)}")
print(f"\nSalvo: {master_path}")
