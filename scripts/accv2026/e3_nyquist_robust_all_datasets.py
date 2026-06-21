"""Nyquist robustness tests A, B, C — all 7 datasets on this machine.

Tests:
  A) Temporal burstiness of optical flow  (CPU, OpenCV)
     partial r(temporal_std | mean_flow) vs aliasing
  B) TimeSformer temporal attention entropy  (GPU)
     r(attention_concentration) vs aliasing
  C) Sliding-window confidence concentration  (GPU)
     partial r(confidence_concentration | mean_flow) vs aliasing  ← key result

Reads video files from DATA_ROOT (scratch), manifests and taxonomies from ROOT.
Saves per-dataset CSV + final merged table with all correlations.

Outputs (evaluations/accv2026/e3_spectral/):
  {dataset}_flow_stats_v2.csv
  {dataset}_attention_entropy_v2.csv
  {dataset}_sliding_window_confidence_v2.csv
  nyquist_robust_all_datasets.csv      ← master correlation table
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from scipy import stats
from transformers import AutoModelForVideoClassification, AutoImageProcessor

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

# ── dataset config ─────────────────────────────────────────────────────────────
DATASETS = {
    "ucf101":        ("ucf101_val_20_per_class.csv",          "accv2026_timesformer_ucf101_full_e10_h200"),
    "ssv2":          ("somethingv2_val_20_per_class.csv",     "accv2026_timesformer_ssv2_full_e10_h200"),
    "hmdb51":        ("hmdb51_val_20_per_class.csv",          "accv2026_timesformer_hmdb51_full_e10_h200"),
    "diving48":      ("diving48_val_20_per_class.csv",        "accv2026_timesformer_diving48_full_e10_h200"),
    "autsl":         ("autsl_val_20_per_class.csv",           "accv2026_timesformer_autsl_full_e10_h200"),
    "driveact":      ("driveact_val_20_per_class.csv",        "accv2026_timesformer_driveact_full_e10_h200"),
    "epic_kitchens": ("epic_kitchens_val_20_per_class.csv",   "accv2026_timesformer_epic_kitchens_full_e10_h200"),
}

# option A params
N_FRAMES_FLOW = 32   # frames decoded for flow (dense sample)
MAX_VID_FLOW  = 5

# option B/C params
N_FRAMES_TSF  = 8    # TimeSformer clip length
N_DECODE_WIN  = 48   # frames decoded for sliding window
WIN_STRIDE    = 4    # → (48-8)//4 + 1 = 11 windows
MAX_VID_GPU   = 5


# ── helpers: video decode ──────────────────────────────────────────────────────
def decode_uniform(path: Path, n: int):
    """Decode n frames uniformly sampled from video. Returns (n, H, W, 3) or None."""
    try:
        from decord import VideoReader, cpu as dec_cpu
        vr = VideoReader(str(path), ctx=dec_cpu(0))
        total = len(vr)
        if total < 2: return None
        idxs = np.linspace(0, total-1, n).astype(int)
        return vr.get_batch(idxs).asnumpy()
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
        return np.stack([frames[i] for i in idxs])
    except Exception:
        return None


# ── Option A: temporal flow burstiness ────────────────────────────────────────
def flow_features(video_path: Path):
    """Compute inter-frame optical flow magnitudes and temporal features."""
    frames_rgb = decode_uniform(video_path, N_FRAMES_FLOW)
    if frames_rgb is None: return None
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]
    mags = []
    for i in range(len(grays)-1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()))
    if len(mags) < 2: return None
    mags = np.array(mags)
    return {
        "mean_flow":    float(mags.mean()),
        "temporal_std": float(mags.std()),
        "temporal_cv":  float(mags.std() / (mags.mean() + 1e-8)),
        "peak_ratio":   float(mags.max() / (mags.mean() + 1e-8)),
    }


def run_option_a(ds: str, manifest: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    print(f"  [A] flow burstiness …")
    classes = sorted(manifest["label_id"].unique())
    rows = []
    for label_id in classes:
        paths = [DATA_ROOT / p for p in manifest[manifest["label_id"]==label_id]["video_path"].tolist()]
        feats = [flow_features(p) for p in paths[:MAX_VID_FLOW] if p.exists()]
        feats = [f for f in feats if f is not None]
        if not feats: continue
        row = {"dataset": ds, "label_id": label_id, "n_videos": len(feats)}
        for k in ["mean_flow","temporal_std","temporal_cv","peak_ratio"]:
            row[k] = float(np.mean([f[k] for f in feats]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{ds}_flow_stats_v2.csv", index=False)
    print(f"    saved {len(df)} classes")
    return df


# ── Option B/C: TimeSformer inference ─────────────────────────────────────────
def load_timesformer(ckpt_path: Path):
    proc  = AutoImageProcessor.from_pretrained(str(ckpt_path))
    model = AutoModelForVideoClassification.from_pretrained(str(ckpt_path)).to(DEVICE)
    model.eval()
    return proc, model


def register_attn_hooks(model):
    """Hook attn_drop in all temporal attention blocks. Returns (hooks, store)."""
    store = []
    hooks = []
    n_blocks = len(model.timesformer.encoder.layer)
    per_block = [[] for _ in range(n_blocks)]

    for blk_idx, blk in enumerate(model.timesformer.encoder.layer):
        def make_hook(idx):
            def fn(m, inp, out):
                per_block[idx].append(inp[0].detach().cpu())
            return fn
        hooks.append(blk.temporal_attention.attention.attn_drop.register_forward_hook(make_hook(blk_idx)))

    return hooks, per_block


def compute_entropy(per_block, log_T):
    """Return (mean_entropy, mean_concentration) from hook captures."""
    ents = []
    for lst in per_block:
        for attn in lst:   # [N_patches, heads, T, T]
            eps = 1e-8
            H = -(attn * torch.log(attn + eps)).sum(-1)   # [N_patches, heads, T]
            ents.append(H.mean().item())
    for lst in per_block:
        lst.clear()
    if not ents: return float("nan"), float("nan")
    H_mean = float(np.mean(ents))
    return H_mean, 1.0 - H_mean / log_T


def window_confidence(proc, model, all_frames, true_label, n_windows):
    """Run model on each sliding window. Returns P(true_label) per window."""
    windows = [all_frames[w*WIN_STRIDE : w*WIN_STRIDE + N_FRAMES_TSF]
               for w in range(n_windows)]
    inputs = proc(windows, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(dim=-1).cpu().numpy()
    return [float(probs[w, true_label]) for w in range(n_windows)]


def temporal_entropy_of_conf(conf, log_W):
    c = np.clip(np.array(conf, dtype=float), 1e-8, None)
    c /= c.sum()
    H = float(-(c * np.log(c)).sum())
    return H, 1.0 - H / log_W


def run_option_bc(ds, manifest, taxonomy, ckpt_path):
    """Run Option B (attention entropy) and C (sliding window) together."""
    if not ckpt_path.exists():
        print(f"  [B/C] checkpoint not found: {ckpt_path} — skipping")
        return pd.DataFrame(), pd.DataFrame()

    print(f"  [B/C] loading TimeSformer {ckpt_path.name} …")
    proc, model = load_timesformer(ckpt_path)
    T = model.config.num_frames if hasattr(model.config, "num_frames") else N_FRAMES_TSF
    log_T = float(np.log(T))
    n_windows = (N_DECODE_WIN - T) // WIN_STRIDE + 1
    log_W = float(np.log(n_windows))
    hooks, per_block = register_attn_hooks(model)

    classes = sorted(manifest["label_id"].unique())
    rows_b, rows_c = [], []

    for cls_idx, label_id in enumerate(classes):
        paths = [DATA_ROOT / p for p in manifest[manifest["label_id"]==label_id]["video_path"].tolist()]
        paths = [p for p in paths if p.exists()][:MAX_VID_GPU]
        if not paths: continue

        ents_b, concs_b = [], []
        ents_c, concs_c, peaks_c = [], [], []

        for p in paths:
            # decode dense (for sliding window)
            frames_dense = decode_uniform(p, N_DECODE_WIN)
            if frames_dense is None: continue
            frames_list = [frames_dense[i] for i in range(len(frames_dense))]

            # --- Option B: attention entropy (use N_FRAMES_TSF uniform frames) ---
            frames_tsf = [frames_dense[i] for i in
                          np.linspace(0, N_DECODE_WIN-1, T).astype(int)]
            for lst in per_block: lst.clear()
            inp = proc([frames_tsf], return_tensors="pt")
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            with torch.no_grad():
                model(**inp)
            H_b, conc_b = compute_entropy(per_block, log_T)
            if not np.isnan(H_b):
                ents_b.append(H_b)
                concs_b.append(conc_b)

            # --- Option C: sliding window confidence ---
            conf = window_confidence(proc, model, frames_list, label_id, n_windows)
            H_c, conc_c = temporal_entropy_of_conf(conf, log_W)
            ents_c.append(H_c)
            concs_c.append(conc_c)
            peaks_c.append(int(np.argmax(conf)))

        if ents_b:
            rows_b.append({"dataset": ds, "label_id": label_id,
                           "n_videos": len(ents_b),
                           "mean_entropy": float(np.mean(ents_b)),
                           "mean_concentration": float(np.mean(concs_b)),
                           "std_entropy": float(np.std(ents_b))})
        if ents_c:
            rows_c.append({"dataset": ds, "label_id": label_id,
                           "n_videos": len(ents_c),
                           "mean_entropy": float(np.mean(ents_c)),
                           "mean_concentration": float(np.mean(concs_c)),
                           "mean_peak_window": float(np.mean(peaks_c)),
                           "std_entropy": float(np.std(ents_c))})

        if (cls_idx+1) % 20 == 0 or cls_idx == len(classes)-1:
            print(f"    [{cls_idx+1}/{len(classes)}] B:{len(rows_b)} C:{len(rows_c)} classes")

    for h in hooks: h.remove()

    df_b = pd.DataFrame(rows_b)
    df_c = pd.DataFrame(rows_c)
    if not df_b.empty: df_b.to_csv(OUT / f"{ds}_attention_entropy_v2.csv", index=False)
    if not df_c.empty: df_c.to_csv(OUT / f"{ds}_sliding_window_confidence_v2.csv", index=False)
    print(f"    saved B:{len(df_b)} C:{len(df_c)} classes")
    return df_b, df_c


# ── partial correlation helper ─────────────────────────────────────────────────
def partial_r(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Partial correlation r(x,y | z) — residualise both on z."""
    bx = np.polyfit(z, x, 1); rx = x - np.polyval(bx, z)
    by = np.polyfit(z, y, 1); ry = y - np.polyval(by, z)
    r, p = stats.pearsonr(rx, ry)
    return r, p


# ── main loop ─────────────────────────────────────────────────────────────────
all_corr_rows = []

for ds, (manifest_fn, ckpt_name) in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"  Dataset: {ds}")
    print(f"{'='*60}")

    manifest_path = MANIFESTS / manifest_fn
    taxonomy_path = TAXONOMY  / f"{ds}_class_taxonomy.csv"

    if not manifest_path.exists():
        print(f"  manifest missing — skip"); continue
    if not taxonomy_path.exists():
        print(f"  taxonomy missing — skip"); continue

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["split"].isin(["val","validation","test"])].copy()
    if "exists" in manifest.columns:
        manifest = manifest[manifest["exists"]==True].copy()
    taxonomy = pd.read_csv(taxonomy_path)

    # map video paths: prepend DATA_ROOT
    manifest["abs_path"] = manifest["video_path"].apply(lambda p: DATA_ROOT / p)
    n_exist = manifest["abs_path"].apply(lambda p: p.exists()).sum()
    print(f"  {n_exist}/{len(manifest)} videos accessible")
    if n_exist < 5:
        print("  too few videos — skip"); continue

    ckpt_path = CKPT_ROOT / ckpt_name

    # --- Option A ---
    df_a = run_option_a(ds, manifest, taxonomy)

    # --- Options B & C ---
    df_b, df_c = run_option_bc(ds, manifest, taxonomy, ckpt_path)

    # --- correlate ---
    y_col = "mean_abs_drop"
    if y_col not in taxonomy.columns:
        print("  no aliasing column in taxonomy — skip corr"); continue

    row = {"dataset": ds}

    # A: partial r(temporal_std | mean_flow)
    if not df_a.empty:
        mA = df_a.merge(taxonomy[["label_id", y_col]], on="label_id")
        if len(mA) >= 10:
            r_a,  p_a  = stats.pearsonr(mA["temporal_std"].values,  mA[y_col].values)
            pr_a, pp_a = partial_r(mA["temporal_std"].values, mA[y_col].values, mA["mean_flow"].values)
            row.update({"n_A": len(mA), "r_A_marginal": round(r_a,3), "p_A_marginal": round(p_a,4),
                        "partial_r_A": round(pr_a,3), "partial_p_A": round(pp_a,4)})
            print(f"  A: n={len(mA)}  r={r_a:+.3f}  partial_r={pr_a:+.3f}  p={pp_a:.4f}")

    # B: r(concentration, aliasing)
    if not df_b.empty:
        mB = df_b.merge(taxonomy[["label_id", y_col]], on="label_id")
        if len(mB) >= 10:
            r_b, p_b = stats.pearsonr(mB["mean_concentration"].values, mB[y_col].values)
            row.update({"n_B": len(mB), "r_B": round(r_b,3), "p_B": round(p_b,4)})
            print(f"  B: n={len(mB)}  r(concentration,aliasing)={r_b:+.3f}  p={p_b:.4f}")

    # C: partial r(confidence_concentration | mean_flow)
    if not df_c.empty and not df_a.empty:
        mC = df_c.merge(taxonomy[["label_id", y_col]], on="label_id")
        mC = mC.merge(df_a[["label_id","mean_flow"]], on="label_id", how="left")
        if len(mC) >= 10:
            r_c,  p_c  = stats.pearsonr(mC["mean_concentration"].values, mC[y_col].values)
            pr_c, pp_c = partial_r(mC["mean_concentration"].values, mC[y_col].values,
                                   mC["mean_flow"].fillna(mC["mean_flow"].mean()).values)
            row.update({"n_C": len(mC), "r_C_marginal": round(r_c,3), "p_C_marginal": round(p_c,4),
                        "partial_r_C": round(pr_c,3), "partial_p_C": round(pp_c,4)})
            print(f"  C: n={len(mC)}  r={r_c:+.3f}  partial_r={pr_c:+.3f}  p={pp_c:.4f}  ← KEY")

    all_corr_rows.append(row)

# ── master table ──────────────────────────────────────────────────────────────
master = pd.DataFrame(all_corr_rows)
master_path = OUT / "nyquist_robust_all_datasets.csv"
master.to_csv(master_path, index=False)

print(f"\n{'='*70}")
print("MASTER CORRELATION TABLE — all 7 datasets")
print(f"{'='*70}")
print(master.to_string(index=False))

# Pooled stats across datasets (weighted by n)
print(f"\n{'='*70}")
print("POOLED across all datasets (unweighted mean of per-dataset correlations):")
for col in ["partial_r_A", "r_B", "partial_r_C"]:
    vals = master[col].dropna()
    if len(vals) > 0:
        print(f"  {col}: mean={vals.mean():+.3f}  (n={len(vals)} datasets)")
print(f"\nSaved master table: {master_path}")
