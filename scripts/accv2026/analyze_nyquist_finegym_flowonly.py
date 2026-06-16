#!/usr/bin/env python3
"""FineGym flow-only spectral measurement (run on the machine that has FineGym videos).

This is the missing piece for scripts/accv2026/analyze_nyquist_spectral_v2.py:
that analysis already covers 7 datasets x 5 resolutions, but FineGym's source
videos were not accessible on this machine. The accuracy side for FineGym
(empirical stride-sensitivity per resolution) is ALREADY available from
evaluations/accv2026/coverage_stride_resolution_sweep/{model}_finegym/
sweep_summary_all_resolutions.csv and does not need to be recomputed --
this script only needs to produce the flow-derived spectral cutoff frequency.

Output: evaluations/accv2026/e3_spectral/finegym_cutoff_freq.csv
        columns: dataset, resolution, n_videos, cutoff_freq

After running, copy that CSV back to the main machine and merge it into
evaluations/accv2026/e3_spectral/nyquist_resolution_validation.csv (see
instructions printed at the end of this script).
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy.signal import periodogram

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "evaluations/accv2026/manifests/finegym_val_20_per_class.csv"
OUT_DIR = ROOT / "evaluations/accv2026/e3_spectral"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTIONS = [48, 96, 112, 160, 224]
N_FRAMES_CONSEC = 64
MAX_CLASSES = 12
N_VIDEOS_PER_CLASS = 2


def flow_series(video_path: str, frame_size: tuple, n_frames: int = N_FRAMES_CONSEC):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 10:
        cap.release()
        return None
    n = min(n_frames, total)
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < 10:
        return None
    mags = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()))
    return np.array(mags)


def spectral_cutoff(signal: np.ndarray, energy_frac: float = 0.90) -> float:
    sig = signal - signal.mean()
    if np.allclose(sig, 0):
        return 0.0
    freqs, psd = periodogram(sig, fs=1.0)
    if psd.sum() <= 0:
        return 0.0
    cum = np.cumsum(psd) / psd.sum()
    idx = min(int(np.searchsorted(cum, energy_frac)), len(freqs) - 1)
    return float(freqs[idx])


def main():
    rng = np.random.default_rng(0)
    df_m = pd.read_csv(MANIFEST)
    if "exists" in df_m.columns:
        df_m = df_m[df_m["exists"].astype(str).str.lower().isin(["true", "1"])]
    df_m["abs_video_path"] = df_m["video_path"].apply(lambda p: ROOT / p)
    df_m = df_m[df_m["abs_video_path"].apply(lambda p: p.exists())]

    classes = [c for c in sorted(df_m.label_id.unique())
               if len(df_m[df_m.label_id == c]) >= N_VIDEOS_PER_CLASS]
    if len(classes) > MAX_CLASSES:
        classes = list(rng.choice(classes, size=MAX_CLASSES, replace=False))

    sample_rows = []
    for c in classes:
        sub = df_m[df_m.label_id == c]
        sample_rows.append(sub.sample(n=min(N_VIDEOS_PER_CLASS, len(sub)), random_state=0))
    sample_df = pd.concat(sample_rows)
    video_paths = sample_df["abs_video_path"].astype(str).tolist()
    print(f"Sampled {len(video_paths)} FineGym videos across {len(classes)} classes")

    rows = []
    for res in RESOLUTIONS:
        cutoffs = []
        for vp in video_paths:
            sig = flow_series(vp, frame_size=(res, res))
            if sig is None or len(sig) < 10:
                continue
            fc = spectral_cutoff(sig)
            if fc > 0:
                cutoffs.append(fc)
        if not cutoffs:
            print(f"[finegym@{res}px] no usable videos -- check video_path resolution "
                  f"(are you running from the directory where manifest paths resolve?)")
            continue
        mean_fc = float(np.mean(cutoffs))
        rows.append({"dataset": "finegym", "resolution": res,
                      "n_videos": len(cutoffs), "cutoff_freq": round(mean_fc, 4)})
        print(f"[finegym@{res}px] n={len(cutoffs)}  f_c={mean_fc:.4f} cyc/frame")

    result = pd.DataFrame(rows)
    out_csv = OUT_DIR / "finegym_cutoff_freq.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print("\n=== NEXT STEPS ===")
    print(f"1. Copy this file back to the main machine: {out_csv}")
    print("2. Place it at the same relative path: evaluations/accv2026/e3_spectral/finegym_cutoff_freq.csv")
    print("3. Tell the assistant on the main machine it's ready to merge into the")
    print("   Nyquist v2 analysis (nyquist_resolution_validation.csv) and update the paper.")


if __name__ == "__main__":
    main()
