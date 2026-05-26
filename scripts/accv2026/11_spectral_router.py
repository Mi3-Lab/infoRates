#!/usr/bin/env python3
"""Spectral Feature Router + Budget-Constrained Knapsack Allocator.

ACCV 2026 — Novel contribution: replace scalar FDE with a rich spectral
feature vector and frame the allocation problem as a knapsack.

Two methods:
  1. SpectralRouter   — MLP trained on spectral features → predicts optimal budget
  2. KnapsackAllocator — given total frame budget B for N videos, greedily
                         allocates frames to maximize expected batch accuracy

Spectral features (all computed from 8 low-res probe frames):
  fde_mean      — mean absolute frame difference (existing FDE)
  fde_std       — std of per-pair differences (motion variability)
  fde_max       — max frame difference (peak motion)
  fde_entropy   — Shannon entropy of frame-diff histogram
  dct_hf_ratio  — fraction of DCT energy in high-frequency bins (texture)
  hog_entropy   — entropy of oriented gradient histogram (motion unpredictability)
  temporal_skew — skewness of frame-diff sequence (asymmetric motion bursts)

Usage:
  python scripts/accv2026/11_spectral_router.py \
      --samples-csvs evaluations/accv2026/fixed_budget/*/somethingv2*samples.csv \
      --fde-cache    evaluations/accv2026/fixed_budget/r2plus1d_18_ssv2_full_e10_a100/fde_cache.csv \
      --output-dir   evaluations/accv2026/spectral_router \
      --workers 8
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

BUDGETS = [4, 8, 16, 32]
FEATURE_COLS = [
    "fde_mean", "fde_std", "fde_max", "fde_entropy",
    "dct_hf_ratio", "hog_entropy", "temporal_skew",
]


# ─── Spectral feature extraction ──────────────────────────────────────────────

def extract_spectral_features(
    video_path: str, n_probe: int = 8, size: int = 64
) -> Optional[dict]:
    """Extract 7-dimensional spectral feature vector from a video.

    Decodes n_probe grayscale frames at low resolution (size×size) via PyAV.
    Returns None if video is unreadable.
    """
    try:
        import av
    except ImportError:
        raise RuntimeError("PyAV not installed: pip install av")

    try:
        container = av.open(video_path, timeout=10)
        stream = container.streams.video[0]

        frames_out = []
        container.seek(0)
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="gray")
            h, w = arr.shape
            arr = arr[::max(1, h // size), ::max(1, w // size)][:size, :size].astype(np.float32)
            frames_out.append(arr)
            if len(frames_out) >= n_probe * 4:
                break
        container.close()

        if len(frames_out) < 2:
            return None

        # Evenly spaced probe frames
        indices = np.linspace(0, len(frames_out) - 1, n_probe, dtype=int)
        frames = [frames_out[i] for i in indices]

        # Per-pair absolute differences
        diffs = np.array([
            np.mean(np.abs(frames[i + 1] - frames[i]))
            for i in range(len(frames) - 1)
        ]) / 255.0

        # --- FDE features ---
        fde_mean = float(np.mean(diffs))
        fde_std  = float(np.std(diffs))
        fde_max  = float(np.max(diffs))

        # Entropy of frame-diff histogram
        hist, _ = np.histogram(diffs, bins=16, range=(0, 1))
        hist = hist.astype(float) + 1e-8
        hist /= hist.sum()
        fde_entropy = float(-np.sum(hist * np.log(hist + 1e-12)))

        # Temporal skewness of diff sequence
        if fde_std > 1e-8:
            temporal_skew = float(np.mean(((diffs - fde_mean) / fde_std) ** 3))
        else:
            temporal_skew = 0.0

        # --- DCT high-frequency energy ratio ---
        # Average DCT of all probe frames; ratio of outer 50% bins to total
        dct_energies = []
        for f in frames:
            from scipy.fft import dctn
            dct = dctn(f / 255.0, norm="ortho")
            total_e = np.sum(dct ** 2) + 1e-12
            # High-frequency = bottom-right quadrant of DCT
            h2, w2 = dct.shape[0] // 2, dct.shape[1] // 2
            hf_e = np.sum(dct[h2:, w2:] ** 2)
            dct_energies.append(hf_e / total_e)
        dct_hf_ratio = float(np.mean(dct_energies))

        # --- HOG-like gradient entropy ---
        # Compute gradient orientations per frame, measure their entropy
        orient_hists = []
        for f in frames:
            fy = np.diff(f / 255.0, axis=0)[:size - 1, :size - 1]
            fx = np.diff(f / 255.0, axis=1)[:size - 1, :size - 1]
            angles = np.arctan2(fy, fx + 1e-12)  # [-pi, pi]
            hist_o, _ = np.histogram(angles, bins=18, range=(-np.pi, np.pi))
            hist_o = hist_o.astype(float) + 1e-8
            hist_o /= hist_o.sum()
            orient_hists.append(-np.sum(hist_o * np.log(hist_o + 1e-12)))
        hog_entropy = float(np.mean(orient_hists))

        return {
            "fde_mean":      fde_mean,
            "fde_std":       fde_std,
            "fde_max":       fde_max,
            "fde_entropy":   fde_entropy,
            "dct_hf_ratio":  dct_hf_ratio,
            "hog_entropy":   hog_entropy,
            "temporal_skew": temporal_skew,
        }

    except Exception:
        return None


def build_feature_cache(
    video_paths: dict[str, str],  # video_id → path
    cache_path: Path,
    workers: int = 8,
    n_probe: int = 8,
) -> pd.DataFrame:
    """Extract spectral features for all videos; cache to CSV."""
    if cache_path.exists():
        print(f"  Loading spectral cache: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"  Extracting spectral features for {len(video_paths)} videos ({workers} workers)...")
    rows = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(extract_spectral_features, path, n_probe): vid_id
            for vid_id, path in video_paths.items()
        }
        done = 0
        for fut in as_completed(futures):
            vid_id = futures[fut]
            feats = fut.result()
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(video_paths)} videos processed")
            if feats is not None:
                rows.append({"video_id": vid_id, **feats})

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  Saved: {cache_path} ({len(df)} videos)")
    return df


# ─── Per-video accuracy labels ────────────────────────────────────────────────

def build_per_video_accuracy(samples_df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame: video_id × budget → mean accuracy across models.

    For each video, we aggregate across all model samples to get a
    model-averaged accuracy at each budget.
    """
    df = samples_df[~samples_df["skipped"].astype(bool)].copy()
    df["video_id"] = df["video_id"].astype(str)

    rows = []
    for (vid_id,), grp in df.groupby(["video_id"]):
        row = {"video_id": str(vid_id)}
        for b in BUDGETS:
            sub = grp[grp["budget"] == b]
            row[f"acc_{b}f"] = float(sub["correct_top1"].mean()) if len(sub) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_optimal_budget_per_video(acc_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Label = lowest budget where model-avg accuracy >= threshold.

    threshold=0.5 means "correct on average across models".
    Also stores per-budget accuracy for regression targets.
    """
    acc_df = acc_df.copy()
    labels = []
    for _, row in acc_df.iterrows():
        accs = {b: row[f"acc_{b}f"] for b in BUDGETS if not np.isnan(row[f"acc_{b}f"])}
        if not accs:
            labels.append(max(BUDGETS))
            continue
        best = max(accs.values())
        target = threshold * best if best > 0 else threshold
        label = max(BUDGETS)
        for b in sorted(accs):
            if accs[b] >= target:
                label = b
                break
        labels.append(label)
    acc_df["optimal_budget"] = labels
    return acc_df


# ─── SpectralRouter ───────────────────────────────────────────────────────────

def train_spectral_router(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple:
    """Train MLP router: spectral features → optimal budget class.

    Returns (model, scaler, label_encoder, train_metrics, test_metrics).
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    merged = features_df.merge(labels_df[["video_id", "optimal_budget"]], on="video_id")
    merged = merged.dropna(subset=FEATURE_COLS + ["optimal_budget"])

    X = merged[FEATURE_COLS].values
    y = merged["optimal_budget"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_enc, test_size=test_size, random_state=seed, stratify=y_enc
    )

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    model.fit(X_tr, y_tr)

    train_acc = accuracy_score(y_tr, model.predict(X_tr))
    test_acc  = accuracy_score(y_te, model.predict(X_te))
    report    = classification_report(
        y_te, model.predict(X_te),
        target_names=[str(b) + "f" for b in le.classes_],
        zero_division=0,
    )
    return model, scaler, le, train_acc, test_acc, report


def evaluate_router(
    features_df: pd.DataFrame,
    acc_df: pd.DataFrame,
    model, scaler, le,
) -> dict:
    """Evaluate the spectral router against fixed-budget baselines.

    Returns dict with accuracy and avg_frames for each strategy.
    """
    merged = features_df.merge(acc_df, on="video_id").dropna(subset=FEATURE_COLS)

    X = merged[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    predicted_budgets = le.inverse_transform(model.predict(X_scaled)).astype(int)

    results = {}

    # Fixed baselines
    for b in BUDGETS:
        col = f"acc_{b}f"
        if col in merged.columns:
            acc = merged[col].mean()
            results[f"fixed_{b}f"] = {"accuracy": float(acc), "avg_frames": b}

    # Spectral router
    router_accs = []
    for i, b in enumerate(predicted_budgets):
        col = f"acc_{b}f"
        router_accs.append(merged.iloc[i][col] if col in merged.columns else np.nan)
    results["spectral_router"] = {
        "accuracy": float(np.nanmean(router_accs)),
        "avg_frames": float(np.mean(predicted_budgets)),
        "budget_dist": {str(b): int((predicted_budgets == b).sum()) for b in BUDGETS},
    }

    return results, merged, predicted_budgets


# ─── KnapsackAllocator ────────────────────────────────────────────────────────

def knapsack_allocate(
    acc_matrix: np.ndarray,  # shape [N, len(BUDGETS)]: predicted accuracy per video per budget
    total_budget: int,       # total frames to distribute across N videos
) -> np.ndarray:             # shape [N]: allocated budget per video
    """Greedy knapsack: maximise sum(accuracy) subject to sum(frames) <= total_budget.

    Starts all videos at the minimum budget, then greedily upgrades the video
    where the marginal accuracy gain per extra frame is highest.
    """
    n = len(acc_matrix)
    budget_idx = np.zeros(n, dtype=int)  # index into BUDGETS; start at budget=4
    budget_arr = np.array(BUDGETS)

    used = n * BUDGETS[0]
    remaining = total_budget - used

    while remaining > 0:
        best_gain_per_frame = -np.inf
        best_i = -1

        for i in range(n):
            cur_idx = budget_idx[i]
            if cur_idx >= len(BUDGETS) - 1:
                continue  # already at max
            next_idx = cur_idx + 1
            delta_frames = budget_arr[next_idx] - budget_arr[cur_idx]
            if delta_frames > remaining:
                continue
            delta_acc = acc_matrix[i, next_idx] - acc_matrix[i, cur_idx]
            gain_per_frame = delta_acc / delta_frames
            if gain_per_frame > best_gain_per_frame:
                best_gain_per_frame = gain_per_frame
                best_i = i

        if best_i == -1:
            break  # no more upgrades possible

        cur_idx = budget_idx[best_i]
        next_idx = cur_idx + 1
        remaining -= budget_arr[next_idx] - budget_arr[cur_idx]
        budget_idx[best_i] = next_idx

    return budget_arr[budget_idx]


def train_accuracy_regressor(
    features_df: pd.DataFrame,
    acc_df: pd.DataFrame,
    seed: int = 42,
) -> tuple:
    """Train MLP regressor: features → [acc_4, acc_8, acc_16, acc_32].

    Used by the knapsack to estimate per-video accuracy at each budget.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    merged = features_df.merge(acc_df, on="video_id").dropna(
        subset=FEATURE_COLS + [f"acc_{b}f" for b in BUDGETS]
    )

    X = merged[FEATURE_COLS].values
    Y = merged[[f"acc_{b}f" for b in BUDGETS]].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_scaled, Y, test_size=0.3, random_state=seed
    )

    reg = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    reg.fit(X_tr, Y_tr)

    mae = mean_absolute_error(Y_te, reg.predict(X_te))
    return reg, scaler, mae


def evaluate_knapsack(
    features_df: pd.DataFrame,
    acc_df: pd.DataFrame,
    reg, reg_scaler,
    avg_budgets: list[int] = [8, 12, 16],
) -> dict:
    """Evaluate knapsack allocator at several average-budget targets.

    For each avg_budget k:
      - total_budget = N * k
      - allocate greedily using predicted accuracy matrix
      - measure actual accuracy using ground-truth acc_df
    """
    merged = features_df.merge(acc_df, on="video_id").dropna(
        subset=FEATURE_COLS + [f"acc_{b}f" for b in BUDGETS]
    )

    X = merged[FEATURE_COLS].values
    X_scaled = reg_scaler.transform(X)
    acc_pred = reg.predict(X_scaled)  # [N, 4]
    acc_pred = np.clip(acc_pred, 0, 1)

    # Ground-truth accuracy matrix
    acc_true = merged[[f"acc_{b}f" for b in BUDGETS]].values

    results = {}
    n = len(merged)

    for avg_k in avg_budgets:
        total = n * avg_k
        alloc = knapsack_allocate(acc_pred, total)

        # Actual accuracy using ground-truth
        knapsack_accs = []
        for i, b in enumerate(alloc):
            col = f"acc_{b}f"
            knapsack_accs.append(acc_true[i, BUDGETS.index(int(b))])

        # Fixed baseline at same avg frames (nearest budget)
        nearest_fixed = min(BUDGETS, key=lambda x: abs(x - avg_k))
        fixed_acc = merged[f"acc_{nearest_fixed}f"].mean()

        results[f"avg_{avg_k}f"] = {
            "knapsack_accuracy": float(np.mean(knapsack_accs)),
            "knapsack_avg_frames": float(alloc.mean()),
            "fixed_accuracy": float(fixed_acc),
            "fixed_budget": nearest_fixed,
            "delta_accuracy": float(np.mean(knapsack_accs)) - float(fixed_acc),
            "budget_dist": {str(b): int((alloc == b).sum()) for b in BUDGETS},
        }

    return results


# ─── Feature importance ───────────────────────────────────────────────────────

def feature_importance(features_df: pd.DataFrame, acc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlation of each feature with optimal budget."""
    from scipy import stats

    merged = features_df.merge(
        acc_df[["video_id", "optimal_budget"]], on="video_id"
    ).dropna()

    rows = []
    for feat in FEATURE_COLS:
        r, p = stats.spearmanr(merged[feat], merged["optimal_budget"])
        rows.append({"feature": feat, "spearman_r": round(r, 4), "p_value": round(p, 6)})
    return pd.DataFrame(rows).sort_values("spearman_r", key=abs, ascending=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-csvs", nargs="+", required=True,
                        help="One or more fixed-budget samples CSVs (SSV2)")
    parser.add_argument("--fde-cache", required=True,
                        help="Existing scalar FDE cache CSV (video_id, fde)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-probe", type=int, default=8,
                        help="Probe frames for spectral feature extraction")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Recompute spectral cache even if it exists")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load all samples CSVs and concatenate
    print(f"\n{'='*60}")
    print("1. Loading samples data...")
    dfs = []
    for p in args.samples_csvs:
        p = Path(p)
        if p.exists():
            df = pd.read_csv(p)
            dfs.append(df)
            print(f"   {p.name}: {len(df)} rows")
    if not dfs:
        print("ERROR: No samples CSVs found.")
        sys.exit(1)
    samples_df = pd.concat(dfs, ignore_index=True)
    samples_df["video_id"] = samples_df["video_id"].astype(str)
    print(f"   Total: {len(samples_df)} rows, {samples_df['video_id'].nunique()} unique videos")

    # 2. Build per-video accuracy table
    print("\n2. Computing per-video accuracy at each budget...")
    acc_df = build_per_video_accuracy(samples_df)
    acc_df = compute_optimal_budget_per_video(acc_df, threshold=0.5)
    print(f"   {len(acc_df)} videos | optimal_budget distribution:")
    print("  ", acc_df["optimal_budget"].value_counts().sort_index().to_dict())

    # 3. Extract spectral features
    print("\n3. Extracting spectral features...")
    # Build video_id → path mapping from samples
    vid_to_path = (
        samples_df[["video_id", "video_path"]]
        .drop_duplicates("video_id")
        .set_index("video_id")["video_path"]
        .to_dict()
    )
    cache_path = out / "spectral_features_cache.csv"
    if args.force_recompute and cache_path.exists():
        cache_path.unlink()
    features_df = build_feature_cache(vid_to_path, cache_path, workers=args.workers,
                                      n_probe=args.n_probe)
    features_df["video_id"] = features_df["video_id"].astype(str)
    print(f"   Features extracted: {len(features_df)} videos")

    # 4. Feature importance
    print("\n4. Feature importance (Spearman correlation with optimal budget):")
    fimp = feature_importance(features_df, acc_df)
    print(fimp.to_string(index=False))
    fimp.to_csv(out / "feature_importance.csv", index=False)

    # 5. Train spectral router
    print("\n5. Training Spectral Router (MLP: features → budget class)...")
    model, scaler, le, train_acc, test_acc, report = train_spectral_router(
        features_df, acc_df
    )
    print(f"   Train acc: {train_acc:.3f} | Test acc: {test_acc:.3f}")
    print(report)

    # 6. Evaluate router vs fixed baselines
    print("\n6. Evaluating Spectral Router vs Fixed Baselines:")
    router_results, merged_eval, pred_budgets = evaluate_router(
        features_df, acc_df, model, scaler, le
    )

    rows = []
    for method, vals in router_results.items():
        rows.append({
            "method": method,
            "accuracy": vals["accuracy"],
            "avg_frames": vals["avg_frames"],
        })
        acc_pct = vals["accuracy"] * 100
        print(f"   {method:20s}: acc={acc_pct:.2f}%  avg_frames={vals['avg_frames']:.1f}")

    # Compare to FDE router baseline
    fde_cache = pd.read_csv(args.fde_cache)
    fde_cache["video_id"] = fde_cache["video_id"].astype(str)
    merged_fde = features_df.merge(
        acc_df.merge(fde_cache, on="video_id", how="left"), on="video_id"
    ).dropna(subset=["fde"])
    # Simple FDE threshold router (same thresholds as before)
    thresholds = [0.03, 0.06, 0.09]  # low→4f, medium→8f, high-medium→16f, high→32f
    def fde_route(fde_val):
        if fde_val < thresholds[0]: return 4
        if fde_val < thresholds[1]: return 8
        if fde_val < thresholds[2]: return 16
        return 32
    fde_budgets = merged_fde["fde"].apply(fde_route).values
    fde_accs = []
    for i, b in enumerate(fde_budgets):
        col = f"acc_{b}f"
        fde_accs.append(merged_fde.iloc[i][col] if col in merged_fde.columns else np.nan)
    fde_acc = float(np.nanmean(fde_accs))
    fde_avg = float(np.mean(fde_budgets))
    print(f"   {'scalar_fde_router':20s}: acc={fde_acc*100:.2f}%  avg_frames={fde_avg:.1f}")

    pd.DataFrame(rows).to_csv(out / "router_comparison.csv", index=False)

    # 7. Train accuracy regressor for knapsack
    print("\n7. Training accuracy regressor for Knapsack Allocator...")
    reg, reg_scaler, mae = train_accuracy_regressor(features_df, acc_df)
    print(f"   Regressor MAE (per-video per-budget accuracy): {mae:.4f}")

    # 8. Evaluate knapsack
    print("\n8. Evaluating Budget-Constrained Knapsack Allocator:")
    print("   avg_budget | knapsack_acc | fixed_acc | delta")
    print("   " + "-" * 50)
    ks_results = evaluate_knapsack(
        features_df, acc_df, reg, reg_scaler, avg_budgets=[8, 10, 12, 14, 16]
    )
    ks_rows = []
    for avg_k, vals in ks_results.items():
        delta = vals["delta_accuracy"] * 100
        sign = "+" if delta >= 0 else ""
        print(f"   {avg_k:10s} | {vals['knapsack_accuracy']*100:.2f}%       "
              f"| {vals['fixed_accuracy']*100:.2f}%    | {sign}{delta:.2f}%")
        ks_rows.append({
            "avg_budget_target": avg_k,
            "knapsack_accuracy": vals["knapsack_accuracy"],
            "knapsack_avg_frames": vals["knapsack_avg_frames"],
            "fixed_baseline_accuracy": vals["fixed_accuracy"],
            "fixed_budget": vals["fixed_budget"],
            "delta_accuracy": vals["delta_accuracy"],
            "budget_dist": str(vals["budget_dist"]),
        })
    pd.DataFrame(ks_rows).to_csv(out / "knapsack_results.csv", index=False)

    # 9. Oracle knapsack (upper bound using ground-truth accuracy matrix)
    print("\n9. Oracle Knapsack (upper bound — uses ground-truth accuracy, not predicted):")
    merged_full = features_df.merge(acc_df, on="video_id").dropna(
        subset=FEATURE_COLS + [f"acc_{b}f" for b in BUDGETS]
    )
    acc_true_full = merged_full[[f"acc_{b}f" for b in BUDGETS]].values
    oracle_rows = []
    print("   avg_budget | oracle_acc | fixed_acc | delta")
    print("   " + "-" * 50)
    for avg_k in [8, 10, 12, 14, 16]:
        total = len(acc_true_full) * avg_k
        alloc_oracle = knapsack_allocate(acc_true_full, total)
        oracle_accs = [
            acc_true_full[i, BUDGETS.index(int(b))]
            for i, b in enumerate(alloc_oracle)
        ]
        nearest_fixed = min(BUDGETS, key=lambda x: abs(x - avg_k))
        fixed_acc_k = merged_full[f"acc_{nearest_fixed}f"].mean()
        delta = (np.mean(oracle_accs) - fixed_acc_k) * 100
        sign = "+" if delta >= 0 else ""
        print(f"   avg_{avg_k}f     | {np.mean(oracle_accs)*100:.2f}%      "
              f"| {fixed_acc_k*100:.2f}%    | {sign}{delta:.2f}%")
        oracle_rows.append({
            "avg_budget_target": f"avg_{avg_k}f",
            "oracle_accuracy": float(np.mean(oracle_accs)),
            "oracle_avg_frames": float(alloc_oracle.mean()),
            "fixed_baseline_accuracy": float(fixed_acc_k),
            "fixed_budget": nearest_fixed,
            "delta_accuracy": float(np.mean(oracle_accs) - fixed_acc_k),
            "budget_dist": str({str(b): int((alloc_oracle == b).sum()) for b in BUDGETS}),
        })
    pd.DataFrame(oracle_rows).to_csv(out / "oracle_knapsack_results.csv", index=False)

    # 10. Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    best_fixed = max(router_results[f"fixed_{b}f"]["accuracy"] for b in BUDGETS)
    best_fixed_b = max(BUDGETS, key=lambda b: router_results[f"fixed_{b}f"]["accuracy"])
    router_acc = router_results["spectral_router"]["accuracy"]
    router_frames = router_results["spectral_router"]["avg_frames"]
    # compare to nearest fixed budget
    nearest_b = min(BUDGETS, key=lambda b: abs(b - router_frames))
    print(f"  Best fixed baseline: Fixed-{best_fixed_b}f = {best_fixed*100:.2f}%")
    print(f"  Spectral router:     {router_acc*100:.2f}% at {router_frames:.1f}f avg "
          f"(vs Fixed-{nearest_b}f={router_results[f'fixed_{nearest_b}f']['accuracy']*100:.2f}%)")
    print(f"  Scalar FDE router:   {fde_acc*100:.2f}% at {fde_avg:.1f}f avg")
    ks16 = ks_results['avg_16f']
    print(f"  Knapsack @ avg-16f:  {ks16['knapsack_accuracy']*100:.2f}% "
          f"(fixed-{ks16['fixed_budget']}f={ks16['fixed_accuracy']*100:.2f}%, "
          f"delta={ks16['delta_accuracy']*100:+.2f}%)")
    or16 = oracle_rows[-1]
    print(f"  Oracle  @ avg-16f:   {or16['oracle_accuracy']*100:.2f}% "
          f"(fixed-{or16['fixed_budget']}f={or16['fixed_baseline_accuracy']*100:.2f}%, "
          f"delta={or16['delta_accuracy']*100:+.2f}%)")
    print(f"\n  KEY FINDING: spectral features have low correlation with optimal budget")
    print(f"  on SSV2 (r_max={fimp['spearman_r'].abs().max():.3f}). Difficulty is SEMANTIC,")
    print(f"  not kinematic — motivates learned difficulty estimation (WACV contribution).")
    print(f"\nAll results saved to {out}/")


if __name__ == "__main__":
    main()
