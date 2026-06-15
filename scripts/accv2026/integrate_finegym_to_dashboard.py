#!/usr/bin/env python3
"""Integrate FineGym data from coverage_stride_resolution_sweep into dashboard data files.

This merges local FineGym evaluation data with existing dashboard CSVs.
"""
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]

# ── Source: local coverage×stride×resolution sweep data ──────────────────────
sweep_root = ROOT / "evaluations/accv2026/coverage_stride_resolution_sweep"
if not sweep_root.exists():
    print(f"[ERROR] Sweep root not found: {sweep_root}")
    sys.exit(1)

# ── Destination: dashboard data ──────────────────────────────────────────────
dashboard_data = ROOT / "dashboard/data"
dashboard_data.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("INTEGRATING FINEGYM DATA INTO DASHBOARD")
print("=" * 80)

# ── 1. SWEEP_SUMMARY.CSV: Merge coverage×stride temporal sweeps ─────────────
print("\n[1/4] Integrating sweep_summary.csv (temporal sweep data)...")

# Load existing sweep data
sweep_csv = dashboard_data / "sweep_summary.csv"
if sweep_csv.exists():
    df_existing = pd.read_csv(sweep_csv)
    print(f"  Existing: {len(df_existing)} rows ({df_existing['dataset'].nunique()} datasets)")
else:
    df_existing = pd.DataFrame()
    print("  No existing sweep_summary.csv — creating new")

# Collect FineGym sweep data from local evaluation folders
finegym_rows = []
for model_ds_dir in sorted(sweep_root.glob("*_finegym")):
    for res_dir in sorted(model_ds_dir.glob("res*px")):
        summary_csv = res_dir / "sweep_summary.csv"
        if not summary_csv.exists():
            continue
        try:
            df = pd.read_csv(summary_csv)
            finegym_rows.append(df)
            model_ds = model_ds_dir.name
            res = res_dir.name
            print(f"  ✓ {model_ds}/{res}: {len(df)} configs")
        except Exception as e:
            print(f"  ✗ {summary_csv}: {e}")

if finegym_rows:
    df_finegym = pd.concat(finegym_rows, ignore_index=True)
    df_finegym["dataset"] = "finegym"
    df_finegym = df_finegym.rename(columns={"top1": "top1"})  # Ensure column name

    # Merge with existing (deduplicate)
    if not df_existing.empty:
        df_existing = df_existing[df_existing["dataset"] != "finegym"]  # Remove old FineGym if present

    df_merged = pd.concat([df_existing, df_finegym], ignore_index=True)
    df_merged.to_csv(sweep_csv, index=False)
    print(f"\n  ✓ Saved: {sweep_csv} ({len(df_merged)} rows, {df_merged['dataset'].nunique()} datasets)")
else:
    print("  ⚠️  No FineGym sweep data found in coverage_stride_resolution_sweep/")

# ── 2. P3_RESULTS.CSV: Merge spatial resolution evaluation (stride=1, cov=100%) ──
print("\n[2/4] Integrating p3_results.csv (spatial resolution eval)...")

p3_root = ROOT / "evaluations/accv2026/p3_retrained"
p3_csv = dashboard_data / "p3_results.csv"

if p3_csv.exists():
    df_p3_existing = pd.read_csv(p3_csv)
    print(f"  Existing: {len(df_p3_existing)} rows")
else:
    df_p3_existing = pd.DataFrame()

# Collect P3 data: res*_*_summary.csv files from p3_retrained/{model}_finegym/
p3_rows = []
for model_ds_dir in sorted(p3_root.glob("*_finegym")):
    for summary_csv in sorted(model_ds_dir.glob("res*_*_summary.csv")):
        try:
            df = pd.read_csv(summary_csv)
            if df.empty:
                continue
            # Extract resolution from filename: res112_accv2026_..._summary.csv
            parts = summary_csv.name.split("_")
            res_str = parts[0].replace("res", "")
            if res_str.isdigit():
                res = int(res_str)
            else:
                continue
            model_ds = model_ds_dir.name
            model = model_ds.replace("_finegym", "")
            df["model"] = model
            df["dataset"] = "finegym"
            df["res"] = res
            df["acc"] = df["top1"] * 100
            p3_rows.append(df[["model", "dataset", "res", "acc"]])
            print(f"  ✓ {model_ds}/{summary_csv.name}: res={res}px, {len(df)} entries")
        except Exception as e:
            print(f"  ✗ {summary_csv}: {e}")

if p3_rows:
    df_p3_finegym = pd.concat(p3_rows, ignore_index=True)
    # Merge with existing
    if not df_p3_existing.empty:
        df_p3_existing = df_p3_existing[df_p3_existing["dataset"] != "finegym"]
    df_p3_merged = pd.concat([df_p3_existing, df_p3_finegym], ignore_index=True)
    df_p3_merged.to_csv(p3_csv, index=False)
    print(f"\n  ✓ Saved: {p3_csv} ({len(df_p3_merged)} rows)")
else:
    print("  ⚠️  No P3 data found in p3_retrained/")

# ── 3. RETRAINED_SPATIAL.CSV: P3 retrained checkpoints ──────────────────────
print("\n[3/4] Integrating retrained_spatial.csv (P3 retrained models)...")

retrained_csv = dashboard_data / "retrained_spatial.csv"
if retrained_csv.exists():
    df_retrained_existing = pd.read_csv(retrained_csv)
    print(f"  Existing: {len(df_retrained_existing)} rows")
else:
    df_retrained_existing = pd.DataFrame()

# Parse P3 checkpoint data for FineGym
retrained_rows = []
for model_ds_dir in sorted(p3_root.glob("*_finegym")):
    for summary_csv in sorted(model_ds_dir.glob("res*_*_summary.csv")):
        try:
            df = pd.read_csv(summary_csv)
            if df.empty:
                continue
            parts = summary_csv.name.split("_")
            res_str = parts[0].replace("res", "")
            if not res_str.isdigit():
                continue
            train_res = int(res_str)
            model_ds = model_ds_dir.name
            model = model_ds.replace("_finegym", "")

            # Best accuracy from this checkpoint
            best_acc = df["top1"].max() * 100
            retrained_rows.append({
                "model": model,
                "dataset": "finegym",
                "train_res": train_res,
                "acc": best_acc
            })
            print(f"  ✓ {model}/{train_res}px: {best_acc:.1f}%")
        except Exception as e:
            print(f"  ✗ {summary_csv}: {e}")

if retrained_rows:
    df_retrained_finegym = pd.DataFrame(retrained_rows)
    if not df_retrained_existing.empty:
        df_retrained_existing = df_retrained_existing[df_retrained_existing["dataset"] != "finegym"]
    df_retrained_merged = pd.concat([df_retrained_existing, df_retrained_finegym], ignore_index=True)
    df_retrained_merged.to_csv(retrained_csv, index=False)
    print(f"\n  ✓ Saved: {retrained_csv} ({len(df_retrained_merged)} rows)")
else:
    print("  ⚠️  No retrained checkpoint data found")

# ── 4. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("INTEGRATION COMPLETE")
print("=" * 80)
print("\nFineGym is now available in the dashboard. Reload the Streamlit app to see it.")
print(f"\nDatasets now available:")
datasets = set()
for csv_path in [sweep_csv, p3_csv, retrained_csv]:
    if csv_path.exists():
        try:
            df_check = pd.read_csv(csv_path)
            datasets.update(df_check["dataset"].dropna().unique())
        except:
            pass

for ds in sorted(datasets):
    print(f"  • {ds}")
