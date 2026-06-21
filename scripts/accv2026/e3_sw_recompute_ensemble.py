"""Recomputa o ensemble de sliding window com todos os 8 modelos disponíveis.

Lê sw_{model}_{dataset}.csv para cada modelo que tiver arquivo,
recalcula sw_ensemble_{dataset}.csv e a tabela master sw_all_models_master.csv.
Roda em .venv normal (sem dependência de mamba).
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT     = Path(__file__).resolve().parents[2]
TAXONOMY = ROOT / "evaluations/accv2026/e5_taxonomy"
OUT      = ROOT / "evaluations/accv2026/e3_spectral"

ALL_MODELS = [
    "timesformer", "videomae", "vivit",
    "r3d_18", "mc3_18", "r2plus1d_18",
    "slowfast_r50", "videomamba",
]

DATASETS = [
    "ucf101", "ssv2", "hmdb51", "diving48",
    "autsl", "driveact", "epic_kitchens", "finegym",
]

y_col = "mean_abs_drop"

master_rows = []

for ds in DATASETS:
    tax_f = TAXONOMY / f"{ds}_class_taxonomy.csv"
    if not tax_f.exists():
        print(f"[{ds}] taxonomy ausente — skip"); continue
    tax = pd.read_csv(tax_f)
    if y_col not in tax.columns:
        print(f"[{ds}] sem {y_col} na taxonomy — skip"); continue

    # junta todos os modelos disponíveis
    model_dfs = {}
    for m in ALL_MODELS:
        f = OUT / f"sw_{m}_{ds}.csv"
        if f.exists():
            model_dfs[m] = pd.read_csv(f).set_index("label_id")["mean_concentration"]

    if not model_dfs:
        print(f"[{ds}] nenhum modelo com CSV — skip"); continue

    print(f"\n[{ds}] modelos disponíveis ({len(model_dfs)}): {list(model_dfs.keys())}")

    # ensemble por classe
    all_labels = sorted(set.union(*[set(s.index) for s in model_dfs.values()]))
    ens_rows = []
    for label_id in all_labels:
        vals = [float(model_dfs[m][label_id])
                for m in model_dfs if label_id in model_dfs[m].index]
        if vals:
            ens_rows.append({
                "label_id": label_id,
                "n_models": len(vals),
                "ensemble_concentration": float(np.mean(vals)),
                "std_across_models": float(np.std(vals)),
            })

    df_ens = pd.DataFrame(ens_rows)
    df_ens.to_csv(OUT / f"sw_ensemble_{ds}.csv", index=False)

    # correlação
    merged = df_ens.merge(tax[["label_id", y_col]], on="label_id")
    if len(merged) < 10:
        print(f"  classes insuficientes ({len(merged)}) — skip corr"); continue

    x = merged["ensemble_concentration"].values
    y = merged[y_col].values
    r_sp, p_sp = stats.spearmanr(x, y)
    r_pe, p_pe = stats.pearsonr(x, y)
    n_m = df_ens["n_models"].mean()
    print(f"  n_classes={len(merged)}  n_models_avg={n_m:.1f}")
    print(f"  Spearman ρ = {r_sp:+.3f}  p={p_sp:.4f}")
    print(f"  Pearson  r = {r_pe:+.3f}  p={p_pe:.4f}")

    master_rows.append({
        "dataset":      ds,
        "n_classes":    len(merged),
        "n_models_avg": round(n_m, 1),
        "spearman_rho": round(r_sp, 3),
        "spearman_p":   round(p_sp, 4),
        "pearson_r":    round(r_pe, 3),
        "pearson_p":    round(p_pe, 4),
    })

master = pd.DataFrame(master_rows)
master_path = OUT / "sw_all_models_master.csv"
master.to_csv(master_path, index=False)

print(f"\n{'='*70}")
print("SLIDING WINDOW ENSEMBLE — 8 MODELOS — TABELA FINAL")
print(f"{'='*70}")
print(master.to_string(index=False))

if not master.empty:
    print(f"\nPooled (média não-ponderada):")
    print(f"  Spearman ρ = {master['spearman_rho'].mean():+.3f}  (n={len(master)} datasets)")
    print(f"  Pearson  r = {master['pearson_r'].mean():+.3f}")

print(f"\nSalvo: {master_path}")
