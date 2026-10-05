"""Rewrite the data rows of two supplementary tables from their source CSVs.

  tab:anova_full    between-cell ANOVA eta^2 per model and dataset
  tab:routing_full  in-sample confidence cascade, best accuracy at <= 8 frames

Usage:  .venv/bin/python scripts/fg2027/supp_tables.py   (edits supplementary.tex in place)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper/fg2027/supplementary.tex"


def replace_rows(tex: str, label: str, rows: str) -> str:
    a = tex.index("\\label{%s}" % label)
    i = tex.index("\\midrule\n", a) + len("\\midrule\n")
    j = tex.index("\\bottomrule", i)
    return tex[:i] + rows + tex[j:]


def anova_rows() -> str:
    an = pd.read_csv(ROOT / "evaluations/accv2026/e4_anova/anova_results.csv")
    names = [("VMamba", "videomamba"), ("TSF", "timesformer"), ("MC3", "mc3_18"),
             ("R3D", "r3d_18"), ("R2+1D", "r2plus1d_18"), ("VMAE", "videomae"),
             ("ViViT", "vivit"), ("SF", "slowfast_r50")]
    cols = ["autsl", "finegym", "ssv2", "diving48", "hmdb51", "driveact",
            "epic_kitchens", "ucf101"]
    out = []
    for short, m in names:
        cells = []
        for d in cols:
            r = an[(an.model == m) & (an.dataset == d)]
            if r.empty:
                cells += ["---", "---"]
            else:
                cells += [f"{r.eta2_coverage.iloc[0]:.2f}".lstrip("0"),
                          f"{r.eta2_stride.iloc[0]:.2f}".lstrip("0")]
        out.append(f"{short:<7s}& " + " & ".join(cells) + " \\\\")
    return "\n".join(out) + "\n"


def routing_rows() -> str:
    rs = pd.read_csv(ROOT / "dashboard/data/routing_summary.csv")
    names = [("R3D-18", "r3d_18"), ("MC3-18", "mc3_18"), ("R2+1D", "r2plus1d_18"),
             ("SlowFast", "slowfast_r50"), ("TimeSformer", "timesformer"),
             ("ViViT", "vivit"), ("VideoMAE", "videomae"), ("VideoMamba", "videomamba")]
    cols = ["autsl", "finegym", "diving48", "ssv2", "hmdb51", "driveact",
            "epic_kitchens", "ucf101"]
    acc = rs.assign(a=rs.best_accuracy * 100).pivot_table(index="model", columns="dataset", values="a")
    acc = acc.reindex(index=[m for _, m in names], columns=cols)
    acc["avg"] = acc.mean(axis=1)
    best = acc.max()
    out = []
    for short, m in names:
        cells = []
        for c in cols + ["avg"]:
            v = acc.loc[m, c]
            if pd.isna(v):
                cells.append("---")
            else:
                t = f"{v:.1f}"
                cells.append(f"\\textbf{{{t}}}" if abs(v - best[c]) < 1e-9 else t)
        out.append(f"{short:<12s} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out) + "\n"


def main() -> None:
    tex = TEX.read_text()
    tex = replace_rows(tex, "tab:anova_full", anova_rows())
    tex = replace_rows(tex, "tab:routing_full", routing_rows())
    TEX.write_text(tex)
    print(anova_rows())
    print(routing_rows())


if __name__ == "__main__":
    main()
