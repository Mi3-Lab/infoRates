#!/usr/bin/env python3
"""Compare VideoMamba3 fixed-budget results against existing VideoMamba baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluations/accv2026/videomamba3"


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    view = df.copy()
    for col in cols:
        if col not in view.columns:
            view[col] = ""
    rows = view[cols].fillna("").astype(str).values.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def load_summary(path: Path, model: str, checkpoint: str, resize: int | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["model"] = model
    df["checkpoint"] = checkpoint
    df["resize"] = resize
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--videomamba3-summary",
        default="evaluations/accv2026/fixed_budget/videomamba3_middle_complex_ucf101_112r_f2_d4_e3_fast_a100/"
        "ucf101_val_accv2026_videomamba3_middle_complex_ucf101_112r_f2_d4_e3_fast_h200_fixed_budget_summary.csv",
    )
    p.add_argument(
        "--videomamba-summary",
        default="evaluations/accv2026/fixed_budget/videomamba_ucf101_full_e10_h200/"
        "ucf101_val_accv2026_videomamba_ucf101_full_e10_h200_fixed_budget_summary.csv",
    )
    p.add_argument("--output-csv", default="evaluations/accv2026/videomamba3/fixed_budget_comparison.csv")
    p.add_argument("--output-md", default="evaluations/accv2026/videomamba3/fixed_budget_comparison.md")
    args = p.parse_args()

    rows = []
    vm3 = ROOT / args.videomamba3_summary
    if vm3.exists():
        rows.append(load_summary(vm3, "VideoMamba3-middle-complex-d4", Path(vm3).parent.name, 112))
    baseline = ROOT / args.videomamba_summary
    if baseline.exists():
        rows.append(load_summary(baseline, "VideoMamba", Path(baseline).parent.name, 224))
    if not rows:
        raise FileNotFoundError("No summary CSVs found to compare")

    df = pd.concat(rows, ignore_index=True)
    df["top1_pct"] = (df["top1"] * 100).round(2)
    df["top5_pct"] = (df["top5"] * 100).round(2)
    df["mean_inference_ms"] = (df["mean_inference_time_s"] * 1000).round(3)
    df["mean_total_ms"] = (df["mean_total_time_s"] * 1000).round(3)
    df["mean_model_input_frames"] = df["mean_model_input_frames"].round(2)
    df["mean_processed_frames"] = df["mean_processed_frames"].round(2)

    out_csv = ROOT / args.output_csv
    out_md = ROOT / args.output_md
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    cols = [
        "model",
        "budget",
        "resize",
        "n",
        "top1_pct",
        "top5_pct",
        "mean_processed_frames",
        "mean_model_input_frames",
        "mean_inference_ms",
        "mean_total_ms",
    ]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# VideoMamba3 Fixed-Budget Comparison\n\n")
        f.write(markdown_table(df.sort_values(["budget", "model"]), cols))
        f.write("\n\n")
        f.write(
            "Note: the current VideoMamba3 checkpoint is a fast validation candidate "
            "trained at 112px and 2 input frames. The existing VideoMamba baseline was "
            "trained/evaluated at 224px and 8 model input frames, so this table is a "
            "sanity comparison, not yet a matched SOTA claim.\n"
        )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
