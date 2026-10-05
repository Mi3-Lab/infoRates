"""Repeated-measures ANOVA and held-out cascade for all 8 architectures.

Runs the two rebuttal-round analyses unchanged, with VideoMamba added. Its
per-clip predictions were deleted from the sweep directory (commit d021baee,
2026-06-13) and are recovered from git history into
evaluations/accv2026/coverage_stride_sweep_perclip/ (AUTSL from the 224px
re-run, which reproduces the published 65.6% / 16.0pp). Outputs go to
evaluations/fg2027/ so the rebuttal CSVs stay as they were.

Usage:  .venv/bin/python scripts/fg2027/perclip_analyses.py
"""
import importlib.util
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "evaluations/accv2026/coverage_stride_sweep"
PERCLIP = ROOT / "evaluations/accv2026/coverage_stride_sweep_perclip"
OUT = ROOT / "evaluations/fg2027"
BASE_MODELS = ["r3d_18", "mc3_18", "r2plus1d_18", "slowfast_r50",
               "timesformer", "vivit", "videomae"]


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/accv2026" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        view = Path(tmp)
        # One directory that looks like the sweep folder: the original entries
        # for the seven models, the recovered per-clip folders for VideoMamba.
        for d in SWEEP.iterdir():
            if d.is_dir() and not d.name.startswith("videomamba_"):
                (view / d.name).symlink_to(d)
        for d in PERCLIP.glob("videomamba_*"):
            (view / d.name).symlink_to(d)

        for name in ("rebuttal_repeated_measures_anova", "rebuttal_routing_heldout"):
            mod = load(name)
            mod.SWEEP = view
            mod.OUT = OUT
            # VideoMamba last, so the random splits of the other pairs are unchanged.
            mod.MODELS = BASE_MODELS + ["videomamba"]
            print("\n" + "#" * 78 + f"\n# {name}\n" + "#" * 78)
            mod.main()


if __name__ == "__main__":
    main()
