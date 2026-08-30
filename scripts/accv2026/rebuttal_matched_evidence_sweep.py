#!/usr/bin/env python3
"""Rebuttal sweeps that decouple what the cov x stride grid confounds.

The submitted protocol picks candidates = arange(0, window, stride) and then
re-uniformizes them with linspace, so at fixed coverage the selected frames span
the same window at every stride. Stride only bites once ceil(window/s) < budget,
and select_frame_indices fills the deficit by repeating the LAST candidate. Three
modes isolate the factors that behaviour ties together:

  matched   Every model sees the SAME k distinct frames, at the same positions,
            uniformly spanning the clip, resampled to its own input length. Since
            frame budget correlates with the reported robustness ranking at
            rho=0.85, this is the test of whether architecture matters once the
            evidence is equalized.       (reviewer pxtb: confounded comparison)

  span      k = the model's native budget, drawn from a CENTERED window of width
            W. Frame count is fixed, so W alone sets the true sampling rate k/W.
            This is the temporal-density axis the submitted sweep never varied.
                                         (reviewer vdwp: "how early", not aliasing)

  covstride The submitted 5x5 grid, but adapting to the model's input length by
            uniform resampling instead of repeat-last. The gap against the
            published numbers is the share of the "aliasing cliff" that was the
            frozen-tail artifact rather than lost evidence.
                                         (reviewer pxtb: input-length shift)

Checkpoints, manifests and decoding are imported from sweep_coverage_stride.py,
so results are directly comparable to the published sweep.

Usage:
    python scripts/accv2026/rebuttal_matched_evidence_sweep.py \
        --model timesformer --dataset ssv2 --mode matched
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu as decord_cpu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from info_rates.evaluation.benchmark import _move_batch_to_device  # noqa: E402
from sweep_coverage_stride import (  # noqa: E402
    MODEL_CFG, DATASET_CFG, load_model,
)

# Kinetics-400 has no fine-tuned checkpoint: every architecture in the pool was
# already pretrained on it, so the K400-pretrained backbone IS the model under
# test. Label order was verified identical to the manifest for the six models
# that expose an explicit class list; SlowFast and VideoMamba do not, and are
# checked instead by whether their top-1 lands near the published ~76%.
K400_PRETRAINED = {
    "videomamba": ROOT / "fine_tuned_models/videomamba_pretrained"
                       / "videomamba_m16_k400_f8_res224.pth",
}

# k <= 8 is the strictly matched regime: every model's budget is >= 8, so all k
# distinct frames survive resampling. Larger k is recorded but only the models
# with budget >= k can actually use it.
K_VALUES = [1, 2, 4, 8, 16]
SPAN_PCTS = [10, 25, 50, 75, 100]
COVERAGES = [10, 25, 50, 75, 100]
STRIDES = [1, 2, 4, 8, 16]
# antialias mode: same k, same span, point-sampled vs temporally box-filtered.
ANTIALIAS_KS = [1, 2, 4, 8, 16]

OUT_BASE = ROOT / "evaluations/accv2026/rebuttal_sweeps"


def load_pretrained_k400(model_name: str):
    """Load the K400-pretrained backbone with its original 400-way head."""
    import torch as _torch
    device = "cuda" if _torch.cuda.is_available() else "cpu"

    # Every factory below replaces the classification head with a freshly
    # initialised Linear *after* loading pretrained weights, even when
    # num_labels is already 400. For K400 the pretrained head is exactly what we
    # want to keep, so each branch restores it.
    if model_name in ("r3d_18", "mc3_18", "r2plus1d_18"):
        from info_rates.models.torchvision_video import (
            MODEL_REGISTRY, TorchvisionVideoClassifier, TorchvisionVideoProcessor)
        info = MODEL_REGISTRY[model_name]
        net = info["builder"](weights=info["weights"])  # keeps the pretrained fc
        model = TorchvisionVideoClassifier(net, num_labels=400).to(device)
        processor = TorchvisionVideoProcessor(size=MODEL_CFG[model_name]["resize"])
    elif model_name == "slowfast_r50":
        import torch.hub as hub
        from info_rates.models.slowfast_video import (
            create_slowfast_model, SlowFastVideoProcessor)
        model = create_slowfast_model(num_labels=400, pretrained=True)
        state = hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/"
            "SLOWFAST_8x8_R50.pyth", map_location="cpu", check_hash=False)
        inner = getattr(model, "model", model)
        inner.load_state_dict(state.get("model_state", state), strict=False)
        model = model.to(device)
        processor = SlowFastVideoProcessor(size=MODEL_CFG[model_name]["resize"])
    elif model_name == "videomamba":
        from info_rates.models.videomamba_model import (
            build_videomamba, VideoMambaProcessor, _extract_state)
        model = build_videomamba(
            num_classes=400, num_frames=MODEL_CFG[model_name]["frames"],
            pretrained_path=str(K400_PRETRAINED["videomamba"]), img_size=224)
        state = _extract_state(_torch.load(
            str(K400_PRETRAINED["videomamba"]), map_location="cpu"))
        model.backbone.load_state_dict(state, strict=False)
        model = model.to(device)
        processor = VideoMambaProcessor(size=MODEL_CFG[model_name]["resize"])
    else:
        from transformers import AutoImageProcessor, AutoModelForVideoClassification
        from info_rates.models.model_factory import ModelFactory
        model_id = ModelFactory.get_model_info(model_name)["model_id"]
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForVideoClassification.from_pretrained(model_id).to(device)

    model.eval()
    return model, processor, device


def resample_to_length(items: list, target: int) -> list:
    """Uniformly resample a frame list to `target` entries, never repeat-last.

    Downsampling and upsampling both go through linspace, so an upsampled clip
    repeats interior frames evenly instead of freezing on the final one.
    """
    if target <= 0 or len(items) == target:
        return items
    idx = np.linspace(0, len(items) - 1, target).round().astype(np.int64)
    return [items[int(i)] for i in idx]


def matched_indices(total: int, k: int) -> np.ndarray:
    """k distinct positions uniformly spanning the whole clip."""
    k = max(1, min(k, total))
    return np.unique(np.linspace(0, total - 1, k).round().astype(np.int64))


def span_indices(total: int, budget: int, span_pct: int) -> np.ndarray:
    """budget positions inside a centered window covering span_pct of the clip."""
    width = max(1, int(round(total * span_pct / 100.0)))
    start = max(0, (total - width) // 2)
    end = min(total - 1, start + width - 1)
    return np.unique(np.linspace(start, end, min(budget, width)).round().astype(np.int64))


def covstride_indices(total: int, budget: int, coverage: int, stride: int) -> np.ndarray:
    """Submitted grid's candidate pool, WITHOUT the repeat-last padding."""
    window = max(1, int(round(total * coverage / 100.0)))
    candidates = np.arange(0, window, max(1, stride), dtype=np.int64)
    if len(candidates) == 0:
        candidates = np.array([0], dtype=np.int64)
    if len(candidates) > budget:
        pick = np.linspace(0, len(candidates) - 1, budget).round().astype(np.int64)
        candidates = candidates[pick]
    return candidates


def antialias_groups(total: int, k: int, filt: str,
                     max_per_window: int = 8) -> list[np.ndarray]:
    """Frame groups for the anti-aliasing intervention.

    Both arms deliver k samples spanning the whole clip; they differ only in
    whether each sample is a single frame or the mean of its window.

      point  frame at t_i                     -- what the paper does
      box    mean of frames in the window of
             width T/k centred on t_i         -- temporal low-pass before
                                                 sampling, so no energy remains
                                                 above the Nyquist limit k/2T

    If the accuracy loss is aliasing, removing the above-Nyquist energy has to
    recover accuracy: that is the defining signature of aliasing, and the same
    test used for spatial aliasing in anti-aliased CNNs. If the loss is missing
    evidence, low-pass filtering cannot help, because it adds no new evidence.

    Windows are subsampled to `max_per_window` frames so decoding stays bounded;
    a box filter over 8 evenly spaced samples is still a low-pass filter.
    """
    k = max(1, min(k, total))
    centres = np.linspace(0, total - 1, k)
    if filt == "point":
        return [np.array([int(round(c))], dtype=np.int64) for c in centres]
    # The box must be as wide as the actual sampling interval, which is the
    # spacing between centres -- (total-1)/(k-1), since linspace includes both
    # endpoints -- not total/k. Using total/k leaves gaps between windows, so
    # part of the signal is filtered by nothing and the cutoff no longer matches
    # the Nyquist limit the sampling imposes.
    spacing = (total - 1) / (k - 1) if k > 1 else float(total)
    half = spacing / 2.0
    groups = []
    for c in centres:
        lo, hi = max(0, int(np.floor(c - half))), min(total - 1, int(np.ceil(c + half)))
        span = np.arange(lo, hi + 1, dtype=np.int64)
        if len(span) > max_per_window:
            pick = np.linspace(0, len(span) - 1, max_per_window).round().astype(np.int64)
            span = span[pick]
        groups.append(span)
    return groups


def build_configs(mode: str, budget: int) -> list[dict]:
    if mode == "matched":
        return [dict(k=k) for k in K_VALUES]
    if mode == "span":
        return [dict(span_pct=w) for w in SPAN_PCTS]
    if mode == "antialias":
        return [dict(k=k, filt=f) for k in ANTIALIAS_KS for f in ("point", "box")]
    # covstride and covstride_pad share the grid and differ only in the sampler.
    return [dict(coverage=c, stride=s) for c in COVERAGES for s in STRIDES]


def config_groups(mode: str, cfg: dict, total: int, budget: int) -> list[np.ndarray]:
    """Frame groups per output slot; a group of one means plain point sampling."""
    if mode == "antialias":
        return antialias_groups(total, cfg["k"], cfg["filt"])
    if mode == "matched":
        idx = matched_indices(total, cfg["k"])
    elif mode == "span":
        idx = span_indices(total, budget, cfg["span_pct"])
    elif mode == "covstride_pad":
        # The published sampler, repeat-last padding included. Needed so the TRA
        # ablation is evaluated under the exact protocol whose cliff reviewer
        # XdvJ is asking about; the `covstride` mode removes the padding and so
        # cannot reproduce that cliff by construction.
        from info_rates.evaluation.benchmark import select_frame_indices
        idx = select_frame_indices(total, budget, cfg["coverage"], cfg["stride"])
    else:
        idx = covstride_indices(total, budget, cfg["coverage"], cfg["stride"])
    return [np.array([int(i)], dtype=np.int64) for i in idx]


def config_tag(mode: str, cfg: dict) -> str:
    if mode == "matched":
        return f"k{cfg['k']}"
    if mode == "span":
        return f"span{cfg['span_pct']}"
    if mode == "antialias":
        return f"k{cfg['k']}_{cfg['filt']}"
    return f"cov{cfg['coverage']}_s{cfg['stride']}"


@torch.inference_mode()
def run(
    manifest_df: pd.DataFrame,
    model,
    processor,
    mode: str,
    budget: int,
    resize: int,
    device: str,
    batch_size: int = 32,
    chunk_size: int = 128,
) -> pd.DataFrame:
    """Decode each video once, then evaluate every config from the frame cache."""
    configs = build_configs(mode, budget)
    num_labels = int(getattr(model.config, "num_labels",
                             len(getattr(model.config, "id2label", {})))) or 400
    device_obj = torch.device(device)

    df = manifest_df.copy()
    if "exists" in df.columns:
        df = df[df["exists"].astype(bool)]
    df = df.reset_index(drop=True)

    records: list[dict] = []
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    t0 = time.perf_counter()

    for chunk_idx in range(n_chunks):
        chunk = df.iloc[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]

        cache: list = []
        for row in chunk.itertuples(index=False):
            row_d = row._asdict()
            label_id = int(row_d["label_id"])
            if label_id < 0 or label_id >= num_labels:
                cache.append(None)
                continue
            try:
                vr = VideoReader(str(row_d["video_path"]), ctx=decord_cpu(0))
                total = len(vr)
                if total <= 0:
                    cache.append(None)
                    continue
                needed = set()
                for cfg in configs:
                    for g in config_groups(mode, cfg, total, budget):
                        needed.update(int(i) for i in g)
                needed_sorted = sorted(needed)
                raw = vr.get_batch(needed_sorted).asnumpy()
                resized = np.stack([cv2.resize(f, (resize, resize)) for f in raw])
                pos_map = {pos: i for i, pos in enumerate(needed_sorted)}
                cache.append((resized, pos_map, total, label_id, row_d))
            except Exception:
                cache.append(None)

        for cfg in configs:
            tag = config_tag(mode, cfg)
            batch_frames: list = []
            batch_meta: list = []

            def _flush(bf, bm):
                if not bf:
                    return
                inp = _move_batch_to_device(
                    processor(bf, return_tensors="pt"), device_obj)
                with torch.amp.autocast(device_type=device_obj.type,
                                        enabled=device_obj.type == "cuda"):
                    logits = model(**inp).logits
                probs = torch.softmax(logits.float(), dim=-1)
                conf, preds = probs.max(dim=-1)
                for pred, c, meta in zip(preds.cpu().numpy(),
                                         conf.cpu().numpy(), bm):
                    records.append({**meta, "config": tag,
                                    "correct_top1": int(pred) == meta["label_id"],
                                    "confidence": float(c)})
                del inp, logits, probs

            for entry in cache:
                if entry is None:
                    continue
                resized, pos_map, total, label_id, row_d = entry
                groups = config_groups(mode, cfg, total, budget)
                # A one-element group is point sampling; a longer group is
                # averaged, which is the temporal low-pass of the antialias arm.
                frames = [
                    resized[pos_map[int(g[0])]] if len(g) == 1
                    else np.mean([resized[pos_map[int(i)]] for i in g],
                                 axis=0).astype(resized.dtype)
                    for g in groups
                ]
                n_distinct = len(frames)
                frames = resample_to_length(frames, budget)
                batch_frames.append(frames)
                batch_meta.append({
                    "video_id": row_d.get("video_id"),
                    "label_id": label_id,
                    "source_frames": total,
                    "distinct_frames": n_distinct,
                    "model_input_frames": budget,
                    **cfg,
                })
                if len(batch_frames) >= batch_size:
                    _flush(batch_frames, batch_meta)
                    batch_frames, batch_meta = [], []
            _flush(batch_frames, batch_meta)

        done = min((chunk_idx + 1) * chunk_size, len(df))
        print(f"  chunk {chunk_idx + 1}/{n_chunks}  ({done}/{len(df)} videos"
              f"  {time.perf_counter() - t0:.0f}s)", flush=True)

    return pd.DataFrame(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CFG))
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CFG))
    ap.add_argument("--mode", required=True,
                    choices=["matched", "span", "covstride", "covstride_pad", "antialias"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N clips (smoke tests)")
    ap.add_argument("--checkpoint", default=None,
                    help="explicit checkpoint dir, bypassing get_checkpoint. "
                         "Needed for the TRA ablation, whose checkpoints are "
                         "named accv2026_<model>_<dataset>_tra_<arm> and so are "
                         "not discoverable by the standard resolver.")
    ap.add_argument("--tag", default="",
                    help="suffix for the output dir, to keep runs on the same "
                         "(model, dataset) from overwriting each other")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    mcfg = MODEL_CFG[args.model]
    dcfg = DATASET_CFG[args.dataset]
    budget, resize = mcfg["frames"], mcfg["resize"]

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = OUT_BASE / args.mode / f"{args.model}_{args.dataset}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    if summary_csv.exists() and not args.overwrite:
        print(f"[SKIP] already done: {summary_csv}")
        return

    manifest = ROOT / "evaluations/accv2026/manifests" / dcfg["manifest"]
    manifest_df = pd.read_csv(manifest)
    if args.limit:
        manifest_df = manifest_df.head(args.limit)

    print(f"=== {args.model} / {args.dataset} — mode={args.mode} ===")
    print(f"  budget={budget}  resize={resize}  videos={len(manifest_df)}")

    if args.checkpoint:
        # load_model resolves the path through get_checkpoint; redirect it so
        # non-standard checkpoints (the TRA arms) load through the same code
        # path as everything else, keeping results comparable.
        import sweep_coverage_stride as scs
        ckpt = Path(args.checkpoint)
        if not ckpt.is_dir():
            raise SystemExit(f"checkpoint dir not found: {ckpt}")
        scs.get_checkpoint = lambda *a, **k: ckpt
        model, processor, device = load_model(args.model, args.dataset)
        print(f"  device={device}  checkpoint={ckpt.name}")
    elif args.dataset == "kinetics400":
        model, processor, device = load_pretrained_k400(args.model)
        print(f"  device={device}  (K400-pretrained backbone, no fine-tuning)")
    else:
        model, processor, device = load_model(args.model, args.dataset)
        print(f"  device={device}")

    per_clip = run(manifest_df, model, processor, args.mode,
                   budget, resize, device,
                   batch_size=args.batch_size, chunk_size=args.chunk_size)
    if per_clip.empty:
        print("  [WARN] no results produced")
        return

    per_clip["model"] = args.model
    per_clip["dataset"] = args.dataset
    per_clip.to_csv(out_dir / "per_clip.csv", index=False)

    summary = (per_clip.groupby("config")
               .agg(top1=("correct_top1", "mean"),
                    n=("correct_top1", "size"),
                    mean_distinct=("distinct_frames", "mean"))
               .reset_index())
    summary["model"] = args.model
    summary["dataset"] = args.dataset
    summary["mode"] = args.mode
    summary["budget"] = budget
    summary.to_csv(summary_csv, index=False)

    print(f"\n=== {args.model} / {args.dataset} ({args.mode}) ===")
    for r in summary.itertuples():
        print(f"  {r.config:<16s} top1={r.top1 * 100:5.1f}%  "
              f"n={r.n:5d}  distinct={r.mean_distinct:5.1f}")
    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
