"""Trusted fixed-budget video evaluation for ACCV 2026 experiments."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from tqdm import tqdm


@dataclass
class SampleResult:
    dataset: str
    split: str
    video_id: str
    video_path: str
    label_id: int
    budget: int
    coverage: int
    stride: int
    pred_id: int
    correct_top1: bool
    correct_top5: bool
    confidence: float
    decode_time_s: float
    inference_time_s: float
    total_time_s: float
    source_frames: int
    candidate_frames: int
    processed_frames: int
    model_input_frames: int = 0
    skipped: bool = False
    error: str = ""


def select_frame_indices(
    total_frames: int,
    budget: int,
    coverage: int = 100,
    stride: int = 1,
) -> np.ndarray:
    """Select deterministic frame indices for a fixed temporal budget."""
    if total_frames <= 0:
        return np.array([], dtype=np.int64)
    budget = max(1, int(budget))
    coverage = int(np.clip(coverage, 1, 100))
    stride = max(1, int(stride))

    window = max(1, int(round(total_frames * coverage / 100.0)))
    candidates = np.arange(0, window, stride, dtype=np.int64)
    if len(candidates) == 0:
        candidates = np.array([0], dtype=np.int64)

    if len(candidates) >= budget:
        pick = np.linspace(0, len(candidates) - 1, budget).round().astype(np.int64)
        return candidates[pick]

    # Pad by repeating the last real frame. This keeps model input shape stable
    # while preserving how many distinct candidate frames were available.
    pad = np.full(budget - len(candidates), candidates[-1], dtype=np.int64)
    return np.concatenate([candidates, pad])


def adapt_frames_for_model(frames: list[np.ndarray], target_frames: int | None) -> list[np.ndarray]:
    """Pad or downsample decoded evidence frames to a model's fixed temporal length."""
    if not target_frames or target_frames <= 0 or len(frames) == target_frames:
        return frames
    if not frames:
        return frames
    if len(frames) > target_frames:
        idxs = np.linspace(0, len(frames) - 1, target_frames).round().astype(np.int64)
        return [frames[int(i)] for i in idxs]
    return frames + [frames[-1]] * (target_frames - len(frames))


def decode_video_frames(
    video_path: str | Path,
    budget: int,
    coverage: int = 100,
    stride: int = 1,
    resize: int = 224,
) -> tuple[list[np.ndarray], dict]:
    """Decode selected RGB frames from a video."""
    t0 = time.perf_counter()
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        indices = select_frame_indices(total_frames, budget, coverage, stride)
        if len(indices) == 0:
            raise ValueError("video has no decodable frames")
        frames = vr.get_batch(indices).asnumpy()
    except Exception as decord_error:
        frames, total_frames, indices = _decode_video_frames_cv2(video_path, budget, coverage, stride)
        if frames is None:
            raise decord_error
    if resize:
        frames = np.stack([cv2.resize(frame, (resize, resize)) for frame in frames], axis=0)
    elapsed = time.perf_counter() - t0
    metadata = {
        "decode_time_s": elapsed,
        "source_frames": int(total_frames),
        "candidate_frames": int(len(np.unique(indices))),
        "processed_frames": int(len(frames)),
    }
    return [frame for frame in frames], metadata


def _decode_video_frames_cv2(
    video_path: str | Path,
    budget: int,
    coverage: int,
    stride: int,
) -> tuple[np.ndarray | None, int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None, 0, np.array([], dtype=np.int64)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = select_frame_indices(total_frames, budget, coverage, stride)
    frames = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None, total_frames, indices
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None, total_frames, indices
    return np.stack(frames, axis=0), total_frames, indices


def iter_manifest(
    manifest: pd.DataFrame,
    split: str | None = None,
    max_samples: int = 0,
    samples_per_class: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Filter and optionally class-balance a manifest."""
    df = manifest.copy()
    if split and split != "all" and "split" in df.columns:
        df = df[df["split"].astype(str) == split].copy()
    if "exists" in df.columns:
        df = df[df["exists"].astype(bool)].copy()
    if samples_per_class > 0:
        parts = []
        for _, group in df.groupby("label_id", sort=False):
            n = min(len(group), samples_per_class)
            parts.append(group.sample(n, random_state=seed))
        df = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
    if max_samples and max_samples > 0 and max_samples < len(df):
        df = df.sample(max_samples, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


@torch.inference_mode()
def evaluate_fixed_budgets(
    manifest: pd.DataFrame,
    model,
    processor,
    budgets: Iterable[int],
    output_csv: str | Path,
    split: str | None = None,
    coverage: int = 100,
    stride: int = 1,
    batch_size: int = 8,
    max_samples: int = 0,
    samples_per_class: int = 0,
    device: str = "cuda",
    resize: int = 224,
    model_frames: int = 0,
    save_logits: bool = False,
    logits_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate a video classifier over fixed frame budgets."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    logits_dir = Path(logits_dir) if logits_dir else output_csv.parent / "logits"
    if save_logits:
        logits_dir.mkdir(parents=True, exist_ok=True)

    df = iter_manifest(
        manifest,
        split=split,
        max_samples=max_samples,
        samples_per_class=samples_per_class,
    )
    if df.empty:
        raise ValueError("manifest selection is empty")

    num_labels = int(getattr(model.config, "num_labels", len(getattr(model.config, "id2label", {}))))
    model.eval()
    device_obj = torch.device(device)
    results: list[dict] = []

    for budget in budgets:
        batch_frames = []
        batch_rows = []
        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"budget={budget}"):
            row_dict = row._asdict()
            label_id = int(row_dict["label_id"])
            if label_id < 0 or label_id >= num_labels:
                results.append(
                    asdict(
                        SampleResult(
                            dataset=str(row_dict.get("dataset", "")),
                            split=str(row_dict.get("split", "")),
                            video_id=str(row_dict.get("video_id", "")),
                            video_path=str(row_dict["video_path"]),
                            label_id=label_id,
                            budget=int(budget),
                            coverage=int(coverage),
                            stride=int(stride),
                            pred_id=-1,
                            correct_top1=False,
                            correct_top5=False,
                            confidence=0.0,
                            decode_time_s=0.0,
                            inference_time_s=0.0,
                            total_time_s=0.0,
                            source_frames=0,
                            candidate_frames=0,
                            processed_frames=0,
                            skipped=True,
                            error=f"label_id {label_id} outside model num_labels {num_labels}",
                        )
                    )
                )
                continue
            t0 = time.perf_counter()
            try:
                frames, meta = decode_video_frames(row_dict["video_path"], budget, coverage, stride, resize)
                meta["total_start"] = t0
                frames = adapt_frames_for_model(frames, model_frames)
                meta["model_input_frames"] = len(frames)
                batch_frames.append(frames)
                batch_rows.append((row_dict, label_id, meta))
            except Exception as exc:
                results.append(
                    asdict(
                        SampleResult(
                            dataset=str(row_dict.get("dataset", "")),
                            split=str(row_dict.get("split", "")),
                            video_id=str(row_dict.get("video_id", "")),
                            video_path=str(row_dict["video_path"]),
                            label_id=label_id,
                            budget=int(budget),
                            coverage=int(coverage),
                            stride=int(stride),
                            pred_id=-1,
                            correct_top1=False,
                            correct_top5=False,
                            confidence=0.0,
                            decode_time_s=0.0,
                            inference_time_s=0.0,
                            total_time_s=time.perf_counter() - t0,
                            source_frames=0,
                            candidate_frames=0,
                            processed_frames=0,
                            skipped=True,
                            error=str(exc),
                        )
                    )
                )
                continue

            if len(batch_frames) >= batch_size:
                _flush_batch(
                    batch_frames,
                    batch_rows,
                    model,
                    processor,
                    results,
                    budget,
                    coverage,
                    stride,
                    device_obj,
                    save_logits,
                    logits_dir,
                )
                batch_frames, batch_rows = [], []

        if batch_frames:
            _flush_batch(
                batch_frames,
                batch_rows,
                model,
                processor,
                results,
                budget,
                coverage,
                stride,
                device_obj,
                save_logits,
                logits_dir,
            )

        pd.DataFrame(results).to_csv(output_csv, index=False)

    return pd.DataFrame(results)


def _flush_batch(
    batch_frames,
    batch_rows,
    model,
    processor,
    results,
    budget,
    coverage,
    stride,
    device_obj,
    save_logits,
    logits_dir,
) -> None:
    t_infer = time.perf_counter()
    inputs = _move_batch_to_device(processor(batch_frames, return_tensors="pt"), device_obj)
    with torch.amp.autocast(device_type=device_obj.type, enabled=device_obj.type == "cuda"):
        logits = model(**inputs).logits
    inference_time = time.perf_counter() - t_infer
    probs = torch.softmax(logits.float(), dim=-1)
    topk = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1)

    logits_cpu = logits.detach().cpu()
    top_indices = topk.indices.detach().cpu().numpy()
    top_values = topk.values.detach().cpu().numpy()

    for idx, (row_dict, label_id, meta) in enumerate(batch_rows):
        pred_id = int(top_indices[idx, 0])
        correct_top1 = pred_id == label_id
        correct_top5 = bool(label_id in set(int(x) for x in top_indices[idx]))
        if save_logits:
            np.save(logits_dir / f"{row_dict.get('video_id', idx)}_budget{budget}.npy", logits_cpu[idx].numpy())
        results.append(
            asdict(
                SampleResult(
                    dataset=str(row_dict.get("dataset", "")),
                    split=str(row_dict.get("split", "")),
                    video_id=str(row_dict.get("video_id", "")),
                    video_path=str(row_dict["video_path"]),
                    label_id=int(label_id),
                    budget=int(budget),
                    coverage=int(coverage),
                    stride=int(stride),
                    pred_id=pred_id,
                    correct_top1=bool(correct_top1),
                    correct_top5=correct_top5,
                    confidence=float(top_values[idx, 0]),
                    decode_time_s=float(meta["decode_time_s"]),
                    inference_time_s=float(inference_time / max(1, len(batch_rows))),
                    total_time_s=float(time.perf_counter() - meta["total_start"]),
                    source_frames=int(meta["source_frames"]),
                    candidate_frames=int(meta["candidate_frames"]),
                    processed_frames=int(meta["processed_frames"]),
                    model_input_frames=int(meta.get("model_input_frames", meta["processed_frames"])),
                )
            )
        )

    del inputs, logits, probs, topk
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()


def _move_batch_to_device(inputs, device_obj):
    if hasattr(inputs, "to"):
        return inputs.to(device_obj)
    if isinstance(inputs, dict):
        return {k: v.to(device_obj) if torch.is_tensor(v) else v for k, v in inputs.items()}
    raise TypeError(f"Unsupported processor output type: {type(inputs)!r}")


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    valid = results[~results["skipped"].astype(bool)].copy()
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby(["dataset", "split", "budget", "coverage", "stride"], dropna=False)
        .agg(
            n=("correct_top1", "size"),
            top1=("correct_top1", "mean"),
            top5=("correct_top5", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_decode_time_s=("decode_time_s", "mean"),
            mean_inference_time_s=("inference_time_s", "mean"),
            mean_total_time_s=("total_time_s", "mean"),
            mean_processed_frames=("processed_frames", "mean"),
            mean_model_input_frames=("model_input_frames", "mean"),
            mean_source_frames=("source_frames", "mean"),
        )
        .reset_index()
    )
