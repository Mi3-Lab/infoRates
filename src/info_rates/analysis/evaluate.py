import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..sampling.temporal import extract_and_prepare, norm_label


def evaluate_fixed_parallel(
    df: pd.DataFrame,
    processor,
    model,
    coverages: List[int] = [10, 25, 50, 75, 100],
    strides: List[int] = [1, 2, 4, 8, 16],
    sample_size: int = 200,
    batch_size: int = 8,
    num_workers: int = 8,
    jitter_coverage_pct: float = 0.0,
) -> pd.DataFrame:
    # sample_size <= 0 means use full dataset
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        subset = df.sample(sample_size, random_state=42)
    else:
        subset = df
    results = []
    device = next(model.parameters()).device

    rng = np.random.default_rng(42)

    for stride in strides:
        for cov in coverages:
            correct = total = 0
            t0 = time.time()

            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = []
                for _, row in subset.iterrows():
                    cov_use = cov
                    if jitter_coverage_pct > 0:
                        delta = cov * (jitter_coverage_pct / 100.0)
                        low = max(1, cov - delta)
                        high = min(100, cov + delta)
                        cov_use = int(np.clip(rng.uniform(low, high), 1, 100))
                    futures.append(
                        ex.submit(
                            extract_and_prepare,
                            row._asdict() if hasattr(row, "_asdict") else row.to_dict(),
                            cov_use,
                            stride,
                        )
                    )

                batch_frames, batch_labels = [], []
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"stride={stride} cov={cov}%"):
                    frames, label = fut.result()
                    if frames is None:
                        continue
                    batch_frames.append(frames)
                    batch_labels.append(label)

                    if len(batch_frames) == batch_size:
                        with torch.amp.autocast(device.type, dtype=torch.float16):
                            inputs = processor(batch_frames, return_tensors="pt").to(device)
                            logits = model(**inputs).logits
                        preds = [model.config.id2label[i] for i in logits.argmax(-1).cpu().numpy()]
                        for p, l in zip(preds, batch_labels):
                            if norm_label(p) == norm_label(l):
                                correct += 1
                        total += len(batch_labels)
                        batch_frames, batch_labels = [], []

                if batch_frames:
                    with torch.amp.autocast(device.type, dtype=torch.float16):
                        inputs = processor(batch_frames, return_tensors="pt").to(device)
                        logits = model(**inputs).logits
                    preds = [model.config.id2label[i] for i in logits.argmax(-1).cpu().numpy()]
                    for p, l in zip(preds, batch_labels):
                        if norm_label(p) == norm_label(l):
                            correct += 1
                    total += len(batch_labels)

            acc = correct / max(1, total)
            results.append({
                "coverage": cov,
                "stride": stride,
                "accuracy": acc,
                "avg_time": (time.time() - t0) / max(1, total),
                "correct": correct,
                "total": total,
            })

    return pd.DataFrame(results)


def evaluate_fixed_parallel_counts(
    df: pd.DataFrame,
    processor,
    model,
    coverages: List[int] = [10, 25, 50, 75, 100],
    strides: List[int] = [1, 2, 4, 8, 16],
    sample_size: int = 200,
    batch_size: int = 8,
    num_workers: int = 8,
    jitter_coverage_pct: float = 0.0,
):
    """
    Like evaluate_fixed_parallel but returns raw counts and total_time for aggregation.

    Returns a list of dicts with keys: coverage, stride, correct, total, total_time
    """
    # sample_size <= 0 means use full dataset
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        subset = df.sample(sample_size, random_state=42)
    else:
        subset = df
    results = []
    device = next(model.parameters()).device

    rng = np.random.default_rng(42)

    for stride in strides:
        for cov in coverages:
            correct = total = 0
            t0 = time.time()

            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = []
                for _, row in subset.iterrows():
                    cov_use = cov
                    if jitter_coverage_pct > 0:
                        delta = cov * (jitter_coverage_pct / 100.0)
                        low = max(1, cov - delta)
                        high = min(100, cov + delta)
                        cov_use = int(np.clip(rng.uniform(low, high), 1, 100))
                    futures.append(
                        ex.submit(
                            extract_and_prepare,
                            row._asdict() if hasattr(row, "_asdict") else row.to_dict(),
                            cov_use,
                            stride,
                        )
                    )

                batch_frames, batch_labels = [], []
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"stride={stride} cov={cov}%"):
                    frames, label = fut.result()
                    if frames is None:
                        continue
                    batch_frames.append(frames)
                    batch_labels.append(label)

                    if len(batch_frames) == batch_size:
                        with torch.amp.autocast(device.type, dtype=torch.float16):
                            inputs = processor(batch_frames, return_tensors="pt").to(device)
                            logits = model(**inputs).logits
                        preds = [model.config.id2label[i] for i in logits.argmax(-1).cpu().numpy()]
                        for p, l in zip(preds, batch_labels):
                            if norm_label(p) == norm_label(l):
                                correct += 1
                        total += len(batch_labels)
                        batch_frames, batch_labels = [], []

                if batch_frames:
                    with torch.amp.autocast(device.type, dtype=torch.float16):
                        inputs = processor(batch_frames, return_tensors="pt").to(device)
                        logits = model(**inputs).logits
                    preds = [model.config.id2label[i] for i in logits.argmax(-1).cpu().numpy()]
                    for p, l in zip(preds, batch_labels):
                        if norm_label(p) == norm_label(l):
                            correct += 1
                    total += len(batch_labels)

            total_time = (time.time() - t0)
            results.append({
                "coverage": cov,
                "stride": stride,
                "correct": correct,
                "total": total,
                "total_time": total_time,
            })

    return results


def per_class_analysis_fast(df: pd.DataFrame, processor, model, coverages: List[int] = [10, 25, 50, 75, 100], strides: List[int] = [1, 2, 4, 8, 16], sample_size: int = 200, batch_size: int = 8, num_workers: int = 8) -> pd.DataFrame:
    subset = df.sample(sample_size, random_state=42) if sample_size < len(df) else df
    results = []
    device = next(model.parameters()).device

    id2label = model.config.id2label
    label2id = model.config.label2id
    n_classes = len(id2label)

    for stride in strides:
        for cov in coverages:
            correct_per_class = np.zeros(n_classes, dtype=np.int32)
            total_per_class = np.zeros(n_classes, dtype=np.int32)
            t0 = time.time()

            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = [ex.submit(extract_and_prepare, row._asdict() if hasattr(row, "_asdict") else row.to_dict(), cov, stride) for _, row in subset.iterrows()]

                batch_frames, batch_labels = [], []
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"stride={stride} cov={cov}%"):
                    frames, label = fut.result()
                    if frames is None:
                        continue
                    batch_frames.append(frames)
                    batch_labels.append(label)

                    if len(batch_frames) == batch_size:
                        with torch.amp.autocast(device.type, dtype=torch.float16):
                            inputs = processor(batch_frames, return_tensors="pt").to(device)
                            logits = model(**inputs).logits
                        preds = logits.argmax(-1).cpu().numpy()
                        for p, l in zip(preds, batch_labels):
                            if l not in label2id:
                                continue
                            true_id = label2id[l]
                            total_per_class[true_id] += 1
                            if p == true_id:
                                correct_per_class[true_id] += 1
                        batch_frames, batch_labels = [], []

                if batch_frames:
                    with torch.amp.autocast(device.type, dtype=torch.float16):
                        inputs = processor(batch_frames, return_tensors="pt").to(device)
                        logits = model(**inputs).logits
                    preds = logits.argmax(-1).cpu().numpy()
                    for p, l in zip(preds, batch_labels):
                        if l not in label2id:
                            continue
                        true_id = label2id[l]
                        total_per_class[true_id] += 1
                        if p == true_id:
                            correct_per_class[true_id] += 1

            accs = correct_per_class / np.maximum(1, total_per_class)
            for i in range(n_classes):
                if total_per_class[i] > 0:
                    results.append({
                        "class": id2label[i],
                        "coverage": cov,
                        "stride": stride,
                        "accuracy": float(accs[i]),
                        "n_samples": int(total_per_class[i])
                    })

            avg_time = (time.time() - t0) / np.maximum(1, total_per_class.sum())
            print(f"stride={stride} cov={cov}% | mean time: {avg_time:.3f}s/frame")

    return pd.DataFrame(results)
