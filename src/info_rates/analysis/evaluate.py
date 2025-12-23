import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..sampling.temporal import extract_and_prepare, norm_label


# IMPORTANT: GPU Memory Management
# All inference functions in this module now include explicit tensor cleanup
# to prevent memory leaks during long evaluation runs with multiple coverage/stride
# combinations. The cleanup pattern is:
#   del inputs, logits  # Release tensor references
#   if device.type == "cuda":
#       torch.cuda.empty_cache()  # Clear CUDA cache


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
    rank: int = 0,
    num_frames: int = None,
    checkpoint_path: str = None,
) -> pd.DataFrame:
    # Auto-detect model's frame requirement if not provided
    if num_frames is None:
        model_config = getattr(model, 'config', None)
        if model_config is None:
            num_frames = 8  # Default fallback
        elif hasattr(model_config, 'num_frames'):
            num_frames = model_config.num_frames
        else:
            model_type = getattr(model_config, 'model_type', '').lower()
            if 'vivit' in model_type:
                num_frames = 32
            elif 'videomae' in model_type:
                num_frames = 16
            else:  # TimeSformer or unknown
                num_frames = 8
    
    # sample_size <= 0 means use full dataset
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        subset = df.sample(sample_size, random_state=42)
    else:
        subset = df

    # Load checkpoint if exists
    completed = set()
    results = []
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            prev = pd.read_csv(checkpoint_path)
            for row in prev.itertuples():
                completed.add((row.coverage, row.stride))
            results = prev.to_dict(orient="records")
            if rank == 0:
                pass
        except Exception as e:
            if rank == 0:
                pass
    device = next(model.parameters()).device

    rng = np.random.default_rng(42)

    for stride in strides:
        for cov in coverages:
            if (cov, stride) in completed:
                if rank == 0:
                    pass
                continue
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
                            num_select=num_frames,
                        )
                    )

                batch_frames, batch_labels = [], []
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"stride={stride} cov={cov}%", disable=(rank != 0)):
                    frames, label = fut.result()
                    if frames is None:
                        continue
                    batch_frames.append(frames)
                    batch_labels.append(label)

                    if len(batch_frames) == batch_size:
                        with torch.amp.autocast(device.type, dtype=torch.float16):
                            inputs = processor(batch_frames, return_tensors="pt").to(device)

                            for stride in strides:
                                for cov in coverages:
                                    if (cov, stride) in completed:
                                        if rank == 0:
                                            pass
                                        continue
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
                                                    num_select=num_frames,
                                                )
                                            )

                                        batch_frames, batch_labels = [], []
                                        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"stride={stride} cov={cov}%", disable=(rank != 0)):
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
                                                # Release tensor references to prevent memory leak
                                                del inputs, logits

                                        if batch_frames:
                                            with torch.amp.autocast(device.type, dtype=torch.float16):
                                                inputs = processor(batch_frames, return_tensors="pt").to(device)
                                                logits = model(**inputs).logits
                                            preds = [model.config.id2label[i] for i in logits.argmax(-1).cpu().numpy()]
                                            for p, l in zip(preds, batch_labels):
                                                if norm_label(p) == norm_label(l):
                                                    correct += 1
                                            total += len(batch_labels)
                                            # Release tensor references to prevent memory leak
                                            del inputs, logits

                                    # Clear GPU cache between coverage/stride combinations
                                    if device.type == "cuda":
                                        torch.cuda.empty_cache()

                                    total_time = (time.time() - t0)
                                    acc = (correct / total) if total > 0 else 0.0
                                    results.append({
                                        "coverage": cov,
                                        "stride": stride,
                                        "accuracy": acc,
                                        "correct": correct,
                                        "total": total,
                                        "total_time": total_time,
                                    })

                                    # Save checkpoint after each config
                                    if checkpoint_path is not None and rank == 0:
                                        pd.DataFrame(results).to_csv(checkpoint_path, index=False)
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
                            num_select=num_frames,
                        )
                    )

                batch_frames, batch_labels = [], []
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"stride={stride} cov={cov}%", disable=(rank != 0)):
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
                        # Release tensor references to prevent memory leak
                        del inputs, logits

                if batch_frames:
                    with torch.amp.autocast(device.type, dtype=torch.float16):
                        inputs = processor(batch_frames, return_tensors="pt").to(device)
                        logits = model(**inputs).logits
                    preds = [model.config.id2label[i] for i in logits.argmax(-1).cpu().numpy()]
                    for p, l in zip(preds, batch_labels):
                        if norm_label(p) == norm_label(l):
                            correct += 1
                    total += len(batch_labels)
                    # Release tensor references to prevent memory leak
                    del inputs, logits

            # Clear GPU cache between coverage/stride combinations
            if device.type == "cuda":
                torch.cuda.empty_cache()

            total_time = (time.time() - t0)
            acc = (correct / total) if total > 0 else 0.0
            results.append({
                "coverage": cov,
                "stride": stride,
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "total_time": total_time,
            })

    return results


def per_class_analysis_fast(
    df: pd.DataFrame,
    processor,
    model,
    coverages: List[int] = [10, 25, 50, 75, 100],
    strides: List[int] = [1, 2, 4, 8, 16],
    sample_size: int = 200,
    batch_size: int = 8,
    num_workers: int = 8,
    rank: int = 0,
    num_frames: int = 8,
    checkpoint_path: str = None,
) -> pd.DataFrame:
    # sample_size <= 0 means use full dataset
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        subset = df.sample(sample_size, random_state=42)
    else:
        subset = df
    results = []
    completed = set()
    # Load checkpoint if exists
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            prev = pd.read_csv(checkpoint_path)
            for row in prev.itertuples():
                completed.add((row.coverage, row.stride))
            results = prev.to_dict(orient="records")
            if rank == 0:
                pass
        except Exception as e:
            if rank == 0:
                pass
    device = next(model.parameters()).device

    id2label = model.config.id2label
    label2id = model.config.label2id
    n_classes = len(id2label)

    for stride in strides:
        for cov in coverages:
            if (cov, stride) in completed:
                if rank == 0:
                    pass
                continue
            correct_per_class = np.zeros(n_classes, dtype=np.int32)
            total_per_class = np.zeros(n_classes, dtype=np.int32)
            t0 = time.time()

            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = [ex.submit(extract_and_prepare, row._asdict() if hasattr(row, "_asdict") else row.to_dict(), cov, stride, num_select=num_frames) for _, row in subset.iterrows()]

                batch_frames, batch_labels = [], []
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"stride={stride} cov={cov}%", disable=(rank != 0)):
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
                        # Release tensor references to prevent memory leak
                        del inputs, logits, preds

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
                    # Release tensor references to prevent memory leak
                    del inputs, logits, preds

            # Clear GPU cache between coverage/stride combinations
            if device.type == "cuda":
                torch.cuda.empty_cache()

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
            mean_acc = accs[total_per_class > 0].mean() if (total_per_class > 0).any() else 0.0
            if rank == 0:
                pass

            # Save checkpoint after each config
            if checkpoint_path is not None and rank == 0:
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(results)
    # sample_size <= 0 means use full dataset
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        subset = df.sample(sample_size, random_state=42)
    else:
        subset = df
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
                futures = [ex.submit(extract_and_prepare, row._asdict() if hasattr(row, "_asdict") else row.to_dict(), cov, stride, num_select=num_frames) for _, row in subset.iterrows()]

                batch_frames, batch_labels = [], []
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"stride={stride} cov={cov}%", disable=(rank != 0)):
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
                        # Release tensor references to prevent memory leak
                        del inputs, logits, preds

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
                    # Release tensor references to prevent memory leak
                    del inputs, logits, preds

            # Clear GPU cache between coverage/stride combinations
            if device.type == "cuda":
                torch.cuda.empty_cache()

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
            mean_acc = accs[total_per_class > 0].mean() if (total_per_class > 0).any() else 0.0
            if rank == 0:
                pass

            # Save checkpoint after each config
            if checkpoint_path is not None and rank == 0:
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(results)
