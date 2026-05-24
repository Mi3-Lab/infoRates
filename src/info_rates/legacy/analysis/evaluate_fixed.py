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

    # Debug: print manifest label sample and model label mapping
    # print("[DEBUG] Manifest label sample:", subset['label'].unique()[:10], flush=True)
    # print("[DEBUG] model.config.id2label:", getattr(model.config, 'id2label', None), flush=True)
    # print("[DEBUG] model.config.label2id:", getattr(model.config, 'label2id', None), flush=True)

    for stride in strides:
        for cov in coverages:
            correct = total = 0
            t0 = time.time()

            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = []
                for idx, (_, row) in enumerate(subset.iterrows()):
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
                    if idx % 50 == 0:
                        pass

                batch_frames, batch_labels = [], []
                fut_count = 0
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"stride={stride} cov={cov}%"):
                    frames, label = fut.result()
                    if frames is None:
                        continue
                    batch_frames.append(frames)
                    batch_labels.append(label)
                    fut_count += 1
                    if fut_count % 50 == 0:
                        pass

                    if len(batch_frames) == batch_size:
                        with torch.amp.autocast(device.type, dtype=torch.float16):
                            inputs = processor(batch_frames, return_tensors="pt").to(device)
                            logits = model(**inputs).logits
                        preds = logits.argmax(-1).cpu().numpy()  # Keep as integers
                        for p, l in zip(preds, batch_labels):
                            if int(p) == int(l):  # Compare integers directly
                                correct += 1
                        total += len(batch_labels)
                        batch_frames, batch_labels = [], []
                        # MEMORY FIX: Clear GPU tensors to prevent memory leak
                        del inputs, logits
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                # print(f"[DEBUG] Finished processing futures for stride={stride} cov={cov}%", flush=True)
                if batch_frames:
                    # print(f"[DEBUG] Processing remaining batch of size {len(batch_frames)} for stride={stride} cov={cov}%", flush=True)
                    with torch.amp.autocast(device.type, dtype=torch.float16):
                        inputs = processor(batch_frames, return_tensors="pt").to(device)
                        logits = model(**inputs).logits
                    preds = logits.argmax(-1).cpu().numpy()  # Keep as integers
                    for p, l in zip(preds, batch_labels):
                        if int(p) == int(l):  # Compare integers directly
                            correct += 1
                    total += len(batch_labels)
                    # MEMORY FIX: Clear GPU tensors to prevent memory leak
                    del inputs, logits
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                # print(f"[DEBUG] Finished processing remaining batch for stride={stride} cov={cov}%", flush=True)

            acc = correct / max(1, total)
            print(f"[RESULT] stride={stride} cov={cov}%: acc={acc:.4f}", flush=True)
            results.append({
                "coverage": cov,
                "stride": stride,
                "accuracy": acc,
                "avg_time": (time.time() - t0) / max(1, total),
                "correct": correct,
                "total": total,
            })

    # print(f"[DEBUG] Returning results with {len(results)} entries", flush=True)
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
    num_frames: int = None,
    label_map: dict = None,
    topk: int = 1,
    fuzzy_threshold: float = 0.5,
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


    id2label = getattr(model.config, 'id2label', None)
    label2id = getattr(model.config, 'label2id', None)
    n_classes = len(id2label) if id2label is not None else 0

    # Check if labels are integers (Kinetics400 case) or strings (UCF101 case)
    sample_labels = subset['label'].unique()[:10]
    labels_are_integers = all(str(l).isdigit() for l in sample_labels)

    # Build textual->id mapping if provided
    txt_to_id = {}
    if label_map:
        for k, v in label_map.items():
            txt_to_id[k] = v if isinstance(v, (list, tuple)) else int(v)

    if labels_are_integers:
        # For integer labels (Kinetics400), create direct mapping
        print(f"[INFO] Detected integer labels, using direct integer mapping for {n_classes} classes")
        def get_class_id(label):
            return int(label)
        def get_class_name(class_id):
            return id2label[class_id] if id2label and class_id < len(id2label) else f"class_{class_id}"
    else:
        # For string labels, try label_map first, then model's label2id mapping
        print(f"[INFO] Detected string labels, using label_map and model label2id mapping")
        def get_class_id(label):
            if label is None:
                return -1
            try:
                return int(label)
            except Exception:
                pass
            ln = norm_label(label) if isinstance(label, str) else str(label)
            if ln in txt_to_id:
                v = txt_to_id[ln]
                return int(v) if not isinstance(v, (list, tuple)) else int(v[0])
            # Try pattern-based template match if provided
            if label_map and '__patterns' in label_map:
                import re
                for pat, t_id, sth in label_map['__patterns']:
                    try:
                        if pat.search(str(label)):
                            return int(t_id)
                    except Exception:
                        continue
            if label2id:
                return label2id.get(ln, -1)
            return -1
        def get_class_name(class_id):
            return id2label[class_id] if id2label and class_id < len(id2label) else f"class_{class_id}"

    if not id2label:
        print("[ERROR] Model config is missing id2label! Check model checkpoint.", flush=True)
        return pd.DataFrame()
    # Print a sample of manifest labels
    # print("[DEBUG] Sample manifest labels:", subset['label'].unique()[:10], flush=True)

    # Print progress inside batch loop
    batch_print_interval = max(1, len(subset) // 10)


    for stride in strides:
        for cov in coverages:
            correct_per_class = np.zeros(n_classes, dtype=np.int32)
            total_per_class = np.zeros(n_classes, dtype=np.int32)
            t0 = time.time()


            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = [ex.submit(extract_and_prepare, row._asdict() if hasattr(row, "_asdict") else row.to_dict(), cov, stride, num_select=num_frames) for _, row in subset.iterrows()]

                batch_frames, batch_labels = [], []
                fut_count = 0
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"stride={stride} cov={cov}%"):
                    frames, label = fut.result()
                    if frames is None:
                        continue
                    batch_frames.append(frames)
                    batch_labels.append(label)

                    fut_count += 1
                    if fut_count % batch_print_interval == 0:
                        pass

                    if len(batch_frames) == batch_size:
                        with torch.amp.autocast(device.type, dtype=torch.float16):
                            inputs = processor(batch_frames, return_tensors="pt").to(device)
                            logits = model(**inputs).logits
                        preds = logits.argmax(-1).cpu().numpy()
                        for p, l in zip(preds, batch_labels):
                            true_id = get_class_id(l)
                            if true_id == -1:
                                continue
                            total_per_class[true_id] += 1
                            if p == true_id:
                                correct_per_class[true_id] += 1
                        batch_frames, batch_labels = [], []
                        # MEMORY FIX: Clear GPU tensors to prevent memory leak
                        del inputs, logits, preds
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                if batch_frames:
                    with torch.amp.autocast(device.type, dtype=torch.float16):
                        inputs = processor(batch_frames, return_tensors="pt").to(device)
                        logits = model(**inputs).logits
                    preds = logits.argmax(-1).cpu().numpy()
                    for p, l in zip(preds, batch_labels):
                        true_id = get_class_id(l)
                        if true_id == -1:
                            continue
                        total_per_class[true_id] += 1
                        if p == true_id:
                            correct_per_class[true_id] += 1
                    # MEMORY FIX: Clear GPU tensors to prevent memory leak
                    del inputs, logits, preds
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            accs = correct_per_class / np.maximum(1, total_per_class)
            for i in range(n_classes):
                if total_per_class[i] > 0:
                    results.append({
                        "class": get_class_name(i),
                        "coverage": cov,
                        "stride": stride,
                        "accuracy": float(accs[i]),
                        "n_samples": int(total_per_class[i])
                    })

            avg_time = (time.time() - t0) / np.maximum(1, total_per_class.sum())
            print(f"✅ stride={stride} cov={cov}% | mean time: {avg_time:.3f}s/frame")

    return pd.DataFrame(results)
