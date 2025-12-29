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
    label_map: dict = None,
    topk: int = 1,
    fuzzy_threshold: float = 0.5,
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
                        # Support top-k matching for textual labels
                        topk_vals = logits.topk(topk, dim=-1).indices.cpu().numpy()
                        for pred_row, l in zip(topk_vals, batch_labels):
                            matched = False
                            # Try integer comparison first
                            try:
                                if any(int(p) == int(l) for p in pred_row):
                                    correct += 1
                                    continue
                            except Exception:
                                pass

                            # Try explicit mapping from textual label -> id(s)
                            if label_map and isinstance(l, str):
                                mapped = label_map.get(norm_label(l))
                                if mapped is not None:
                                    mapped_ids = mapped if isinstance(mapped, (list, tuple)) else [int(mapped)]
                                    if any(int(p) in mapped_ids for p in pred_row):
                                        correct += 1
                                        continue
                                # Try pattern-based template match (e.g., template 'lifting something with something' matches 'lifting chair toy with car toy on it')
                                if '__patterns' in label_map:
                                    for pat, t_id, sth in label_map['__patterns']:
                                        try:
                                            if pat.search(str(l)):
                                                if any(int(p) == int(t_id) for p in pred_row):
                                                    correct += 1
                                                    mapped = t_id
                                                    break
                                        except Exception:
                                            continue
                                    if mapped is not None:
                                        continue
                                # Try template-token fuzzy match against mapping templates (if provided)
                                if '__templates' in label_map:
                                    import re
                                    def _tokens(x: str):
                                        toks = set(re.findall(r"\w+", x.lower()))
                                        toks -= {"a", "the", "an", "something"}
                                        return toks
                                    gt_toks = _tokens(l)
                                    for t_toks, t_id in label_map['__templates']:
                                        overlap = len(gt_toks & t_toks) / max(1, min(len(gt_toks), len(t_toks)))
                                        if overlap >= fuzzy_threshold:
                                            if any(int(p) == int(t_id) for p in pred_row):
                                                correct += 1
                                                mapped = t_id
                                                break
                                    if mapped is not None:
                                        continue

                            # Fallback: compare predicted label text to ground-truth text
                            for p in pred_row:
                                p_label = model.config.id2label[int(p)]
                                if norm_label(p_label) == norm_label(l):
                                    matched = True
                                    break
                                # token overlap fuzzy match (use word tokens, ignore stopwords and the placeholder 'something')
                                import re
                                def _tokens(x: str):
                                    toks = set(re.findall(r"\w+", x.lower()))
                                    toks -= {"a", "the", "an", "something"}
                                    return toks
                                gt_tokens = _tokens(l)
                                p_tokens = _tokens(p_label)
                                if gt_tokens and p_tokens:
                                    overlap = len(gt_tokens & p_tokens) / max(1, min(len(gt_tokens), len(p_tokens)))
                                    if overlap >= fuzzy_threshold:
                                        matched = True
                                        break
                            if matched:
                                correct += 1
                        total += len(batch_labels)
                        batch_frames, batch_labels = [], []
                        # Release tensor references to prevent memory leak
                        del inputs, logits

                if batch_frames:
                    with torch.amp.autocast(device.type, dtype=torch.float16):
                        inputs = processor(batch_frames, return_tensors="pt").to(device)
                        logits = model(**inputs).logits
                        topk_vals = logits.topk(topk, dim=-1).indices.cpu().numpy()
                        for pred_row, l in zip(topk_vals, batch_labels):
                            matched = False
                            try:
                                if any(int(p) == int(l) for p in pred_row):
                                    correct += 1
                                    continue
                            except Exception:
                                pass

                            if label_map and isinstance(l, str):
                                mapped = label_map.get(norm_label(l))
                                if mapped is not None:
                                    mapped_ids = mapped if isinstance(mapped, (list, tuple)) else [int(mapped)]
                                    if any(int(p) in mapped_ids for p in pred_row):
                                        correct += 1
                                        continue
                                # Pattern-based template match
                                if '__patterns' in label_map:
                                    for pat, t_id, sth in label_map['__patterns']:
                                        try:
                                            if pat.search(str(l)):
                                                if any(int(p) == int(t_id) for p in pred_row):
                                                    correct += 1
                                                    mapped = t_id
                                                    break
                                        except Exception:
                                            continue
                                    if mapped is not None:
                                        continue

                            for p in pred_row:
                                p_label = model.config.id2label[int(p)]
                                if norm_label(p_label) == norm_label(l):
                                    matched = True
                                    break
                                # token overlap fuzzy match (use word tokens, ignore stopwords and the placeholder 'something')
                                import re
                                def _tokens(x: str):
                                    toks = set(re.findall(r"\w+", x.lower()))
                                    toks -= {"a", "the", "an", "something"}
                                    return toks
                                gt_tokens = _tokens(l)
                                p_tokens = _tokens(p_label)
                                if gt_tokens and p_tokens:
                                    overlap = len(gt_tokens & p_tokens) / max(1, min(len(gt_tokens), len(p_tokens)))
                                    if overlap >= fuzzy_threshold:
                                        matched = True
                                        break
                            if matched:
                                correct += 1
            # Clear GPU cache between coverage/stride combinations
            if device.type == "cuda":
                torch.cuda.empty_cache()

            total_time = (time.time() - t0)
            acc = (correct / total) if total > 0 else 0.0
            avg_time = total_time / total if total > 0 else 0.0
            results.append({
                "coverage": cov,
                "stride": stride,
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "avg_time": avg_time,
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
    label_map: dict = None,
    topk: int = 1,
    fuzzy_threshold: float = 0.5,
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

    # Warm up the model to avoid memory spike during first inference
    if rank == 0:
        print("Warming up model...")
    dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(num_frames)]
    with torch.amp.autocast(device.type, dtype=torch.float16):
        dummy_inputs = processor(dummy_frames, return_tensors="pt").to(device)
        _ = model(**dummy_inputs).logits
    del dummy_inputs, dummy_frames
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if rank == 0:
        print("Model warmed up.")

    id2label = model.config.id2label
    label2id = model.config.label2id
    n_classes = len(id2label)

    # Build textual -> id mapping when ground-truth labels are textual.
    txt_to_id = {}
    if label_map:
        for k, v in label_map.items():
            txt_to_id[k] = v if isinstance(v, (list, tuple)) else int(v)
    else:
        # Try to derive mapping from model's label2id (normalized keys)
        if label2id:
            for k, v in label2id.items():
                txt_to_id[norm_label(k)] = v

    def map_label(label):
        """Map a ground-truth label (int or string) to an integer class id, or return -1 if unmapped."""
        if label is None:
            return -1
        # If already an integer
        try:
            return int(label)
        except Exception:
            pass
        if isinstance(label, str):
            ln = norm_label(label)
            if label_map and ln in label_map:
                v = label_map.get(ln)
                if isinstance(v, (list, tuple)):
                    return int(v[0]) if v else -1
                return int(v)
            if ln in txt_to_id:
                v = txt_to_id.get(ln)
                return int(v) if not isinstance(v, (list, tuple)) else int(v[0])
            if label2id and ln in label2id:
                return int(label2id[ln])
        return -1

    for stride in strides:
        for cov in coverages:
            if (cov, stride) in completed:
                if rank == 0:
                    pass
                continue
            correct_per_class = np.zeros(n_classes, dtype=np.int32)
            total_per_class = np.zeros(n_classes, dtype=np.int32)
            t0 = time.time()

            # Process videos with controlled parallelism to balance speed and memory usage
            with ThreadPoolExecutor(max_workers=min(4, num_workers)) as ex:  # Limit workers to prevent memory explosion
                futures = [ex.submit(extract_and_prepare, row._asdict() if hasattr(row, "_asdict") else row.to_dict(), cov, stride, num_select=num_frames) for _, row in subset.iterrows()]

                batch_frames, batch_labels = [], []
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"stride={stride} cov={cov}%", disable=(rank != 0)):
                    frames, label = fut.result()
                    if frames is None:
                        continue
                    batch_frames.append(frames)
                    batch_labels.append(label)

                    if len(batch_frames) >= batch_size:  # Process when batch is full
                        with torch.amp.autocast(device.type, dtype=torch.float16):
                            inputs = processor(batch_frames, return_tensors="pt").to(device)
                            logits = model(**inputs).logits
                        preds = logits.argmax(-1).cpu().numpy()
                        for p, l in zip(preds, batch_labels):
                            # Support textual ground-truth labels via txt_to_id or label2id
                            true_id = None
                            if isinstance(l, str):
                                l_norm = norm_label(l)
                                if l_norm in txt_to_id:
                                    mapped = txt_to_id[l_norm]
                                    true_id = mapped if isinstance(mapped, int) else (mapped[0] if mapped else None)
                                elif l in label2id:
                                    true_id = label2id[l]
                                else:
                                    # Skip samples we cannot map
                                    continue
                            else:
                                true_id = int(l)

                            if true_id is None:
                                continue
                            total_per_class[true_id] += 1
                            if p == true_id:
                                correct_per_class[true_id] += 1
                        batch_frames, batch_labels = [], []  # Clear batch immediately
                        # Release tensor references to prevent memory leak
                        del inputs, logits, preds
                        if device.type == "cuda":
                            torch.cuda.empty_cache()  # Clear cache after each batch

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
            sd_acc = accs[total_per_class > 0].std() if (total_per_class > 0).any() else 0.0
            if rank == 0:
                print(f"[PER-CLASS RESULT] stride={stride} cov={cov}% -> mean_acc={mean_acc:.4f}, sd_acc={sd_acc:.4f}, total_samples={total_per_class.sum()}")

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
                            true_id = map_label(l)
                            if true_id == -1 or true_id is None:
                                continue
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
