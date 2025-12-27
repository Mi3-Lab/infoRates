#!/usr/bin/env python3
import os, sys
# Project roots
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
for _p in (_PROJECT_ROOT, _SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForVideoClassification, AutoConfig
from scripts.dataset_handler import DatasetHandler
from info_rates.analysis.evaluate_fixed import evaluate_fixed_parallel


def load_hmdb_model(device='cpu'):
    base = "facebook/timesformer-base-finetuned-k400"
    num_labels = 51
    proc = AutoImageProcessor.from_pretrained(base, use_fast=False)
    cfg = AutoConfig.from_pretrained(base)
    cfg.num_labels = num_labels
    cfg.id2label = {i: str(i) for i in range(num_labels)}
    cfg.label2id = {str(i): i for i in range(num_labels)}
    model = AutoModelForVideoClassification.from_pretrained(base, config=cfg, ignore_mismatched_sizes=True).to(device).eval()
    return proc, model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    proc, model = load_hmdb_model(device=device)
    print('Model num_labels:', getattr(model.config, 'num_labels', None))
    print('id2label len:', len(getattr(model.config, 'id2label', {})))

    # Load HMDB manifest and small subset
    df, manifest = DatasetHandler.load_or_build_manifest('hmdb51')
    print('Manifest loaded:', manifest, 'samples:', len(df))
    subset = df.sample(min(20, len(df)), random_state=42)

    # Run evaluate_fixed_parallel on this small subset for 100% coverage, stride 1
    res = evaluate_fixed_parallel(
        df=subset,
        processor=proc,
        model=model,
        coverages=[100],
        strides=[1],
        sample_size=len(subset),
        batch_size=4,
        num_workers=2,
        jitter_coverage_pct=0.0,
    )
    print('Result DataFrame:')
    print(res)
