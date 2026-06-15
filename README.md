# InfoRates: Spatiotemporal Aliasing in Video Action Recognition

**ACCV 2026** | Mi3 Lab, UC Merced

[Dashboard](https://mi3-inforates.streamlit.app/) | [Contributing](CONTRIBUTING_NEW_DATASETS.md)

## Summary

Comprehensive empirical study of temporal aliasing across 8 architectures (CNNs, Transformers, SSM) and 8 datasets, with 8,000+ evaluation configurations spanning 5 resolutions, 5 coverages, and 5 strides. Key finding: **coverage dominates** (F=178.94), followed by stride (F=80.76), with architectural differences driven by attention mechanism type, not family.

---

## Key Findings

### 1. Coverage Dominates (F=178.94)

Coverage effect on accuracy (ANOVA):
- **10% → 100%**: +31.9pp (linear, all datasets)
- **Importance**: 13.6× more than resolution

### 2. Stride Critical (F=80.76)

Temporal degradation (stride 1→16, cov=100%):
- **VideoMamba, TimeSformer**: 6–8pp drop (robust)
- **ViViT, SlowFast**: 34–42pp drop (vulnerable)
- **Implication**: Attention mechanism type governs aliasing, not architecture family

### 3. Resolution Secondary (F=13.16)

Spatial robustness (48–224px, no retraining):
- **CNNs**: -29 to -37pp (resolution-dependent)
- **Transformers/SSM**: -2 to -6pp (stable across resolution)

### 4. Temporal Demand Score (TDS)

Dataset ranking (consistent across all 8 models):

| Dataset | TDS | Temporal Type |
|---------|----:|---------------|
| AUTSL | 42.3pp | High (gesture) |
| SSv2 | 38.7pp | High (causal) |
| Diving-48 | 31.5pp | High (fine-grained) |
| FineGym | 22.1pp | Moderate |
| HMDB-51 | 21.6pp | Low-Moderate |
| EPIC-Kitchens | 24.8pp | Moderate |
| DriveAct | 19.3pp | Low |
| UCF-101 | 14.7pp | Low (appearance) |

## Evaluation Protocol

**Models** (8 architectures):
- CNNs: R3D-18, MC3-18, R2+1D
- Dual-stream: SlowFast-R50
- Transformers: TimeSformer, ViViT, VideoMAE
- SSM: VideoMamba

**Datasets** (8, 20 clips/class validation split):
- AUTSL, SSv2, Diving-48, EPIC-Kitchens, HMDB-51, DriveAct, UCF-101, FineGym

**Sweep**: 5 resolutions × 5 coverages × 5 strides = **8,000+ configurations**

**Methodology**:
1. P3-retrained checkpoints at each resolution (10 epochs)
2. Bicubic interpolation for positional embeddings (Transformers)
3. Top-1 accuracy evaluation at (coverage, stride, resolution) combinations

---

## Access & Contribute

**Interactive Dashboard**: https://mi3-inforates.streamlit.app/

**For new datasets**: See [CONTRIBUTING_NEW_DATASETS.md](CONTRIBUTING_NEW_DATASETS.md)
- Prepare manifest (20 samples/class)
- Train P3-retrained checkpoints
- Run coverage × stride × resolution sweep (~4–8 hours per dataset, all 8 models)
- Integration script to update dashboard

---

## References

```bibtex
@inproceedings{maia2026temporal,
  title={Temporal Aliasing in Video Action Recognition: A Cross-Architecture Analysis at Scale},
  author={Maia, Wesley and others},
  booktitle={ACCV},
  year={2026}
}
```

Related work:
- VideoMAE (Tong et al., 2022)
- VideoMamba (Li et al., 2024)
- ViViT (Arnab et al., 2021)

---

## License

MIT License
