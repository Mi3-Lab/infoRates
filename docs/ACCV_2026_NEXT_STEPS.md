# ACCV 2026 Next Steps

Last updated: 2026-05-22.

## Current State

The project is no longer just a temporal sampling benchmark. The ACCV direction is now:

> Measure temporal evidence demand, show that current action models are poorly calibrated under evidence budgets, and test whether simple adaptive allocation can improve the accuracy/compute trade-off.

The repository now has a working ACCV experiment track:

- SSV2 manifests are available and usable.
- Diving48 manifests are available with an annotation-source caveat.
- Fixed-budget evaluation writes per-sample CSVs, summary CSVs, latency columns, `processed_frames`, and `model_input_frames`.
- Transformer pilots are complete for TimeSformer, VideoMAE, and ViViT.
- Non-transformer TorchVision pilots are complete for R3D-18, MC3-18, and R(2+1)D-18.
- TRA is no longer the main contribution. It should be treated as a baseline or historical artifact unless revalidated cleanly.
- UCF101 and Kinetics should be supporting datasets, not the central evidence.

The strongest pilot signal so far is R(2+1)D-18 on SSV2 5k/1 epoch:

| Model | Budget Grid | Top-1 Trend | Temporal Robustness AUC | Critical Frame Budget |
|---|---:|---|---:|---:|
| TimeSformer | `4,8,16,32` | 2.44%, 2.44%, 3.41%, 3.29% | available | available |
| VideoMAE | `4,8,16` | 3.05%, 5.37%, 7.44% | available | available |
| ViViT | `4,8,16,32` | 1.95%, 1.95%, 2.20%, 2.68% | 0.022648 | 32 |
| R3D-18 | `4,8,16` | 1.34%, 1.83%, 3.54% | 0.023171 | 16 |
| MC3-18 | `4,8,16` | 1.34%, 2.68%, 2.80% | 0.025000 | 8 |
| R(2+1)D-18 | `4,8,16` | 2.07%, 3.29%, 4.39% | 0.034553 | 16 |

These are pilot numbers only. They validate the pipeline and the evidence-budget trend; they are not paper-ready performance claims.

## Immediate Priority

The next milestone is to move from "pipeline works" to "scientific claim is measurable."

Do the next steps in this order:

1. **Freeze the fixed-budget protocol**
   - Keep reporting `processed_frames` and `model_input_frames`.
   - Keep model-valid budget grids.
   - Keep latency columns, but do not overclaim efficiency until timing is audited.
   - Add a short protocol note to the paper draft explaining evidence budget vs model input length.

2. **Build the temporal-demand estimator**
   - Start with frame-difference energy because it is cheap and reproducible.
   - Write one per-video CSV with demand score, class, source frames, and split.
   - Correlate demand score with observed critical budget from the fixed-budget summaries.

3. **Create the adaptive budget baseline**
   - Use the demand score to allocate budgets under a fixed average budget.
   - Compare against fixed 4/8/16-frame baselines and random allocation with the same average budget.
   - First target: SSV2 validation 5-per-class or 20-per-class manifest.
   - Only scale to larger validation once the adaptive table is sane.

4. **Add stronger non-transformer references**
   - TorchVision 3D CNNs answer the transformer-only criticism, but SlowFast/X3D are stronger modern baselines.
   - Add SlowFast R50 first.
   - Add X3D-S or X3D-M second if the adapter is stable.

5. **Run paper-usable training**
   - Use larger SSV2 training subsets or full SSV2.
   - Prefer 2-3 epochs first, then decide whether longer training is worth the GPU time.
   - Evaluate on `somethingv2_val_20_per_class.csv` or a larger reproducible validation subset.

## Do Not Do Yet

- Do not write final paper tables from the 1-epoch pilot results.
- Do not center the method on TRA.
- Do not return to a Nyquist-heavy theory story.
- Do not make Kinetics/Kaggle protocol central.
- Do not launch expensive full runs before the temporal-demand/adaptive baseline code exists.

## Next Deliverable

The next concrete deliverable should be:

```text
evaluations/accv2026/metrics/temporal_demand_scores_ssv2_*.csv
evaluations/accv2026/metrics/demand_vs_critical_budget_*.csv
evaluations/accv2026/metrics/adaptive_budget_baseline_*.csv
```

Once those exist, the paper can claim more than "we benchmarked temporal sampling." It can claim that temporal evidence demand is measurable and exploitable.
