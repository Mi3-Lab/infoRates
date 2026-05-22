# Data Anomaly Analysis: Duplicate Values in Baseline vs TRA Results

**Analysis Date**: March 4, 2026  
**Concern Raised By**: Reviewer Comment  
**Status**: ⚠️ REQUIRES INVESTIGATION

---

## Summary

The baseline vs TRA comparison table contains **multiple instances of identical accuracy values across different coverage levels**. This is statistically improbable and suggests potential experimental errors.

---

## Detailed Findings

### 1. Primary Issue: Coverage 25% = Coverage 50% (Stride 16)

| Metric | Coverage 25% | Coverage 50% | Match? |
|--------|---|---|---|
| Baseline | 0.7160 | 0.7160 | ✓ **IDENTICAL** |
| TRA | 0.8238 | 0.8238 | ✓ **IDENTICAL** |
| Improvement | +15.0% | +15.0% | ✓ **IDENTICAL** |

**Physical Plausibility**: When frame coverage decreases from 50% to 25%, we expect:
- Fewer frames available → harder recognition task
- Likely performance degradation or at least variance
- Identical results suggests **no actual difference in evaluation**

---

### 2. Secondary Issues: Pattern of Repetitions

#### Stride 16 Repetitions
```
Coverage 25%, Stride 16:  0.7160 → 0.8238
Coverage 50%, Stride 16:  0.7160 → 0.8238  ← IDENTICAL
Coverage 75%, Stride 16:  0.8377 → 0.8489
Coverage 100%, Stride 16: 0.8377 → 0.8489  ← IDENTICAL
```

#### Stride 8 Repetitions
```
Coverage 25%, Stride 8:   0.7160 → 0.8238
Coverage 50%, Stride 8:   0.8296 → 0.8459  ← Different (OK)
Coverage 75%, Stride 8:   0.8537 → 0.8536
Coverage 100%, Stride 8:  0.8537 → 0.8536  ← IDENTICAL
```

#### Summary Table: Value Frequency
| Baseline Value | Count | Coverages |
|---|---|---|
| 0.7160 | 3 | 25%-S8, 25%-S16, 50%-S16 |
| 0.8377 | 2 | 75%-S16, 100%-S16 |
| 0.8537 | 2 | 75%-S8, 100%-S8 |

| TRA Value | Count | Coverages |
|---|---|---|
| 0.8238 | 3 | 25%-S8, 25%-S16, 50%-S16 |
| 0.8489 | 2 | 75%-S16, 100%-S16 |
| 0.8536 | 2 | 75%-S8, 100%-S8 |

---

## Possible Root Causes

### Hypothesis 1: Data Duplication Error (MOST LIKELY)
**Scenario**: Values were copy-pasted instead of evaluating separate experiments
- Coverage 25% Stride 16 results copied to Coverage 50% Stride 16
- Coverage 75% Stride 16 results copied to Coverage 100% Stride 16
- **Evidence**: Exact matches with 4+ decimal places
- **Impact**: Results are unreliable; need re-evaluation

### Hypothesis 2: Single Model Evaluation Across Coverages
**Scenario**: Same trained model evaluated at different coverages without retraining
- Models trained on dense data might collapse identically at aggressive undersampling
- **Evidence**: Stride 8 & 16 show this pattern more than lower strides
- **Impact**: Experiment design flaw; TRA training may not be coverage-specific

### Hypothesis 3: Insufficient Evaluation Set Size
**Scenario**: Test set too small; random variation produces coincidental ties
- Very small eval set (<20 samples) might repeat values
- **Evidence**: 3 exact matches is unlikely but possible with n<20
- **Impact**: Need larger eval set for reliable metrics

### Hypothesis 4: Bug in Evaluation Pipeline
**Scenario**: Same checkpoint or data split used across multiple coverage configurations
- Cache miss or hardcoded path affecting multiple runs
- **Evidence**: Systematic pattern in stride dimensions
- **Impact**: Results cannot be trusted; requires pipeline audit

---

## Verification Checklist

- [ ] **Check experiment logs**: Verify that separate training/evaluation runs occurred for each (coverage, stride) pair
- [ ] **Inspect checkpoints**: Confirm different model files for different coverage groups
- [ ] **Audit eval sets**: Validate that different frame samples were used for 25% vs 50% coverage
- [ ] **Visual inspection**: Plot actual frame sequences to confirm different sampling
- [ ] **Re-run critical cases**: Re-evaluate Coverage 25% Stride 16 and Coverage 50% Stride 16 independently
- [ ] **Set size**: Report evaluation set size; if < 50 samples, explain why coincidences are possible

---

## Impact on Paper Claims

### Currently Claimed:
> "TRA significantly flattens the collapse region observed in baseline training under aggressive undersampling. The largest gains occur at 25% coverage with high strides (s ∈ {8, 16}), where TRA improves accuracy from 71.60% to 82.38%—a 15.04% relative improvement."

### Concern:
If values are duplicated, the claim about "25% coverage" is based on unreliable data. The identical 50% coverage results suggest either:
1. No actual difference in performance between 25% and 50% coverage (surprising)
2. Experimental error (more likely)

### Recommended Revision:
**Pending Resolution**
```
"TRA significantly flattens the collapse region observed in baseline training under 
aggressive undersampling. While preliminary results show the largest gains at 25% coverage 
with high strides (s ∈ {8, 16}), reaching 15.04% relative improvement, we note that 
[RESOLUTION STATEMENT PENDING VERIFICATION] ..."
```

---

## Recommendations

### Immediate Actions
1. **Audit experiment logs** for Coverage 25% Stride 16 and 50% Stride 16
2. **Report evaluation set size** (number of test samples)
3. **Verify data split reproducibility** across configurations

### If Data Error Confirmed
1. Re-run missing experiments
2. Update results table with correct values
3. Revise paper claims to reflect corrected data
4. Add note in methods explaining the issue and resolution

### If Data Verified as Correct
1. Provide statistical justification (e.g., small eval set leading to ties)
2. Explain why coverage differences don't affect performance from principled perspective
3. Add ablation study: vary eval set size to show result stability

---

## Status Tracking
- **Issue Identified**: 2026-03-04
- **Analysis Complete**: 2026-03-04
- **Investigation Status**: 🔴 PENDING
- **Resolution Deadline**: [TBD with advisor]
