# AdaptiveFlippedLasso Benchmark Report - Exp2 (4 Models Complete)

**Date**: 2026-03-27
**Experiment**: adaptive_flipped_lasso_exp2_benchmark
**Data Configuration**:
- Correlation: AR(1), rho=0.8
- n_samples: 300
- n_features: 500
- n_nonzero: 20
- sigma values: [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
- SNR values: [10.0, 2.0, 1.0, 0.67, 0.5, 0.33]
- Repeats: 5 per configuration
- Total trials: 120

**Models Status**:
| Model | Status |
|-------|--------|
| AdaptiveFlippedLasso | ✓ Complete |
| LassoCV | ✓ Complete |
| AdaptiveLassoCV | ✓ Complete |
| GroupLassoCV | ✓ Complete |
| UniLassoCV | Pending |
| FusedLassoCV | Pending (extremely slow) |

---

## 1. Overall Performance Summary

### 1.1 Mean ± Std across all SNR levels

| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| **AdaptiveFlippedLasso** | **0.886 ± 0.110** | 3.97 ± 3.86 | 0.970 ± 0.068 | **0.178 ± 0.147** | **0.823 ± 0.147** | 0.970 ± 0.068 | 0.953 ± 0.046 |
| LassoCV | 0.542 ± 0.028 | **3.61 ± 4.13** | 0.994 ± 0.018 | 0.624 ± 0.027 | 0.376 ± 0.027 | 0.994 ± 0.018 | **0.956 ± 0.049** |
| AdaptiveLassoCV | 0.474 ± 0.170 | 65.47 ± 3.86 | 0.978 ± 0.045 | 0.667 ± 0.150 | 0.333 ± 0.150 | 0.978 ± 0.045 | 0.183 ± 0.015 |
| GroupLassoCV | 0.124 ± 0.021 | 65.60 ± 3.95 | **0.994 ± 0.012** | 0.934 ± 0.012 | 0.066 ± 0.012 | **0.994 ± 0.012** | 0.182 ± 0.016 |

---

## 2. Rankings by Metric

| Rank | F1 ↑ | MSE ↓ | TPR ↑ | FDR ↓ | Precision ↑ | Recall ↑ | R2 ↑ |
|------|------|-------|-------|-------|-------------|----------|------|
| 1 | AFL (0.886) | LassoCV (3.61) | GroupLassoCV (0.994) | AFL (0.178) | AFL (0.823) | GroupLassoCV (0.994) | LassoCV (0.956) |
| 2 | LassoCV (0.542) | AFL (3.97) | LassoCV (0.994) | LassoCV (0.624) | LassoCV (0.376) | LassoCV (0.994) | AFL (0.953) |
| 3 | AdaptiveLassoCV (0.474) | AdaptiveLassoCV (65.47) | AdaptiveLassoCV (0.978) | AdaptiveLassoCV (0.667) | AdaptiveLassoCV (0.333) | AdaptiveLassoCV (0.978) | AdaptiveLassoCV (0.183) |
| 4 | GroupLassoCV (0.124) | GroupLassoCV (65.60) | AFL (0.970) | GroupLassoCV (0.934) | GroupLassoCV (0.066) | AFL (0.970) | GroupLassoCV (0.182) |

---

## 3. Performance Across SNR Levels

### 3.1 F1 Score by Sigma

| sigma | SNR | AdaptiveFlippedLasso | LassoCV | AdaptiveLassoCV | GroupLassoCV |
|-------|-----|---------------------|---------|-----------------|--------------|
| 0.1 | 10.00 | **0.9902** | 0.5625 | 0.5961 | 0.1273 |
| 0.5 | 2.00 | **0.9854** | 0.5649 | 0.7031 | 0.1189 |
| 1.0 | 1.00 | **0.9187** | 0.5286 | 0.5322 | 0.1177 |
| 1.5 | 0.67 | **0.8825** | 0.5323 | 0.4357 | 0.1239 |
| 2.0 | 0.50 | **0.8468** | 0.5392 | 0.3388 | 0.1287 |
| 3.0 | 0.33 | **0.6928** | 0.5231 | 0.2370 | 0.1244 |

### 3.2 MSE by Sigma

| sigma | SNR | AdaptiveFlippedLasso | LassoCV | AdaptiveLassoCV | GroupLassoCV |
|-------|-----|---------------------|---------|-----------------|--------------|
| 0.1 | 10.00 | 0.6448 | **0.0134** | 63.0694 | 63.0793 |
| 0.5 | 2.00 | 0.9293 | **0.3342** | 63.1367 | 63.1916 |
| 1.0 | 1.00 | 1.8064 | **1.3273** | 63.7051 | 63.8119 |
| 1.5 | 0.67 | 3.3821 | **2.9843** | 64.8418 | 65.0217 |
| 2.0 | 0.50 | 5.5380 | **5.2719** | 66.5431 | 66.7093 |
| 3.0 | 0.33 | 11.5317 | **11.7532** | 71.5347 | 71.7691 |

### 3.3 FDR by Sigma

| sigma | SNR | AdaptiveFlippedLasso | LassoCV | AdaptiveLassoCV | GroupLassoCV |
|-------|-----|---------------------|---------|-----------------|--------------|
| 0.1 | 10.00 | **0.0190** | 0.6037 | 0.5749 | 0.9319 |
| 0.5 | 2.00 | **0.0286** | 0.6006 | 0.4396 | 0.9366 |
| 1.0 | 1.00 | **0.1484** | 0.6371 | 0.6255 | 0.9374 |
| 1.5 | 0.67 | **0.2087** | 0.6332 | 0.7115 | 0.9337 |
| 2.0 | 0.50 | **0.2519** | 0.6291 | 0.7899 | 0.9307 |
| 3.0 | 0.33 | **0.4084** | 0.6389 | 0.8605 | 0.9328 |

### 3.4 Precision by Sigma

| sigma | SNR | AdaptiveFlippedLasso | LassoCV | AdaptiveLassoCV | GroupLassoCV |
|-------|-----|---------------------|---------|-----------------|--------------|
| 0.1 | 10.00 | **0.9810** | 0.3963 | 0.4251 | 0.0681 |
| 0.5 | 2.00 | **0.9714** | 0.3994 | 0.5604 | 0.0634 |
| 1.0 | 1.00 | **0.8516** | 0.3629 | 0.3745 | 0.0626 |
| 1.5 | 0.67 | **0.7913** | 0.3668 | 0.2885 | 0.0663 |
| 2.0 | 0.50 | **0.7481** | 0.3709 | 0.2101 | 0.0693 |
| 3.0 | 0.33 | **0.5916** | 0.3611 | 0.1395 | 0.0672 |

### 3.5 TPR by Sigma

| sigma | SNR | AdaptiveFlippedLasso | LassoCV | AdaptiveLassoCV | GroupLassoCV |
|-------|-----|---------------------|---------|-----------------|--------------|
| 0.1 | 10.00 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | 2.00 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1.0 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1.5 | 0.67 | 1.000 | 1.000 | 0.998 | 1.000 |
| 2.0 | 0.50 | 0.980 | 1.000 | 0.984 | 0.996 |
| 3.0 | 0.33 | 0.840 | 0.962 | 0.884 | 0.970 |

---

## 4. Detailed Metrics Tables

### 4.1 Per-Sigma, Per-Model Results (Mean ± Std)

#### sigma = 0.1 (SNR = 10.00)
| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.9902 | 0.6448 | 1.0000 | 0.0190 | 0.9810 | 1.0000 | 0.9924 |
| LassoCV | 0.5625 | 0.0134 | 1.0000 | 0.6037 | 0.3963 | 1.0000 | 0.9998 |
| AdaptiveLassoCV | 0.5961 | 63.0694 | 1.0000 | 0.5749 | 0.4251 | 1.0000 | 0.1945 |
| GroupLassoCV | 0.1273 | 63.0793 | 1.0000 | 0.9319 | 0.0681 | 1.0000 | 0.1944 |

#### sigma = 0.5 (SNR = 2.00)
| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.9854 | 0.9293 | 1.0000 | 0.0286 | 0.9714 | 1.0000 | 0.9887 |
| LassoCV | 0.5649 | 0.3342 | 1.0000 | 0.6006 | 0.3994 | 1.0000 | 0.9956 |
| AdaptiveLassoCV | 0.7031 | 63.1367 | 1.0000 | 0.4396 | 0.5604 | 1.0000 | 0.1932 |
| GroupLassoCV | 0.1189 | 63.1916 | 1.0000 | 0.9366 | 0.0634 | 1.0000 | 0.1925 |

#### sigma = 1.0 (SNR = 1.00)
| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.9187 | 1.8064 | 1.0000 | 0.1484 | 0.8516 | 1.0000 | 0.9778 |
| LassoCV | 0.5286 | 1.3273 | 1.0000 | 0.6371 | 0.3629 | 1.0000 | 0.9828 |
| AdaptiveLassoCV | 0.5322 | 63.7051 | 1.0000 | 0.6255 | 0.3745 | 1.0000 | 0.1900 |
| GroupLassoCV | 0.1177 | 63.8119 | 1.0000 | 0.9374 | 0.0626 | 1.0000 | 0.1886 |

#### sigma = 1.5 (SNR = 0.67)
| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.8825 | 3.3821 | 1.0000 | 0.2087 | 0.7913 | 1.0000 | 0.9589 |
| LassoCV | 0.5323 | 2.9843 | 1.0000 | 0.6332 | 0.3668 | 1.0000 | 0.9619 |
| AdaptiveLassoCV | 0.4357 | 64.8418 | 0.9980 | 0.7115 | 0.2885 | 0.9980 | 0.1846 |
| GroupLassoCV | 0.1239 | 65.0217 | 1.0000 | 0.9337 | 0.0663 | 1.0000 | 0.1822 |

#### sigma = 2.0 (SNR = 0.50)
| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.8468 | 5.5380 | 0.9800 | 0.2519 | 0.7481 | 0.9800 | 0.9336 |
| LassoCV | 0.5392 | 5.2719 | 1.0000 | 0.6291 | 0.3709 | 1.0000 | 0.9339 |
| AdaptiveLassoCV | 0.3388 | 66.5431 | 0.9840 | 0.7899 | 0.2101 | 0.9840 | 0.1773 |
| GroupLassoCV | 0.1287 | 66.7093 | 0.9960 | 0.9307 | 0.0693 | 0.9960 | 0.1751 |

#### sigma = 3.0 (SNR = 0.33)
| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.6928 | 11.5317 | 0.8400 | 0.4084 | 0.5916 | 0.8400 | 0.8676 |
| LassoCV | 0.5231 | 11.7532 | 0.9620 | 0.6389 | 0.3611 | 0.9620 | 0.8601 |
| AdaptiveLassoCV | 0.2370 | 71.5347 | 0.8840 | 0.8605 | 0.1395 | 0.8840 | 0.1589 |
| GroupLassoCV | 0.1244 | 71.7691 | 0.9700 | 0.9328 | 0.0672 | 0.9700 | 0.1560 |

---

## 5. Key Findings

### 5.1 Feature Selection Quality
**AdaptiveFlippedLasso is the clear winner for sparse feature selection**:
- F1 Score: 0.886 (63% better than LassoCV's 0.542)
- FDR: 0.178 (72% lower than LassoCV's 0.624)
- Precision: 0.823 (119% higher than LassoCV's 0.376)

### 5.2 Prediction Quality
**LassoCV is slightly better for pure prediction**:
- MSE: 3.61 vs 3.97 (9% better)
- R2: 0.956 vs 0.953 (negligible difference)

### 5.3 SNR Robustness
| Model | F1 Drop (σ=0.1 → σ=3.0) | Robustness |
|-------|--------------------------|------------|
| AdaptiveFlippedLasso | 0.990 → 0.693 = **0.297** | ★★★☆☆ |
| LassoCV | 0.563 → 0.523 = 0.040 | ★★★★☆ |
| AdaptiveLassoCV | 0.596 → 0.237 = **0.359** | ★★☆☆☆ |
| GroupLassoCV | 0.127 → 0.124 = 0.003 | ★★★★★ |

**Note**: AdaptiveFlippedLasso shows larger F1 degradation at low SNR, but maintains much higher absolute F1 throughout.

### 5.4 Critical Observations

1. **GroupLassoCV's False Selection Problem**: TPR≈1.0 but FDR≈0.93, meaning it selects almost all features including noise
2. **AdaptiveLassoCV's Instability**: High variance (F1 std=0.170) and poor low-SNR performance
3. **LassoCV's Conservative Selection**: Very high TPR but poor precision, suggesting it under-regularizes
4. **AdaptiveFlippedLasso's Balance**: Best combination of TPR (0.97) and FDR (0.18)

### 5.5 Trade-off Summary

```
                    Feature Selection    Prediction    Overall
                    (F1/FDR/Prec)        (MSE/R2)     Recommendation
---------------------------------------------------------------------------
AdaptiveFlippedLasso    ★★★★★              ★★★★☆      RECOMMENDED
LassoCV                 ★★☆☆☆              ★★★★★      For prediction only
AdaptiveLassoCV         ★★☆☆☆              ★☆☆☆☆      Not recommended
GroupLassoCV            ★☆☆☆☆              ★☆☆☆☆      Avoid
```

---

## 6. Conclusions

1. **Primary Recommendation**: Use **AdaptiveFlippedLasso** for sparse feature selection tasks
   - Best-in-class F1 (0.886) and FDR (0.178)
   - Stage1 optimal parameters outperform CV-tuned alternatives

2. **When to use LassoCV**: Only when prediction accuracy (MSE/R2) is the sole metric

3. **Avoid**: GroupLassoCV and AdaptiveLassoCV for sparse true model scenarios

4. **Stage1 Parameters Validated**: The optimal parameters (λ_ridge=1.0, λ=1.0, γ=0.5) from grid search are confirmed effective

---

## Appendix: Pending Models

| Model | Status | Issue |
|-------|--------|-------|
| UniLassoCV | Did not complete | cv_unilasso timeout |
| FusedLassoCV | Running >45min, no results | cvxpy optimization extremely slow |

*Report generated: 2026-03-27T07:40:00*
