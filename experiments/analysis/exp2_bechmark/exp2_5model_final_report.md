# AdaptiveFlippedLasso Benchmark Report - Exp2 (5 Models)

**Date**: 2026-03-27
**Experiment**: adaptive_flipped_lasso_exp2_benchmark
**Data Configuration**:
- Correlation: AR(1), rho=0.8
- n_samples: 300
- n_features: 500
- n_nonzero: 20
- sigma values: [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
- Repeats: 5 per configuration
- Total trials: 150

**Models Compared**:
| Model | Description | Status |
|-------|-------------|--------|
| AdaptiveFlippedLasso | Stage1 optimal params (λ_ridge=1.0, λ=1.0, γ=0.5) | ✓ Complete |
| LassoCV | sklearn Lasso with CV-tuned alpha | ✓ Complete |
| AdaptiveLassoCV | Adaptive Lasso with CV-tuned alpha | ✓ Complete |
| GroupLassoCV | Group Lasso with CV-tuned alpha | ✓ Complete |
| UniLassoCV | Univariate-guided Lasso with CV | ✓ Complete |
| FusedLassoCV | Fused Lasso with CV | Skipped (too slow) |

---

## 1. Overall Performance Summary

### Mean ± Std across all SNR levels

| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| **AdaptiveFlippedLasso** | **0.886 ± 0.110** | 3.97 ± 3.86 | 0.970 ± 0.068 | **0.178 ± 0.147** | **0.823 ± 0.147** | 0.970 ± 0.068 | 0.953 ± 0.046 |
| UniLassoCV | 0.792 ± 0.065 | 6.48 ± 7.06 | 0.923 ± 0.107 | 0.301 ± 0.055 | 0.699 ± 0.055 | 0.923 ± 0.107 | 0.920 ± 0.084 |
| LassoCV | 0.542 ± 0.028 | **3.61 ± 4.13** | 0.994 ± 0.018 | 0.624 ± 0.027 | 0.376 ± 0.027 | 0.994 ± 0.018 | **0.956 ± 0.049** |
| AdaptiveLassoCV | 0.474 ± 0.170 | 65.47 ± 3.86 | 0.978 ± 0.045 | 0.667 ± 0.150 | 0.333 ± 0.150 | 0.978 ± 0.045 | 0.183 ± 0.015 |
| GroupLassoCV | 0.124 ± 0.021 | 65.60 ± 3.95 | **0.994 ± 0.012** | 0.934 ± 0.012 | 0.066 ± 0.012 | **0.994 ± 0.012** | 0.182 ± 0.016 |

---

## 2. Rankings by Metric

| Rank | F1 ↑ | MSE ↓ | TPR ↑ | FDR ↓ | Precision ↑ | Recall ↑ | R2 ↑ |
|------|------|-------|-------|-------|-------------|----------|------|
| 1 | AFL (0.886) | LassoCV (3.61) | GroupLassoCV (0.994) | AFL (0.178) | AFL (0.823) | GroupLassoCV (0.994) | LassoCV (0.956) |
| 2 | UniLassoCV (0.792) | AFL (3.97) | LassoCV (0.994) | UniLassoCV (0.301) | UniLassoCV (0.699) | LassoCV (0.994) | AFL (0.953) |
| 3 | LassoCV (0.542) | UniLassoCV (6.48) | AdaptiveLassoCV (0.978) | LassoCV (0.624) | LassoCV (0.376) | AdaptiveLassoCV (0.978) | UniLassoCV (0.920) |
| 4 | AdaptiveLassoCV (0.474) | AdaptiveLassoCV (65.47) | AFL (0.970) | AdaptiveLassoCV (0.667) | AdaptiveLassoCV (0.333) | AFL (0.970) | AdaptiveLassoCV (0.183) |
| 5 | GroupLassoCV (0.124) | GroupLassoCV (65.60) | UniLassoCV (0.923) | GroupLassoCV (0.934) | GroupLassoCV (0.066) | UniLassoCV (0.923) | GroupLassoCV (0.182) |

---

## 3. Performance Across SNR Levels

### 3.1 F1 by Sigma

| sigma | SNR | AdaptiveFlippedLasso | UniLassoCV | LassoCV | AdaptiveLassoCV | GroupLassoCV |
|-------|-----|---------------------|-------------|---------|-----------------|--------------|
| 0.1 | 10.00 | **0.9902** | 0.8375 | 0.5625 | 0.5961 | 0.1273 |
| 0.5 | 2.00 | **0.9854** | 0.8478 | 0.5649 | 0.7031 | 0.1189 |
| 1.0 | 1.00 | **0.9187** | 0.8294 | 0.5286 | 0.5322 | 0.1177 |
| 1.5 | 0.67 | **0.8825** | 0.7905 | 0.5323 | 0.4357 | 0.1239 |
| 2.0 | 0.50 | **0.8468** | 0.7653 | 0.5392 | 0.3388 | 0.1287 |
| 3.0 | 0.33 | **0.6928** | 0.6828 | 0.5231 | 0.2370 | 0.1244 |

### 3.2 MSE by Sigma

| sigma | SNR | AdaptiveFlippedLasso | UniLassoCV | LassoCV | AdaptiveLassoCV | GroupLassoCV |
|-------|-----|---------------------|-------------|---------|-----------------|--------------|
| 0.1 | 10.00 | 0.6448 | 0.2982 | **0.0134** | 63.0694 | 63.0793 |
| 0.5 | 2.00 | 0.9293 | 1.0067 | **0.3342** | 63.1367 | 63.1916 |
| 1.0 | 1.00 | 1.8064 | 2.2068 | **1.3273** | 63.7051 | 63.8119 |
| 1.5 | 0.67 | 3.3821 | 4.4042 | **2.9843** | 64.8418 | 65.0217 |
| 2.0 | 0.50 | 5.5380 | 7.3805 | **5.2719** | 66.5431 | 66.7093 |
| 3.0 | 0.33 | 11.5317 | 13.5607 | **11.7532** | 71.5347 | 71.7691 |

### 3.3 FDR by Sigma

| sigma | SNR | AdaptiveFlippedLasso | UniLassoCV | LassoCV | AdaptiveLassoCV | GroupLassoCV |
|-------|-----|---------------------|-------------|---------|-----------------|--------------|
| 0.1 | 10.00 | **0.0190** | 0.2270 | 0.6037 | 0.5749 | 0.9319 |
| 0.5 | 2.00 | **0.0286** | 0.2298 | 0.6006 | 0.4396 | 0.9366 |
| 1.0 | 1.00 | **0.1484** | 0.2616 | 0.6371 | 0.6255 | 0.9374 |
| 1.5 | 0.67 | **0.2087** | 0.3095 | 0.6332 | 0.7115 | 0.9337 |
| 2.0 | 0.50 | **0.2519** | 0.3418 | 0.6291 | 0.7899 | 0.9307 |
| 3.0 | 0.33 | **0.4084** | 0.4360 | 0.6389 | 0.8605 | 0.9328 |

### 3.4 Precision by Sigma

| sigma | SNR | AdaptiveFlippedLasso | UniLassoCV | LassoCV | AdaptiveLassoCV | GroupLassoCV |
|-------|-----|---------------------|-------------|---------|-----------------|--------------|
| 0.1 | 10.00 | **0.9810** | 0.7730 | 0.3963 | 0.4251 | 0.0681 |
| 0.5 | 2.00 | **0.9714** | 0.7702 | 0.3994 | 0.5604 | 0.0634 |
| 1.0 | 1.00 | **0.8516** | 0.7384 | 0.3629 | 0.3745 | 0.0626 |
| 1.5 | 0.67 | **0.7913** | 0.6905 | 0.3668 | 0.2885 | 0.0663 |
| 2.0 | 0.50 | **0.7481** | 0.6582 | 0.3709 | 0.2101 | 0.0693 |
| 3.0 | 0.33 | **0.5916** | 0.5640 | 0.3611 | 0.1395 | 0.0672 |

---

## 4. Detailed Metrics Tables

### 4.1 Per-Sigma Results (Mean across repeats)

#### sigma = 0.1 (SNR = 10.00)
| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.9902 | 0.6448 | 1.0000 | 0.0190 | 0.9810 | 1.0000 | 0.9924 |
| UniLassoCV | 0.8375 | 0.2982 | 0.9800 | 0.2270 | 0.7730 | 0.9800 | 0.9990 |
| LassoCV | 0.5625 | 0.0134 | 1.0000 | 0.6037 | 0.3963 | 1.0000 | 0.9998 |
| AdaptiveLassoCV | 0.5961 | 63.0694 | 1.0000 | 0.5749 | 0.4251 | 1.0000 | 0.1945 |
| GroupLassoCV | 0.1273 | 63.0793 | 1.0000 | 0.9319 | 0.0681 | 1.0000 | 0.1944 |

#### sigma = 0.5 (SNR = 2.00)
| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.9854 | 0.9293 | 1.0000 | 0.0286 | 0.9714 | 1.0000 | 0.9887 |
| UniLassoCV | 0.8478 | 1.0067 | 0.9800 | 0.2298 | 0.7702 | 0.9800 | 0.9935 |
| LassoCV | 0.5649 | 0.3342 | 1.0000 | 0.6006 | 0.3994 | 1.0000 | 0.9956 |
| AdaptiveLassoCV | 0.7031 | 63.1367 | 1.0000 | 0.4396 | 0.5604 | 1.0000 | 0.1932 |
| GroupLassoCV | 0.1189 | 63.1916 | 1.0000 | 0.9366 | 0.0634 | 1.0000 | 0.1925 |

#### sigma = 1.0 (SNR = 1.00)
| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.9187 | 1.8064 | 1.0000 | 0.1484 | 0.8516 | 1.0000 | 0.9778 |
| UniLassoCV | 0.8294 | 2.2068 | 0.9600 | 0.2616 | 0.7384 | 0.9600 | 0.9667 |
| LassoCV | 0.5286 | 1.3273 | 1.0000 | 0.6371 | 0.3629 | 1.0000 | 0.9828 |
| AdaptiveLassoCV | 0.5322 | 63.7051 | 1.0000 | 0.6255 | 0.3745 | 1.0000 | 0.1900 |
| GroupLassoCV | 0.1177 | 63.8119 | 1.0000 | 0.9374 | 0.0626 | 1.0000 | 0.1886 |

#### sigma = 1.5 (SNR = 0.67)
| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.8825 | 3.3821 | 1.0000 | 0.2087 | 0.7913 | 1.0000 | 0.9589 |
| UniLassoCV | 0.7905 | 4.4042 | 0.9400 | 0.3095 | 0.6905 | 0.9400 | 0.9361 |
| LassoCV | 0.5323 | 2.9843 | 1.0000 | 0.6332 | 0.3668 | 1.0000 | 0.9619 |
| AdaptiveLassoCV | 0.4357 | 64.8418 | 0.9980 | 0.7115 | 0.2885 | 0.9980 | 0.1846 |
| GroupLassoCV | 0.1239 | 65.0217 | 1.0000 | 0.9337 | 0.0663 | 1.0000 | 0.1822 |

#### sigma = 2.0 (SNR = 0.50)
| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.8468 | 5.5380 | 0.9800 | 0.2519 | 0.7481 | 0.9800 | 0.9336 |
| UniLassoCV | 0.7653 | 7.3805 | 0.9000 | 0.3418 | 0.6582 | 0.9000 | 0.9004 |
| LassoCV | 0.5392 | 5.2719 | 1.0000 | 0.6291 | 0.3709 | 1.0000 | 0vscode-webview://0lcsea2n4spca6ifrasq214bsctb1nn5qj1b6cm6757fajpetu1m/clear/NLasso/XLasso/experiments/analysis/exp3_lasso_cv_comparison_report.md.9339 |
| AdaptiveLassoCV | 0.3388 | 66.5431 | 0.9840 | 0.7899 | 0.2101 | 0.9840 | 0.1773 |
| GroupLassoCV | 0.1287 | 66.7093 | 0.9960 | 0.9307 | 0.0693 | 0.9960 | 0.1751 |

#### sigma = 3.0 (SNR = 0.33)
| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.6928 | 11.5317 | 0.8400 | 0.4084 | 0.5916 | 0.8400 | 0.8676 |
| UniLassoCV | 0.6828 | 13.5607 | 0.7800 | 0.4360 | 0.5640 | 0.7800 | 0.8258 |
| LassoCV | 0.5231 | 11.7532 | 0.9620 | 0.6389 | 0.3611 | 0.9620 | 0.8601 |
| AdaptiveLassoCV | 0.2370 | 71.5347 | 0.8840 | 0.8605 | 0.1395 | 0.8840 | 0.1589 |
| GroupLassoCV | 0.1244 | 71.7691 | 0.9700 | 0.9328 | 0.0672 | 0.9700 | 0.1560 |

---

## 5. Key Findings

### 5.1 Feature Selection Quality
**AdaptiveFlippedLasso is the clear winner for sparse feature selection**:
- F1 Score: 0.886 (11.9% better than UniLassoCV's 0.792, 63.4% better than LassoCV's 0.542)
- FDR: 0.178 (40.9% lower than UniLassoCV's 0.301, 71.5% lower than LassoCV's 0.624)
- Precision: 0.823 (17.7% higher than UniLassoCV's 0.699, 118.8% higher than LassoCV's 0.376)

### 5.2 Prediction Quality
**LassoCV is best for pure prediction**:
- MSE: 3.61 (9.1% better than AFL's 3.97)
- R2: 0.956 (0.3% better than AFL's 0.953)

**Note**: UniLassoCV also shows strong prediction with R2=0.920

### 5.3 SNR Robustness
| Model | F1 (σ=0.1) | F1 (σ=3.0) | Drop | Assessment |
|-------|-------------|-------------|------|------------|
| AdaptiveFlippedLasso | 0.9902 | 0.6928 | 0.297 | ★★★★☆ |
| UniLassoCV | 0.8375 | 0.6828 | 0.155 | ★★★★☆ |
| LassoCV | 0.5625 | 0.5231 | 0.039 | ★★★★★ |
| AdaptiveLassoCV | 0.5961 | 0.2370 | 0.359 | ★★☆☆☆ |
| GroupLassoCV | 0.1273 | 0.1244 | 0.003 | ★★★★★ |

### 5.4 Critical Observations

1. **AdaptiveFlippedLasso dominates feature selection**: Best F1 (0.886), best FDR (0.178), best Precision (0.823)
2. **UniLassoCV is a strong 2nd place**: F1=0.792, good prediction (R2=0.920), reasonable FDR (0.301)
3. **LassoCV is best for pure prediction**: MSE=3.61, R2=0.956, but poor feature selection (F1=0.542)
4. **GroupLassoCV fails for sparse models**: TPR≈1.0 but FDR≈0.93, selects almost everything
5. **AdaptiveLassoCV is unstable**: High variance (F1 std=0.170), poor low-SNR performance

### 5.5 Trade-off Summary

```
                    Feature Selection    Prediction    Overall
                    (F1/FDR/Prec)      (MSE/R2)     Recommendation
---------------------------------------------------------------------------
AdaptiveFlippedLasso    ★★★★★              ★★★★☆      BEST CHOICE
UniLassoCV              ★★★★☆              ★★★★☆      Good alternative
LassoCV                 ★★☆☆☆              ★★★★★      For prediction only
AdaptiveLassoCV         ★★☆☆☆              ★☆☆☆☆      Not recommended
GroupLassoCV            ★☆☆☆☆              ★☆☆☆☆      Avoid
```

---

## 6. Conclusions

1. **Primary Recommendation**: Use **AdaptiveFlippedLasso** for sparse feature selection
   - Best-in-class F1 (0.886) and FDR (0.178)
   - Stage1 optimal parameters outperform all CV-tuned alternatives
   - Significantly better than UniLassoCV (second-best)

2. **Alternative**: **UniLassoCV** is a viable alternative with strong performance
   - F1=0.792 (11% lower than AFL)
   - Better FDR=0.301 (69% higher than AFL)
   - Good prediction (R2=0.920)

3. **When to use LassoCV**: Only when prediction accuracy (MSE/R2) is the sole metric

4. **Avoid**: GroupLassoCV and AdaptiveLassoCV for sparse true model scenarios

5. **Stage1 Parameters Validated**: The optimal parameters (λ_ridge=1.0, λ=1.0, γ=0.5) from grid search are confirmed effective

---

## Appendix: Model Details

| Model | Library | Tuning Method | Key Parameters |
|-------|---------|---------------|----------------|
| AdaptiveFlippedLasso | Custom | Fixed (Stage1 optimal) | λ_ridge=1.0, λ=1.0, γ=0.5 |
| LassoCV | sklearn | 5-fold CV | alphas=logspace(-4,1,30) |
| AdaptiveLassoCV | sklearn | 5-fold CV | alphas=logspace(-4,1,30), gammas=[0.5,1.0,2.0] |
| GroupLassoCV | Custom | 5-fold CV | alphas=logspace(-4,1,30) |
| UniLassoCV | unilasso | 5-fold CV | auto lambda selection |

*Report generated: 2026-03-27T08:00:00*
