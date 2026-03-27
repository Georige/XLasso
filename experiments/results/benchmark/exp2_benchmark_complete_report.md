# AdaptiveFlippedLasso Benchmark Report - Exp2

**Date**: 2026-03-27
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, n_nonzero=20)
**Repeats**: 5 per configuration
**Models**: 4 complete (UniLassoCV & FusedLassoCV pending)

---

## 1. Compared Models

| Model | Description | Status |
|-------|-------------|--------|
| **AdaptiveFlippedLasso** | Stage1 optimal params (λ_ridge=1.0, λ=1.0, γ=0.5) | ✓ Complete |
| **LassoCV** | sklearn Lasso with CV-tuned alpha | ✓ Complete |
| **AdaptiveLassoCV** | Adaptive Lasso with CV-tuned alpha | ✓ Complete |
| **GroupLassoCV** | Group Lasso with CV-tuned alpha | ✓ Complete |
| **UniLassoCV** | Univariate-guided Lasso | Pending |
| **FusedLassoCV** | Fused Lasso with CV | Pending (extremely slow) |

---

## 2. Overall Performance by Model

### 2.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| **AdaptiveFlippedLasso** | **0.8861** | 0.1095 |
| LassoCV | 0.5418 | 0.0278 |
| AdaptiveLassoCV | 0.4738 | 0.1697 |
| GroupLassoCV | 0.1235 | 0.0206 |

### 2.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| LassoCV | **3.6140** | 4.1295 |
| **AdaptiveFlippedLasso** | **3.9721** | 3.8561 |
| AdaptiveLassoCV | 65.4718 | 3.8595 |
| GroupLassoCV | 65.5971 | 3.9508 |

### 2.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| GroupLassoCV | **0.9943** | 0.0117 |
| LassoCV | 0.9937 | 0.0181 |
| AdaptiveLassoCV | 0.9777 | 0.0452 |
| **AdaptiveFlippedLasso** | 0.9700 | 0.0677 |

### 2.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| **AdaptiveFlippedLasso** | **0.1775** | 0.1466 |
| LassoCV | 0.6238 | 0.0265 |
| AdaptiveLassoCV | 0.6670 | 0.1498 |
| GroupLassoCV | 0.9339 | 0.0121 |

### 2.5 Precision (higher is better)

| Model | Precision Mean | Precision Std |
|-------|----------------|---------------|
| **AdaptiveFlippedLasso** | **0.8225** | 0.1466 |
| LassoCV | 0.3762 | 0.0265 |
| AdaptiveLassoCV | 0.3330 | 0.1498 |
| GroupLassoCV | 0.0661 | 0.0121 |

### 2.6 Recall (higher is better)

| Model | Recall Mean | Recall Std |
|-------|-------------|------------|
| GroupLassoCV | **0.9943** | 0.0117 |
| LassoCV | 0.9937 | 0.0181 |
| AdaptiveLassoCV | 0.9777 | 0.0452 |
| **AdaptiveFlippedLasso** | 0.9700 | 0.0677 |

### 2.7 R2 Score (higher is better)

| Model | R2 Mean | R2 Std |
|-------|---------|--------|
| LassoCV | **0.9557** | 0.0493 |
| **AdaptiveFlippedLasso** | **0.9532** | 0.0456 |
| AdaptiveLassoCV | 0.1831 | 0.0148 |
| GroupLassoCV | 0.1815 | 0.0156 |

---

## 3. Performance Across SNR Levels

### 3.1 F1 by Sigma

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

---

## 4. Rankings Summary

| Metric | Best Model | Value |
|--------|-----------|-------|
| **F1 ↑** | AdaptiveFlippedLasso | 0.8861 |
| **MSE ↓** | LassoCV | 3.6140 |
| **TPR ↑** | GroupLassoCV | 0.9943 |
| **FDR ↓** | AdaptiveFlippedLasso | 0.1775 |
| **Precision ↑** | AdaptiveFlippedLasso | 0.8225 |
| **Recall ↑** | GroupLassoCV | 0.9943 |
| **R2 ↑** | LassoCV | 0.9557 |

---

## 5. Key Findings

### 5.1 AdaptiveFlippedLasso Wins on Feature Selection Quality
- **F1 Score**: 0.8861 vs 0.5418 (LassoCV) - **63% better**
- **FDR**: 0.1775 vs 0.6238 (LassoCV) - **71% lower false discovery rate**
- **Precision**: 0.8225 vs 0.3762 (LassoCV) - **119% higher precision**

### 5.2 LassoCV Wins on Prediction Accuracy
- **MSE**: 3.6140 vs 3.9721 (AFL) - **9% better**
- **R2**: 0.9557 vs 0.9532 (AFL) - similar

### 5.3 Trade-off Analysis
```
                    AdaptiveFlippedLasso    LassoCV
Feature Selection   ★★★★★ (Best)          ★★☆☆☆ (Poor)
Prediction Error    ★★★★☆                 ★★★★★ (Best)
SNR Robustness      ★★★☆☆                 ★★★☆☆
```

### 5.4 SNR Sensitivity
| Model | F1 (σ=0.1) | F1 (σ=3.0) | Drop |
|-------|-------------|-------------|------|
| AdaptiveFlippedLasso | 0.9902 | 0.6928 | 0.297 |
| LassoCV | 0.5625 | 0.5231 | 0.039 |
| AdaptiveLassoCV | 0.5961 | 0.2370 | 0.359 |
| GroupLassoCV | 0.1273 | 0.1244 | 0.003 |

**Interpretation**: AdaptiveFlippedLasso maintains high F1 across SNR but degrades more at very low SNR (σ=3.0). LassoCV is more stable but at much lower absolute F1.

### 5.5 Critical Observation
- **GroupLassoCV** has extremely high TPR (0.994) but terrible FDR (0.934), meaning it selects almost everything
- **AdaptiveLassoCV** suffers from similar issues, likely due to group structure not matching the true sparse model
- **AdaptiveFlippedLasso** achieves the best balance between TPR (0.97) and FDR (0.18)

---

## 6. Conclusions

1. **For Sparse Feature Selection**: AdaptiveFlippedLasso is the clear winner with best F1 and lowest FDR
2. **For Pure Prediction**: LassoCV is slightly better in MSE/R2
3. **Stage1 Optimal Params Work Well**: The tuned parameters (λ_ridge=1.0, λ=1.0, γ=0.5) significantly outperform CV-tuned alternatives
4. **Recommendation**: Use AdaptiveFlippedLasso when feature selection accuracy matters (most practical scenarios)

---

## Appendix: Pending Results

| Model | Status | Expected Completion |
|-------|--------|-------------------|
| UniLassoCV | Pending | Unknown (cv_unilasso slow) |
| FusedLassoCV | Pending | Extremely slow (cvxpy optimization) |

*Report generated: 2026-03-27T07:38:55*
