# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-27
**Experiment**: adaptive_flipped_lasso_exp2_benchmark
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AdaptiveFlippedLasso** (fixed params)
- **LassoCV** (CV-tuned)
- **AdaptiveLassoCV** (CV-tuned)
- **GroupLassoCV** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AdaptiveFlippedLasso | 0.8861 | 0.1095 |
| LassoCV | 0.5418 | 0.0278 |
| AdaptiveLassoCV | 0.4738 | 0.1697 |
| GroupLassoCV | 0.1235 | 0.0206 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| LassoCV | 3.6140 | 4.1295 |
| AdaptiveFlippedLasso | 3.9721 | 3.8561 |
| AdaptiveLassoCV | 65.4718 | 3.8595 |
| GroupLassoCV | 65.5971 | 3.9508 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| GroupLassoCV | 0.9943 | 0.0117 |
| LassoCV | 0.9937 | 0.0181 |
| AdaptiveLassoCV | 0.9777 | 0.0452 |
| AdaptiveFlippedLasso | 0.9700 | 0.0677 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLasso | 0.1775 | 0.1466 |
| LassoCV | 0.6238 | 0.0265 |
| AdaptiveLassoCV | 0.6670 | 0.1498 |
| GroupLassoCV | 0.9339 | 0.0121 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLasso |   AdaptiveLassoCV |   GroupLassoCV |   LassoCV |
|--------:|-----------------------:|------------------:|---------------:|----------:|
|     0.1 |                 0.9902 |            0.5961 |         0.1273 |    0.5625 |
|     0.5 |                 0.9854 |            0.7031 |         0.1189 |    0.5649 |
|     1   |                 0.9187 |            0.5322 |         0.1177 |    0.5286 |
|     1.5 |                 0.8825 |            0.4357 |         0.1239 |    0.5323 |
|     2   |                 0.8468 |            0.3388 |         0.1287 |    0.5392 |
|     3   |                 0.6928 |            0.237  |         0.1244 |    0.5231 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLasso |   AdaptiveLassoCV |   GroupLassoCV |   LassoCV |
|--------:|-----------------------:|------------------:|---------------:|----------:|
|     0.1 |                 0.6448 |           63.0694 |        63.0793 |    0.0134 |
|     0.5 |                 0.9293 |           63.1367 |        63.1916 |    0.3342 |
|     1   |                 1.8064 |           63.7051 |        63.8119 |    1.3273 |
|     1.5 |                 3.3821 |           64.8418 |        65.0217 |    2.9843 |
|     2   |                 5.538  |           66.5431 |        66.7093 |    5.2719 |
|     3   |                11.5317 |           71.5347 |        71.7691 |   11.7532 |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLasso |   AdaptiveLassoCV |   GroupLassoCV |   LassoCV |
|--------:|-----------------------:|------------------:|---------------:|----------:|
|     0.1 |                   1    |             1     |          1     |     1     |
|     0.5 |                   1    |             1     |          1     |     1     |
|     1   |                   1    |             1     |          1     |     1     |
|     1.5 |                   1    |             0.998 |          1     |     1     |
|     2   |                   0.98 |             0.984 |          0.996 |     1     |
|     3   |                   0.84 |             0.884 |          0.97  |     0.962 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLasso |   AdaptiveLassoCV |   GroupLassoCV |   LassoCV |
|--------:|-----------------------:|------------------:|---------------:|----------:|
|     0.1 |                 0.019  |            0.5749 |         0.9319 |    0.6037 |
|     0.5 |                 0.0286 |            0.4396 |         0.9366 |    0.6006 |
|     1   |                 0.1484 |            0.6255 |         0.9374 |    0.6371 |
|     1.5 |                 0.2087 |            0.7115 |         0.9337 |    0.6332 |
|     2   |                 0.2519 |            0.7899 |         0.9307 |    0.6291 |
|     3   |                 0.4084 |            0.8605 |         0.9328 |    0.6389 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AdaptiveFlippedLasso | 0.9902 | 0.6448 | 1.0000 | 0.0190 | 0.9810 | 1.0000 | 0.9924 |
| 0.1 | 10.0 | LassoCV | 0.5625 | 0.0134 | 1.0000 | 0.6037 | 0.3963 | 1.0000 | 0.9998 |
| 0.1 | 10.0 | AdaptiveLassoCV | 0.5961 | 63.0694 | 1.0000 | 0.5749 | 0.4251 | 1.0000 | 0.1945 |
| 0.1 | 10.0 | GroupLassoCV | 0.1273 | 63.0793 | 1.0000 | 0.9319 | 0.0681 | 1.0000 | 0.1944 |
| 0.5 | 2.0 | AdaptiveFlippedLasso | 0.9854 | 0.9293 | 1.0000 | 0.0286 | 0.9714 | 1.0000 | 0.9887 |
| 0.5 | 2.0 | LassoCV | 0.5649 | 0.3342 | 1.0000 | 0.6006 | 0.3994 | 1.0000 | 0.9956 |
| 0.5 | 2.0 | AdaptiveLassoCV | 0.7031 | 63.1367 | 1.0000 | 0.4396 | 0.5604 | 1.0000 | 0.1932 |
| 0.5 | 2.0 | GroupLassoCV | 0.1189 | 63.1916 | 1.0000 | 0.9366 | 0.0634 | 1.0000 | 0.1925 |
| 1.0 | 1.0 | AdaptiveFlippedLasso | 0.9187 | 1.8064 | 1.0000 | 0.1484 | 0.8516 | 1.0000 | 0.9778 |
| 1.0 | 1.0 | LassoCV | 0.5286 | 1.3273 | 1.0000 | 0.6371 | 0.3629 | 1.0000 | 0.9828 |
| 1.0 | 1.0 | AdaptiveLassoCV | 0.5322 | 63.7051 | 1.0000 | 0.6255 | 0.3745 | 1.0000 | 0.1900 |
| 1.0 | 1.0 | GroupLassoCV | 0.1177 | 63.8119 | 1.0000 | 0.9374 | 0.0626 | 1.0000 | 0.1886 |
| 1.5 | 0.67 | AdaptiveFlippedLasso | 0.8825 | 3.3821 | 1.0000 | 0.2087 | 0.7913 | 1.0000 | 0.9589 |
| 1.5 | 0.67 | LassoCV | 0.5323 | 2.9843 | 1.0000 | 0.6332 | 0.3668 | 1.0000 | 0.9619 |
| 1.5 | 0.67 | AdaptiveLassoCV | 0.4357 | 64.8418 | 0.9980 | 0.7115 | 0.2885 | 0.9980 | 0.1846 |
| 1.5 | 0.67 | GroupLassoCV | 0.1239 | 65.0217 | 1.0000 | 0.9337 | 0.0663 | 1.0000 | 0.1822 |
| 2.0 | 0.5 | AdaptiveFlippedLasso | 0.8468 | 5.5380 | 0.9800 | 0.2519 | 0.7481 | 0.9800 | 0.9336 |
| 2.0 | 0.5 | LassoCV | 0.5392 | 5.2719 | 1.0000 | 0.6291 | 0.3709 | 1.0000 | 0.9339 |
| 2.0 | 0.5 | AdaptiveLassoCV | 0.3388 | 66.5431 | 0.9840 | 0.7899 | 0.2101 | 0.9840 | 0.1773 |
| 2.0 | 0.5 | GroupLassoCV | 0.1287 | 66.7093 | 0.9960 | 0.9307 | 0.0693 | 0.9960 | 0.1751 |
| 3.0 | 0.33 | AdaptiveFlippedLasso | 0.6928 | 11.5317 | 0.8400 | 0.4084 | 0.5916 | 0.8400 | 0.8676 |
| 3.0 | 0.33 | LassoCV | 0.5231 | 11.7532 | 0.9620 | 0.6389 | 0.3611 | 0.9620 | 0.8601 |
| 3.0 | 0.33 | AdaptiveLassoCV | 0.2370 | 71.5347 | 0.8840 | 0.8605 | 0.1395 | 0.8840 | 0.1589 |
| 3.0 | 0.33 | GroupLassoCV | 0.1244 | 71.7691 | 0.9700 | 0.9328 | 0.0672 | 0.9700 | 0.1560 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLasso (F1=0.8861)
- rank_2: LassoCV (F1=0.5418)
- rank_3: AdaptiveLassoCV (F1=0.4738)
- rank_4: GroupLassoCV (F1=0.1235)

### 6.2 By MSE (lower is better)

- rank_1: LassoCV (MSE=3.6140)
- rank_2: AdaptiveFlippedLasso (MSE=3.9721)
- rank_3: AdaptiveLassoCV (MSE=65.4718)
- rank_4: GroupLassoCV (MSE=65.5971)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLasso with F1=0.8861
2. **Best MSE**: LassoCV with MSE=3.6140

3. **SNR Sensitivity**:
   - AdaptiveFlippedLasso: F1 drop = 0.2180 (high SNR to low SNR)
   - LassoCV: F1 drop = 0.0326 (high SNR to low SNR)
   - AdaptiveLassoCV: F1 drop = 0.3617 (high SNR to low SNR)
   - GroupLassoCV: F1 drop = -0.0035 (high SNR to low SNR)

---
*Report generated: 2026-03-27T04:49:47.599028*
