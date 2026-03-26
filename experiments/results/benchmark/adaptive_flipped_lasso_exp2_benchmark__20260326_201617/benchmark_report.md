# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-26
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
- **Lasso** (fixed params)
- **AdaptiveLasso** (fixed params)
- **GroupLasso** (fixed params)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AdaptiveFlippedLasso | 0.8868 | 0.1193 |
| Lasso | 0.8191 | 0.0418 |
| AdaptiveLasso | 0.0000 | 0.0000 |
| GroupLasso | 0.0000 | 0.0000 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLasso | 4.0900 | 4.2272 |
| Lasso | 11.3274 | 4.0772 |
| GroupLasso | 85.9551 | 12.2674 |
| AdaptiveLasso | 85.9551 | 12.2674 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| Lasso | 0.9760 | 0.0357 |
| AdaptiveFlippedLasso | 0.9640 | 0.0729 |
| AdaptiveLasso | 0.0000 | 0.0000 |
| GroupLasso | 0.0000 | 0.0000 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveLasso | 0.0000 | 0.0000 |
| GroupLasso | 0.0000 | 0.0000 |
| AdaptiveFlippedLasso | 0.1713 | 0.1586 |
| Lasso | 0.2932 | 0.0501 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLasso |   AdaptiveLasso |   GroupLasso |   Lasso |
|--------:|-----------------------:|----------------:|-------------:|--------:|
|     0.1 |                 0.9902 |               0 |            0 |  0.8462 |
|     0.5 |                 0.9854 |               0 |            0 |  0.8429 |
|     1   |                 0.9187 |               0 |            0 |  0.8231 |
|     2   |                 0.8468 |               0 |            0 |  0.8107 |
|     3   |                 0.6928 |               0 |            0 |  0.7728 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLasso |   AdaptiveLasso |   GroupLasso |   Lasso |
|--------:|-----------------------:|----------------:|-------------:|--------:|
|     0.1 |                 0.6448 |         83.7629 |      83.7629 |  8.1706 |
|     0.5 |                 0.9293 |         83.7814 |      83.7814 |  8.4287 |
|     1   |                 1.8064 |         84.26   |      84.26   |  9.2631 |
|     2   |                 5.538  |         86.7355 |      86.7355 | 12.6304 |
|     3   |                11.5317 |         91.2354 |      91.2354 | 18.1442 |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLasso |   AdaptiveLasso |   GroupLasso |   Lasso |
|--------:|-----------------------:|----------------:|-------------:|--------:|
|     0.1 |                   1    |               0 |            0 |    0.99 |
|     0.5 |                   1    |               0 |            0 |    0.99 |
|     1   |                   1    |               0 |            0 |    0.99 |
|     2   |                   0.98 |               0 |            0 |    0.98 |
|     3   |                   0.84 |               0 |            0 |    0.93 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLasso |   AdaptiveLasso |   GroupLasso |   Lasso |
|--------:|-----------------------:|----------------:|-------------:|--------:|
|     0.1 |                 0.019  |               0 |            0 |  0.2608 |
|     0.5 |                 0.0286 |               0 |            0 |  0.266  |
|     1   |                 0.1484 |               0 |            0 |  0.2946 |
|     2   |                 0.2519 |               0 |            0 |  0.3078 |
|     3   |                 0.4084 |               0 |            0 |  0.3367 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AdaptiveFlippedLasso | 0.9902 | 0.6448 | 1.0000 | 0.0190 | 0.9810 | 1.0000 | 0.9924 |
| 0.1 | 10.0 | Lasso | 0.8462 | 8.1706 | 0.9900 | 0.2608 | 0.7392 | 0.9900 | 0.9023 |
| 0.1 | 10.0 | AdaptiveLasso | 0.0000 | 83.7629 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0127 |
| 0.1 | 10.0 | GroupLasso | 0.0000 | 83.7629 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0127 |
| 0.5 | 2.0 | AdaptiveFlippedLasso | 0.9854 | 0.9293 | 1.0000 | 0.0286 | 0.9714 | 1.0000 | 0.9887 |
| 0.5 | 2.0 | Lasso | 0.8429 | 8.4287 | 0.9900 | 0.2660 | 0.7340 | 0.9900 | 0.8985 |
| 0.5 | 2.0 | AdaptiveLasso | 0.0000 | 83.7814 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0140 |
| 0.5 | 2.0 | GroupLasso | 0.0000 | 83.7814 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0140 |
| 1.0 | 1.0 | AdaptiveFlippedLasso | 0.9187 | 1.8064 | 1.0000 | 0.1484 | 0.8516 | 1.0000 | 0.9778 |
| 1.0 | 1.0 | Lasso | 0.8231 | 9.2631 | 0.9900 | 0.2946 | 0.7054 | 0.9900 | 0.8881 |
| 1.0 | 1.0 | AdaptiveLasso | 0.0000 | 84.2600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0158 |
| 1.0 | 1.0 | GroupLasso | 0.0000 | 84.2600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0158 |
| 2.0 | 0.5 | AdaptiveFlippedLasso | 0.8468 | 5.5380 | 0.9800 | 0.2519 | 0.7481 | 0.9800 | 0.9336 |
| 2.0 | 0.5 | Lasso | 0.8107 | 12.6304 | 0.9800 | 0.3078 | 0.6922 | 0.9800 | 0.8495 |
| 2.0 | 0.5 | AdaptiveLasso | 0.0000 | 86.7355 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0201 |
| 2.0 | 0.5 | GroupLasso | 0.0000 | 86.7355 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0201 |
| 3.0 | 0.33 | AdaptiveFlippedLasso | 0.6928 | 11.5317 | 0.8400 | 0.4084 | 0.5916 | 0.8400 | 0.8676 |
| 3.0 | 0.33 | Lasso | 0.7728 | 18.1442 | 0.9300 | 0.3367 | 0.6633 | 0.9300 | 0.7929 |
| 3.0 | 0.33 | AdaptiveLasso | 0.0000 | 91.2354 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0247 |
| 3.0 | 0.33 | GroupLasso | 0.0000 | 91.2354 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0247 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLasso (F1=0.8868)
- rank_2: Lasso (F1=0.8191)
- rank_3: AdaptiveLasso (F1=0.0000)
- rank_4: GroupLasso (F1=0.0000)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLasso (MSE=4.0900)
- rank_2: Lasso (MSE=11.3274)
- rank_3: GroupLasso (MSE=85.9551)
- rank_4: AdaptiveLasso (MSE=85.9551)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLasso with F1=0.8868
2. **Best MSE**: AdaptiveFlippedLasso with MSE=4.0900

3. **SNR Sensitivity**:
   - AdaptiveFlippedLasso: F1 drop = 0.2180 (high SNR to low SNR)
   - Lasso: F1 drop = 0.0528 (high SNR to low SNR)
   - AdaptiveLasso: F1 drop = 0.0000 (high SNR to low SNR)
   - GroupLasso: F1 drop = 0.0000 (high SNR to low SNR)

---
*Report generated: 2026-03-26T20:17:21.143935*
