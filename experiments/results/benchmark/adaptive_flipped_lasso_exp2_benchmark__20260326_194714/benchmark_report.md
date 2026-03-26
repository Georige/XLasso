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
- **LassoCV** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AdaptiveFlippedLasso | 0.8868 | 0.1193 |
| LassoCV | 0.5437 | 0.0296 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| LassoCV | 3.7400 | 4.5279 |
| AdaptiveFlippedLasso | 4.0900 | 4.2272 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| LassoCV | 0.9924 | 0.0196 |
| AdaptiveFlippedLasso | 0.9640 | 0.0729 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLasso | 0.1713 | 0.1586 |
| LassoCV | 0.6219 | 0.0280 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLasso |   LassoCV |
|--------:|-----------------------:|----------:|
|     0.1 |                 0.9902 |    0.5625 |
|     0.5 |                 0.9854 |    0.5649 |
|     1   |                 0.9187 |    0.5286 |
|     2   |                 0.8468 |    0.5392 |
|     3   |                 0.6928 |    0.5231 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLasso |   LassoCV |
|--------:|-----------------------:|----------:|
|     0.1 |                 0.6448 |    0.0134 |
|     0.5 |                 0.9293 |    0.3342 |
|     1   |                 1.8064 |    1.3273 |
|     2   |                 5.538  |    5.2719 |
|     3   |                11.5317 |   11.7532 |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLasso |   LassoCV |
|--------:|-----------------------:|----------:|
|     0.1 |                   1    |     1     |
|     0.5 |                   1    |     1     |
|     1   |                   1    |     1     |
|     2   |                   0.98 |     1     |
|     3   |                   0.84 |     0.962 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLasso |   LassoCV |
|--------:|-----------------------:|----------:|
|     0.1 |                 0.019  |    0.6037 |
|     0.5 |                 0.0286 |    0.6006 |
|     1   |                 0.1484 |    0.6371 |
|     2   |                 0.2519 |    0.6291 |
|     3   |                 0.4084 |    0.6389 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AdaptiveFlippedLasso | 0.9902 | 0.6448 | 1.0000 | 0.0190 | 0.9810 | 1.0000 | 0.9924 |
| 0.1 | 10.0 | LassoCV | 0.5625 | 0.0134 | 1.0000 | 0.6037 | 0.3963 | 1.0000 | 0.9998 |
| 0.5 | 2.0 | AdaptiveFlippedLasso | 0.9854 | 0.9293 | 1.0000 | 0.0286 | 0.9714 | 1.0000 | 0.9887 |
| 0.5 | 2.0 | LassoCV | 0.5649 | 0.3342 | 1.0000 | 0.6006 | 0.3994 | 1.0000 | 0.9956 |
| 1.0 | 1.0 | AdaptiveFlippedLasso | 0.9187 | 1.8064 | 1.0000 | 0.1484 | 0.8516 | 1.0000 | 0.9778 |
| 1.0 | 1.0 | LassoCV | 0.5286 | 1.3273 | 1.0000 | 0.6371 | 0.3629 | 1.0000 | 0.9828 |
| 2.0 | 0.5 | AdaptiveFlippedLasso | 0.8468 | 5.5380 | 0.9800 | 0.2519 | 0.7481 | 0.9800 | 0.9336 |
| 2.0 | 0.5 | LassoCV | 0.5392 | 5.2719 | 1.0000 | 0.6291 | 0.3709 | 1.0000 | 0.9339 |
| 3.0 | 0.33 | AdaptiveFlippedLasso | 0.6928 | 11.5317 | 0.8400 | 0.4084 | 0.5916 | 0.8400 | 0.8676 |
| 3.0 | 0.33 | LassoCV | 0.5231 | 11.7532 | 0.9620 | 0.6389 | 0.3611 | 0.9620 | 0.8601 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLasso (F1=0.8868)
- rank_2: LassoCV (F1=0.5437)

### 6.2 By MSE (lower is better)

- rank_1: LassoCV (MSE=3.7400)
- rank_2: AdaptiveFlippedLasso (MSE=4.0900)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLasso with F1=0.8868
2. **Best MSE**: LassoCV with MSE=3.7400

3. **SNR Sensitivity**:
   - AdaptiveFlippedLasso: F1 drop = 0.2180 (high SNR to low SNR)
   - LassoCV: F1 drop = 0.0326 (high SNR to low SNR)

---
*Report generated: 2026-03-26T19:48:02.723629*
