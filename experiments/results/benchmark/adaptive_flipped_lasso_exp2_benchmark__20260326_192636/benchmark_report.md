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

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AdaptiveFlippedLasso | 0.9091 | 0.1031 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLasso | 3.2222 | 3.5797 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLasso | 0.9775 | 0.0599 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLasso | 0.1436 | 0.1407 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLasso |
|--------:|-----------------------:|
|     0.1 |                 0.9902 |
|     0.3 |                 0.9902 |
|     0.5 |                 0.9854 |
|     0.7 |                 0.9663 |
|     1   |                 0.9187 |
|     1.5 |                 0.8825 |
|     2   |                 0.8468 |
|     3   |                 0.6928 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLasso |
|--------:|-----------------------:|
|     0.1 |                 0.6448 |
|     0.3 |                 0.7143 |
|     0.5 |                 0.9293 |
|     0.7 |                 1.2308 |
|     1   |                 1.8064 |
|     1.5 |                 3.3821 |
|     2   |                 5.538  |
|     3   |                11.5317 |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLasso |
|--------:|-----------------------:|
|     0.1 |                   1    |
|     0.3 |                   1    |
|     0.5 |                   1    |
|     0.7 |                   1    |
|     1   |                   1    |
|     1.5 |                   1    |
|     2   |                   0.98 |
|     3   |                   0.84 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLasso |
|--------:|-----------------------:|
|     0.1 |                 0.019  |
|     0.3 |                 0.019  |
|     0.5 |                 0.0286 |
|     0.7 |                 0.0649 |
|     1   |                 0.1484 |
|     1.5 |                 0.2087 |
|     2   |                 0.2519 |
|     3   |                 0.4084 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AdaptiveFlippedLasso | 0.9902 | 0.6448 | 1.0000 | 0.0190 | 0.9810 | 1.0000 | 0.9924 |
| 0.3 | 3.33 | AdaptiveFlippedLasso | 0.9902 | 0.7143 | 1.0000 | 0.0190 | 0.9810 | 1.0000 | 0.9914 |
| 0.5 | 2.0 | AdaptiveFlippedLasso | 0.9854 | 0.9293 | 1.0000 | 0.0286 | 0.9714 | 1.0000 | 0.9887 |
| 0.7 | 1.43 | AdaptiveFlippedLasso | 0.9663 | 1.2308 | 1.0000 | 0.0649 | 0.9351 | 1.0000 | 0.9850 |
| 1.0 | 1.0 | AdaptiveFlippedLasso | 0.9187 | 1.8064 | 1.0000 | 0.1484 | 0.8516 | 1.0000 | 0.9778 |
| 1.5 | 0.67 | AdaptiveFlippedLasso | 0.8825 | 3.3821 | 1.0000 | 0.2087 | 0.7913 | 1.0000 | 0.9589 |
| 2.0 | 0.5 | AdaptiveFlippedLasso | 0.8468 | 5.5380 | 0.9800 | 0.2519 | 0.7481 | 0.9800 | 0.9336 |
| 3.0 | 0.33 | AdaptiveFlippedLasso | 0.6928 | 11.5317 | 0.8400 | 0.4084 | 0.5916 | 0.8400 | 0.8676 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLasso (F1=0.9091)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLasso (MSE=3.2222)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLasso with F1=0.9091
2. **Best MSE**: AdaptiveFlippedLasso with MSE=3.2222

3. **SNR Sensitivity**:
   - AdaptiveFlippedLasso: F1 drop = 0.2188 (high SNR to low SNR)

---
*Report generated: 2026-03-26T19:27:30.228323*
