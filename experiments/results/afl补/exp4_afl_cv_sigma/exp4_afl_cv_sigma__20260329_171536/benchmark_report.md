# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp4_afl_cv_sigma
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AdaptiveFlippedLassoCV** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AdaptiveFlippedLassoCV | 0.7561 | 0.1499 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 35.7540 | 55.6517 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.9660 | 0.0703 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.3651 | 0.1693 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8704 |
|     1   |                   0.8375 |
|     2   |                   0.8613 |
|     5   |                   0.6858 |
|    10   |                   0.5259 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.6188 |
|     1   |                   1.4673 |
|     2   |                   5.026  |
|     5   |                  31.7227 |
|    10   |                 139.935  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                     1    |
|     1   |                     1    |
|     2   |                     1    |
|     5   |                     0.99 |
|    10   |                     0.84 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2285 |
|     1   |                   0.279  |
|     2   |                   0.2404 |
|     5   |                   0.4722 |
|    10   |                   0.6053 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8704 | 0.6188 | 1.0000 | 0.2285 | 0.7715 | 1.0000 | 0.9970 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.8375 | 1.4673 | 1.0000 | 0.2790 | 0.7210 | 1.0000 | 0.9929 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.8613 | 5.0260 | 1.0000 | 0.2404 | 0.7596 | 1.0000 | 0.9758 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.6858 | 31.7227 | 0.9900 | 0.4722 | 0.5278 | 0.9900 | 0.8595 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.5259 | 139.9351 | 0.8400 | 0.6053 | 0.3947 | 0.8400 | 0.5297 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.7561)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=35.7540)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.7561
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=35.7540

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.1794 (high SNR to low SNR)

---
*Report generated: 2026-03-29T17:19:21.997523*
