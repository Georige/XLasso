# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp3_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.7960 | 0.1786 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 30.6030 | 46.0823 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.8420 | 0.2050 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.2173 | 0.1953 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8563 |
|     1   |                   0.9216 |
|     2   |                   0.917  |
|     5   |                   0.7313 |
|    10   |                   0.5539 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.4446 |
|     1   |                   1.3582 |
|     2   |                   4.9957 |
|     5   |                  30.9084 |
|    10   |                 115.308  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                     1    |
|     1   |                     1    |
|     2   |                     0.99 |
|     5   |                     0.71 |
|    10   |                     0.51 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2239 |
|     1   |                   0.1235 |
|     2   |                   0.1347 |
|     5   |                   0.2323 |
|    10   |                   0.372  |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8563 | 0.4446 | 1.0000 | 0.2239 | 0.7761 | 1.0000 | 0.9968 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.9216 | 1.3582 | 1.0000 | 0.1235 | 0.8765 | 1.0000 | 0.9904 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.9170 | 4.9957 | 0.9900 | 0.1347 | 0.8653 | 0.9900 | 0.9651 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.7313 | 30.9084 | 0.7100 | 0.2323 | 0.7677 | 0.7100 | 0.8082 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.5539 | 115.3082 | 0.5100 | 0.3720 | 0.6280 | 0.5100 | 0.5029 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.7960)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=30.6030)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.7960
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=30.6030

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.1222 (high SNR to low SNR)

---
*Report generated: 2026-03-29T18:21:29.453746*
