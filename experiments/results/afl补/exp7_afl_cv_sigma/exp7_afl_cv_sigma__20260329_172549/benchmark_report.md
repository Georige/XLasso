# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp7_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.1406 | 0.1144 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 28.7850 | 42.0880 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.2500 | 0.3536 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.6499 | 0.3329 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2527 |
|     1   |                   0.2432 |
|     2   |                   0.1498 |
|     5   |                   0.0434 |
|    10   |                   0.0138 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   1.2074 |
|     1   |                   2.8856 |
|     2   |                   6.0263 |
|     5   |                  28.1039 |
|    10   |                 105.702  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                     0.83 |
|     1   |                     0.28 |
|     2   |                     0.1  |
|     5   |                     0.03 |
|    10   |                     0.01 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8116 |
|     1   |                   0.7033 |
|     2   |                   0.6361 |
|     5   |                   0.5206 |
|    10   |                   0.5778 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.2527 | 1.2074 | 0.8300 | 0.8116 | 0.1884 | 0.8300 | 0.5696 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.2432 | 2.8856 | 0.2800 | 0.7033 | 0.2967 | 0.2800 | 0.1945 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.1498 | 6.0263 | 0.1000 | 0.6361 | 0.3639 | 0.1000 | 0.0823 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.0434 | 28.1039 | 0.0300 | 0.5206 | 0.0794 | 0.0300 | -0.0193 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.0138 | 105.7021 | 0.0100 | 0.5778 | 0.0222 | 0.0100 | -0.0251 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.1406)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=28.7850)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.1406
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=28.7850

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.1837 (high SNR to low SNR)

---
*Report generated: 2026-03-29T17:42:21.877660*
