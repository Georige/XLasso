# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-30
**Experiment**: exp2_afl_cv_sigma_eps
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
| AdaptiveFlippedLassoCV | 0.6496 | 0.2399 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 41.3466 | 57.5607 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.7800 | 0.3116 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.4341 | 0.2003 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8521 |
|     2   |                   0.7571 |
|    10   |                   0.3396 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.3472 |
|     2   |                   5.1998 |
|    10   |                 118.493  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                     1    |
|     2   |                     0.98 |
|    10   |                     0.36 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2564 |
|     2   |                   0.3772 |
|    10   |                   0.6687 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8521 | 0.3472 | 1.0000 | 0.2564 | 0.7436 | 1.0000 | 0.9954 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.7571 | 5.1998 | 0.9800 | 0.3772 | 0.6228 | 0.9800 | 0.9348 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.3396 | 118.4928 | 0.3600 | 0.6687 | 0.3313 | 0.3600 | 0.3478 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.6496)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=41.3466)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.6496
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=41.3466

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.3038 (high SNR to low SNR)

---
*Report generated: 2026-03-30T14:52:51.961004*
