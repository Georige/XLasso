# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp2_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.6474 | 0.2091 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 30.9273 | 45.7844 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.8080 | 0.2581 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.4511 | 0.1838 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8496 |
|     1   |                   0.8145 |
|     2   |                   0.703  |
|     5   |                   0.542  |
|    10   |                   0.328  |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.3387 |
|     1   |                   1.2358 |
|     2   |                   5.2436 |
|     5   |                  31.1626 |
|    10   |                 116.656  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                     1    |
|     1   |                     1    |
|     2   |                     0.98 |
|     5   |                     0.69 |
|    10   |                     0.37 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2592 |
|     1   |                   0.3116 |
|     2   |                   0.4385 |
|     5   |                   0.5521 |
|    10   |                   0.6942 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8496 | 0.3387 | 1.0000 | 0.2592 | 0.7408 | 1.0000 | 0.9955 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.8145 | 1.2358 | 1.0000 | 0.3116 | 0.6884 | 1.0000 | 0.9837 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.7030 | 5.2436 | 0.9800 | 0.4385 | 0.5615 | 0.9800 | 0.9341 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.5420 | 31.1626 | 0.6900 | 0.5521 | 0.4479 | 0.6900 | 0.6947 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.3280 | 116.6558 | 0.3700 | 0.6942 | 0.3058 | 0.3700 | 0.3551 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.6474)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=30.9273)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.6474
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=30.9273

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.3253 (high SNR to low SNR)

---
*Report generated: 2026-03-29T18:18:18.563704*
