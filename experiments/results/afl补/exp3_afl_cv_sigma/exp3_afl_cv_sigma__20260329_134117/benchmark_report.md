# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp3_afl_cv_sigma
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 10 per configuration

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
| AdaptiveFlippedLassoCV | 0.7920 | 0.1644 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 29.6465 | 43.5120 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.8642 | 0.1894 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.2461 | 0.1472 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.9073 |
|     1   |                   0.9109 |
|     2   |                   0.8973 |
|     5   |                   0.7156 |
|    10   |                   0.5288 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.3719 |
|     1   |                   1.2115 |
|     2   |                   4.6527 |
|     5   |                  29.1751 |
|    10   |                 112.821  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                    1     |
|     1   |                    1     |
|     2   |                    0.999 |
|     5   |                    0.794 |
|    10   |                    0.528 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.1459 |
|     1   |                   0.1476 |
|     2   |                   0.1757 |
|     5   |                   0.3219 |
|    10   |                   0.4392 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.9073 | 0.3719 | 1.0000 | 0.1459 | 0.8541 | 1.0000 | 0.9972 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.9109 | 1.2115 | 1.0000 | 0.1476 | 0.8524 | 1.0000 | 0.9910 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.8973 | 4.6527 | 0.9990 | 0.1757 | 0.8243 | 0.9990 | 0.9662 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.7156 | 29.1751 | 0.7940 | 0.3219 | 0.6781 | 0.7940 | 0.8167 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.5288 | 112.8214 | 0.5280 | 0.4392 | 0.5608 | 0.5280 | 0.5164 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.7920)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=29.6465)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.7920
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=29.6465

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.1934 (high SNR to low SNR)

---
*Report generated: 2026-03-29T13:54:49.714877*
