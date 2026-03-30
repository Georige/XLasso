# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp6_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.6585 | 0.2698 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 29.8680 | 43.6819 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.8060 | 0.3030 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.3605 | 0.1867 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.9037 |
|     1   |                   0.8651 |
|     2   |                   0.7857 |
|     5   |                   0.5213 |
|    10   |                   0.2167 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2934 |
|     1   |                   1.113  |
|     2   |                   4.5385 |
|     5   |                  30.1297 |
|    10   |                 113.265  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                    1     |
|     1   |                    1     |
|     2   |                    1     |
|     5   |                    0.786 |
|    10   |                    0.244 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.1599 |
|     1   |                   0.2223 |
|     2   |                   0.3371 |
|     5   |                   0.5751 |
|    10   |                   0.5078 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.9037 | 0.2934 | 1.0000 | 0.1599 | 0.8401 | 1.0000 | 0.9797 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.8651 | 1.1130 | 1.0000 | 0.2223 | 0.7777 | 1.0000 | 0.9267 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.7857 | 4.5385 | 1.0000 | 0.3371 | 0.6629 | 1.0000 | 0.7503 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.5213 | 30.1297 | 0.7860 | 0.5751 | 0.4249 | 0.7860 | 0.2262 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.2167 | 113.2654 | 0.2440 | 0.5078 | 0.2922 | 0.2440 | -0.0082 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.6585)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=29.8680)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.6585
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=29.8680

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.3958 (high SNR to low SNR)

---
*Report generated: 2026-03-29T14:39:24.098930*
