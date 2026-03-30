# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp6_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.5868 | 0.2907 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 26.1454 | 40.3009 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.7920 | 0.3439 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.4236 | 0.2686 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8495 |
|     1   |                   0.7538 |
|     2   |                   0.701  |
|     5   |                   0.454  |
|    10   |                   0.1759 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2764 |
|     1   |                   1.0198 |
|     2   |                   4.0522 |
|     5   |                  26.6607 |
|    10   |                  98.7181 |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                     1    |
|     1   |                     1    |
|     2   |                     1    |
|     5   |                     0.8  |
|    10   |                     0.16 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2386 |
|     1   |                   0.3672 |
|     2   |                   0.4496 |
|     5   |                   0.6638 |
|    10   |                   0.3986 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8495 | 0.2764 | 1.0000 | 0.2386 | 0.7614 | 1.0000 | 0.9811 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.7538 | 1.0198 | 1.0000 | 0.3672 | 0.6328 | 1.0000 | 0.9328 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.7010 | 4.0522 | 1.0000 | 0.4496 | 0.5504 | 1.0000 | 0.7736 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.4540 | 26.6607 | 0.8000 | 0.6638 | 0.3362 | 0.8000 | 0.2745 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.1759 | 98.7181 | 0.1600 | 0.3986 | 0.2014 | 0.1600 | 0.0403 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.5868)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=26.1454)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.5868
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=26.1454

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.4058 (high SNR to low SNR)

---
*Report generated: 2026-03-29T17:25:46.298489*
