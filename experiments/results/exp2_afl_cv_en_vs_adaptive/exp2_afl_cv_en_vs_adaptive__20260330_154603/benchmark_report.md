# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-30
**Experiment**: exp2_afl_cv_en_vs_adaptive
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AdaptiveLassoCV (fixed)** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AdaptiveLassoCV (fixed) | 0.6516 | 0.3001 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveLassoCV (fixed) | 44.1720 | 60.3546 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveLassoCV (fixed) | 0.7133 | 0.3573 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveLassoCV (fixed) | 0.3770 | 0.2493 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |
|--------:|--------------------------:|
|     0.5 |                    0.9355 |
|     2   |                    0.7518 |
|    10   |                    0.2675 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |
|--------:|--------------------------:|
|     0.5 |                    0.9056 |
|     2   |                    6.5673 |
|    10   |                  125.043  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |
|--------:|--------------------------:|
|     0.5 |                      1    |
|     2   |                      0.9  |
|    10   |                      0.24 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |
|--------:|--------------------------:|
|     0.5 |                    0.1199 |
|     2   |                    0.3469 |
|    10   |                    0.6643 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.9355 | 0.9056 | 1.0000 | 0.1199 | 0.8801 | 1.0000 | 0.9872 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.7518 | 6.5673 | 0.9000 | 0.3469 | 0.6531 | 0.9000 | 0.9170 |
| 10.0 | 0.1 | AdaptiveLassoCV (fixed) | 0.2675 | 125.0432 | 0.2400 | 0.6643 | 0.3357 | 0.2400 | 0.3101 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveLassoCV (fixed) (F1=0.6516)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveLassoCV (fixed) (MSE=44.1720)

## 7. Key Findings

1. **Best F1**: AdaptiveLassoCV (fixed) with F1=0.6516
2. **Best MSE**: AdaptiveLassoCV (fixed) with MSE=44.1720

3. **SNR Sensitivity**:
   - AdaptiveLassoCV (fixed): F1 drop = 0.4258 (high SNR to low SNR)

---
*Report generated: 2026-03-30T15:50:55.572858*
