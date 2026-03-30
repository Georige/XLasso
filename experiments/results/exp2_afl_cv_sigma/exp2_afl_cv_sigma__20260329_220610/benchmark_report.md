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

- **AdaptiveLassoCV (fixed)** (CV-tuned)
- **AdaptiveFlippedLassoCV** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AdaptiveLassoCV (fixed) | 0.6516 | 0.3001 |
| AdaptiveFlippedLassoCV | 0.6194 | 0.2499 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 41.1546 | 57.0565 |
| AdaptiveLassoCV (fixed) | 44.1720 | 60.3546 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.7567 | 0.3262 |
| AdaptiveLassoCV (fixed) | 0.7133 | 0.3573 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveLassoCV (fixed) | 0.3770 | 0.2493 |
| AdaptiveFlippedLassoCV | 0.4616 | 0.2138 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   0.8474 |                    0.9355 |
|     2   |                   0.7018 |                    0.7518 |
|    10   |                   0.309  |                    0.2675 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   0.3475 |                    0.9056 |
|     2   |                   5.3979 |                    6.5673 |
|    10   |                 117.718  |                  125.043  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                     1    |                      1    |
|     2   |                     0.95 |                      0.9  |
|    10   |                     0.32 |                      0.24 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   0.2612 |                    0.1199 |
|     2   |                   0.4369 |                    0.3469 |
|    10   |                   0.6867 |                    0.6643 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.9355 | 0.9056 | 1.0000 | 0.1199 | 0.8801 | 1.0000 | 0.9872 |
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8474 | 0.3475 | 1.0000 | 0.2612 | 0.7388 | 1.0000 | 0.9954 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.7518 | 6.5673 | 0.9000 | 0.3469 | 0.6531 | 0.9000 | 0.9170 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.7018 | 5.3979 | 0.9500 | 0.4369 | 0.5631 | 0.9500 | 0.9322 |
| 10.0 | 0.1 | AdaptiveLassoCV (fixed) | 0.2675 | 125.0432 | 0.2400 | 0.6643 | 0.3357 | 0.2400 | 0.3101 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.3090 | 117.7183 | 0.3200 | 0.6867 | 0.3133 | 0.3200 | 0.3515 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveLassoCV (fixed) (F1=0.6516)
- rank_2: AdaptiveFlippedLassoCV (F1=0.6194)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=41.1546)
- rank_2: AdaptiveLassoCV (fixed) (MSE=44.1720)

## 7. Key Findings

1. **Best F1**: AdaptiveLassoCV (fixed) with F1=0.6516
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=41.1546

3. **SNR Sensitivity**:
   - AdaptiveLassoCV (fixed): F1 drop = 0.4258 (high SNR to low SNR)
   - AdaptiveFlippedLassoCV: F1 drop = 0.3419 (high SNR to low SNR)

---
*Report generated: 2026-03-29T22:12:52.120397*
