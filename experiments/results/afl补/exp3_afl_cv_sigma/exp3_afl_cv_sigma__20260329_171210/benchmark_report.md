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
| AdaptiveFlippedLassoCV | 0.7710 | 0.1972 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 30.4553 | 45.7719 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.8460 | 0.2096 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.2701 | 0.2144 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8545 |
|     1   |                   0.937  |
|     2   |                   0.8696 |
|     5   |                   0.7043 |
|    10   |                   0.4896 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.4455 |
|     1   |                   1.3558 |
|     2   |                   5.0477 |
|     5   |                  30.3848 |
|    10   |                 115.042  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                     1    |
|     1   |                     1    |
|     2   |                     0.99 |
|     5   |                     0.76 |
|    10   |                     0.48 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2257 |
|     1   |                   0.1049 |
|     2   |                   0.2081 |
|     5   |                   0.325  |
|    10   |                   0.4869 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8545 | 0.4455 | 1.0000 | 0.2257 | 0.7743 | 1.0000 | 0.9968 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.9370 | 1.3558 | 1.0000 | 0.1049 | 0.8951 | 1.0000 | 0.9904 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.8696 | 5.0477 | 0.9900 | 0.2081 | 0.7919 | 0.9900 | 0.9646 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.7043 | 30.3848 | 0.7600 | 0.3250 | 0.6750 | 0.7600 | 0.8101 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.4896 | 115.0424 | 0.4800 | 0.4869 | 0.5131 | 0.4800 | 0.5024 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.7710)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=30.4553)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.7710
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=30.4553

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.1667 (high SNR to low SNR)

---
*Report generated: 2026-03-29T17:15:33.465776*
