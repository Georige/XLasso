# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp1_afl_cv_sigma
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AdaptiveFlippedLassoCV** (CV-tuned)
- **AdaptiveLassoCV** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AdaptiveFlippedLassoCV | 0.5941 | 0.3285 |
| AdaptiveLassoCV | 0.2949 | 0.2142 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 40.1231 | 55.0487 |
| AdaptiveLassoCV | 241.4289 | 57.8135 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.7667 | 0.3421 |
| AdaptiveLassoCV | 0.1900 | 0.1429 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveLassoCV | 0.0444 | 0.1721 |
| AdaptiveFlippedLassoCV | 0.4859 | 0.3317 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |   AdaptiveLassoCV |
|--------:|-------------------------:|------------------:|
|     0.5 |                   0.9492 |            0.458  |
|     2   |                   0.6229 |            0.4093 |
|    10   |                   0.2101 |            0.0174 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |   AdaptiveLassoCV |
|--------:|-------------------------:|------------------:|
|     0.5 |                   0.5395 |           205.831 |
|     2   |                   5.6827 |           207.237 |
|    10   |                 114.147  |           311.219 |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |   AdaptiveLassoCV |
|--------:|-------------------------:|------------------:|
|     0.5 |                      1   |              0.3  |
|     2   |                      1   |              0.26 |
|    10   |                      0.3 |              0.01 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |   AdaptiveLassoCV |
|--------:|-------------------------:|------------------:|
|     0.5 |                   0.0944 |            0      |
|     2   |                   0.5301 |            0      |
|    10   |                   0.8332 |            0.1333 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.9492 | 0.5395 | 1.0000 | 0.0944 | 0.9056 | 1.0000 | 0.9975 |
| 0.5 | 2.0 | AdaptiveLassoCV | 0.4580 | 205.8306 | 0.3000 | 0.0000 | 1.0000 | 0.3000 | 0.0333 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.6229 | 5.6827 | 1.0000 | 0.5301 | 0.4699 | 1.0000 | 0.9733 |
| 2.0 | 0.5 | AdaptiveLassoCV | 0.4093 | 207.2368 | 0.2600 | 0.0000 | 1.0000 | 0.2600 | 0.0352 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.2101 | 114.1471 | 0.3000 | 0.8332 | 0.1668 | 0.3000 | 0.6192 |
| 10.0 | 0.1 | AdaptiveLassoCV | 0.0174 | 311.2193 | 0.0100 | 0.1333 | 0.0667 | 0.0100 | -0.0336 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.5941)
- rank_2: AdaptiveLassoCV (F1=0.2949)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=40.1231)
- rank_2: AdaptiveLassoCV (MSE=241.4289)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.5941
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=40.1231

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.5327 (high SNR to low SNR)
   - AdaptiveLassoCV: F1 drop = 0.2446 (high SNR to low SNR)

---
*Report generated: 2026-03-29T16:37:29.041419*
