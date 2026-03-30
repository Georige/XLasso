# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp1_afl_cv_sigma
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 3 per configuration

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
| AdaptiveFlippedLassoCV | 0.5599 | 0.3492 |
| AdaptiveLassoCV | 0.2684 | 0.1853 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 36.8067 | 51.3252 |
| AdaptiveLassoCV | 234.2897 | 45.8728 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.7611 | 0.3586 |
| AdaptiveLassoCV | 0.1667 | 0.1173 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveLassoCV | 0.0741 | 0.2222 |
| AdaptiveFlippedLassoCV | 0.5172 | 0.3626 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |   AdaptiveLassoCV |
|--------:|-------------------------:|------------------:|
|     0.5 |                   0.9616 |            0.4    |
|     2   |                   0.5334 |            0.3761 |
|    10   |                   0.1846 |            0.029  |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |   AdaptiveLassoCV |
|--------:|-------------------------:|------------------:|
|     0.5 |                   0.3984 |           206.539 |
|     2   |                   5.4138 |           205.338 |
|    10   |                 104.608  |           290.992 |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |   AdaptiveLassoCV |
|--------:|-------------------------:|------------------:|
|     0.5 |                   1      |            0.25   |
|     2   |                   1      |            0.2333 |
|    10   |                   0.2833 |            0.0167 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |   AdaptiveLassoCV |
|--------:|-------------------------:|------------------:|
|     0.5 |                   0.0714 |            0      |
|     2   |                   0.6242 |            0      |
|    10   |                   0.8561 |            0.2222 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.9616 | 0.3984 | 1.0000 | 0.0714 | 0.9286 | 1.0000 | 0.9981 |
| 0.5 | 2.0 | AdaptiveLassoCV | 0.4000 | 206.5391 | 0.2500 | 0.0000 | 1.0000 | 0.2500 | 0.0001 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.5334 | 5.4138 | 1.0000 | 0.6242 | 0.3758 | 1.0000 | 0.9738 |
| 2.0 | 0.5 | AdaptiveLassoCV | 0.3761 | 205.3376 | 0.2333 | 0.0000 | 1.0000 | 0.2333 | 0.0099 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.1846 | 104.6078 | 0.2833 | 0.8561 | 0.1439 | 0.2833 | 0.6233 |
| 10.0 | 0.1 | AdaptiveLassoCV | 0.0290 | 290.9923 | 0.0167 | 0.2222 | 0.1111 | 0.0167 | -0.0505 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.5599)
- rank_2: AdaptiveLassoCV (F1=0.2684)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=36.8067)
- rank_2: AdaptiveLassoCV (MSE=234.2897)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.5599
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=36.8067

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.6025 (high SNR to low SNR)
   - AdaptiveLassoCV: F1 drop = 0.1975 (high SNR to low SNR)

---
*Report generated: 2026-03-29T15:50:54.396887*
