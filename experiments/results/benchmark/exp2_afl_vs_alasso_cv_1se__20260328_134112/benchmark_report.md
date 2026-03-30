# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-28
**Experiment**: exp2_afl_vs_alasso_cv_1se
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AFL-CV-1SE** (CV-tuned)
- **AdaptiveLassoCV-1SE** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AFL-CV-1SE | 0.8933 | 0.0915 |
| AdaptiveLassoCV-1SE | 0.8049 | 0.1400 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-1SE | 3.0809 | 4.1863 |
| AdaptiveLassoCV-1SE | 68.2899 | 3.8010 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.9725 | 0.0655 |
| AdaptiveLassoCV-1SE | 0.7263 | 0.1712 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveLassoCV-1SE | 0.0713 | 0.0872 |
| AFL-CV-1SE | 0.1674 | 0.1173 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLassoCV-1SE |
|--------:|-------------:|----------------------:|
|     0.1 |       0.9923 |                0.9072 |
|     0.3 |       0.9718 |                0.902  |
|     0.5 |       0.9516 |                0.8956 |
|     0.7 |       0.9265 |                0.8838 |
|     1   |       0.9046 |                0.8418 |
|     1.5 |       0.8737 |                0.7721 |
|     2   |       0.8203 |                0.6839 |
|     3   |       0.7053 |                0.5526 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLassoCV-1SE |
|--------:|-------------:|----------------------:|
|     0.1 |       0.1433 |               66.0913 |
|     0.3 |       0.2535 |               66.2922 |
|     0.5 |       0.4473 |               66.4241 |
|     0.7 |       0.7348 |               66.5272 |
|     1   |       1.3987 |               67.1479 |
|     1.5 |       3.1054 |               68.245  |
|     2   |       5.5991 |               70.2401 |
|     3   |      12.9652 |               75.3515 |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLassoCV-1SE |
|--------:|-------------:|----------------------:|
|     0.1 |        1     |                 0.846 |
|     0.3 |        1     |                 0.836 |
|     0.5 |        1     |                 0.828 |
|     0.7 |        1     |                 0.812 |
|     1   |        1     |                 0.766 |
|     1.5 |        0.998 |                 0.69  |
|     2   |        0.97  |                 0.582 |
|     3   |        0.812 |                 0.45  |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLassoCV-1SE |
|--------:|-------------:|----------------------:|
|     0.1 |       0.0149 |                0.0045 |
|     0.3 |       0.0534 |                0.0024 |
|     0.5 |       0.0907 |                0.0099 |
|     0.7 |       0.1349 |                0.0128 |
|     1   |       0.1725 |                0.0427 |
|     1.5 |       0.2207 |                0.1    |
|     2   |       0.2859 |                0.1426 |
|     3   |       0.3665 |                0.256  |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AFL-CV-1SE | 0.9923 | 0.1433 | 1.0000 | 0.0149 | 0.9851 | 1.0000 | 0.9982 |
| 0.1 | 10.0 | AdaptiveLassoCV-1SE | 0.9072 | 66.0913 | 0.8460 | 0.0045 | 0.9955 | 0.8460 | 0.1560 |
| 0.3 | 3.33 | AFL-CV-1SE | 0.9718 | 0.2535 | 1.0000 | 0.0534 | 0.9466 | 1.0000 | 0.9967 |
| 0.3 | 3.33 | AdaptiveLassoCV-1SE | 0.9020 | 66.2922 | 0.8360 | 0.0024 | 0.9976 | 0.8360 | 0.1526 |
| 0.5 | 2.0 | AFL-CV-1SE | 0.9516 | 0.4473 | 1.0000 | 0.0907 | 0.9093 | 1.0000 | 0.9942 |
| 0.5 | 2.0 | AdaptiveLassoCV-1SE | 0.8956 | 66.4241 | 0.8280 | 0.0099 | 0.9901 | 0.8280 | 0.1512 |
| 0.7 | 1.43 | AFL-CV-1SE | 0.9265 | 0.7348 | 1.0000 | 0.1349 | 0.8651 | 1.0000 | 0.9905 |
| 0.7 | 1.43 | AdaptiveLassoCV-1SE | 0.8838 | 66.5272 | 0.8120 | 0.0128 | 0.9872 | 0.8120 | 0.1507 |
| 1.0 | 1.0 | AFL-CV-1SE | 0.9046 | 1.3987 | 1.0000 | 0.1725 | 0.8275 | 1.0000 | 0.9819 |
| 1.0 | 1.0 | AdaptiveLassoCV-1SE | 0.8418 | 67.1479 | 0.7660 | 0.0427 | 0.9573 | 0.7660 | 0.1466 |
| 1.5 | 0.67 | AFL-CV-1SE | 0.8737 | 3.1054 | 0.9980 | 0.2207 | 0.7793 | 0.9980 | 0.9603 |
| 1.5 | 0.67 | AdaptiveLassoCV-1SE | 0.7721 | 68.2450 | 0.6900 | 0.1000 | 0.9000 | 0.6900 | 0.1422 |
| 2.0 | 0.5 | AFL-CV-1SE | 0.8203 | 5.5991 | 0.9700 | 0.2859 | 0.7141 | 0.9700 | 0.9295 |
| 2.0 | 0.5 | AdaptiveLassoCV-1SE | 0.6839 | 70.2401 | 0.5820 | 0.1426 | 0.8574 | 0.5820 | 0.1314 |
| 3.0 | 0.33 | AFL-CV-1SE | 0.7053 | 12.9652 | 0.8120 | 0.3665 | 0.6335 | 0.8120 | 0.8454 |
| 3.0 | 0.33 | AdaptiveLassoCV-1SE | 0.5526 | 75.3515 | 0.4500 | 0.2560 | 0.7440 | 0.4500 | 0.1141 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-1SE (F1=0.8933)
- rank_2: AdaptiveLassoCV-1SE (F1=0.8049)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-1SE (MSE=3.0809)
- rank_2: AdaptiveLassoCV-1SE (MSE=68.2899)

## 7. Key Findings

1. **Best F1**: AFL-CV-1SE with F1=0.8933
2. **Best MSE**: AFL-CV-1SE with MSE=3.0809

3. **SNR Sensitivity**:
   - AFL-CV-1SE: F1 drop = 0.2091 (high SNR to low SNR)
   - AdaptiveLassoCV-1SE: F1 drop = 0.2833 (high SNR to low SNR)

---
*Report generated: 2026-03-28T14:54:16.793443*
