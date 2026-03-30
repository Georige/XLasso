# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-28
**Experiment**: exp2_afl_vs_alasso_cv_1se
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 10 per configuration

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
| AFL-CV-1SE | 0.9107 | 0.0851 |
| AdaptiveLassoCV-1SE | 0.8255 | 0.1319 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-1SE | 3.0062 | 4.0282 |
| AdaptiveLassoCV-1SE | 69.6385 | 4.4108 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.9771 | 0.0568 |
| AdaptiveLassoCV-1SE | 0.7524 | 0.1654 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveLassoCV-1SE | 0.0609 | 0.0772 |
| AFL-CV-1SE | 0.1414 | 0.1119 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLassoCV-1SE |
|--------:|-------------:|----------------------:|
|     0.1 |       0.9922 |                0.931  |
|     0.3 |       0.9773 |                0.9268 |
|     0.5 |       0.9621 |                0.9115 |
|     0.7 |       0.9446 |                0.8967 |
|     1   |       0.9269 |                0.8614 |
|     1.5 |       0.8955 |                0.7889 |
|     2   |       0.8492 |                0.7092 |
|     3   |       0.7376 |                0.5787 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLassoCV-1SE |
|--------:|-------------:|----------------------:|
|     0.1 |       0.1546 |               66.9887 |
|     0.3 |       0.266  |               67.221  |
|     0.5 |       0.4469 |               67.5376 |
|     0.7 |       0.7423 |               67.7761 |
|     1   |       1.3825 |               68.3858 |
|     1.5 |       3.0341 |               69.8222 |
|     2   |       5.4668 |               71.9614 |
|     3   |      12.5561 |               77.4151 |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLassoCV-1SE |
|--------:|-------------:|----------------------:|
|     0.1 |        1     |                 0.883 |
|     0.3 |        1     |                 0.876 |
|     0.5 |        1     |                 0.854 |
|     0.7 |        1     |                 0.834 |
|     1   |        1     |                 0.793 |
|     1.5 |        0.999 |                 0.705 |
|     2   |        0.977 |                 0.606 |
|     3   |        0.841 |                 0.468 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLassoCV-1SE |
|--------:|-------------:|----------------------:|
|     0.1 |       0.0151 |                0.0042 |
|     0.3 |       0.0433 |                0.0041 |
|     0.5 |       0.0712 |                0.0112 |
|     0.7 |       0.1025 |                0.016  |
|     1   |       0.1337 |                0.039  |
|     1.5 |       0.1851 |                0.0832 |
|     2   |       0.2449 |                0.1195 |
|     3   |       0.3352 |                0.2099 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AFL-CV-1SE | 0.9922 | 0.1546 | 1.0000 | 0.0151 | 0.9849 | 1.0000 | 0.9980 |
| 0.1 | 10.0 | AdaptiveLassoCV-1SE | 0.9310 | 66.9887 | 0.8830 | 0.0042 | 0.9958 | 0.8830 | 0.1628 |
| 0.3 | 3.33 | AFL-CV-1SE | 0.9773 | 0.2660 | 1.0000 | 0.0433 | 0.9567 | 1.0000 | 0.9966 |
| 0.3 | 3.33 | AdaptiveLassoCV-1SE | 0.9268 | 67.2210 | 0.8760 | 0.0041 | 0.9959 | 0.8760 | 0.1601 |
| 0.5 | 2.0 | AFL-CV-1SE | 0.9621 | 0.4469 | 1.0000 | 0.0712 | 0.9288 | 1.0000 | 0.9943 |
| 0.5 | 2.0 | AdaptiveLassoCV-1SE | 0.9115 | 67.5376 | 0.8540 | 0.0112 | 0.9888 | 0.8540 | 0.1569 |
| 0.7 | 1.43 | AFL-CV-1SE | 0.9446 | 0.7423 | 1.0000 | 0.1025 | 0.8975 | 1.0000 | 0.9906 |
| 0.7 | 1.43 | AdaptiveLassoCV-1SE | 0.8967 | 67.7761 | 0.8340 | 0.0160 | 0.9840 | 0.8340 | 0.1556 |
| 1.0 | 1.0 | AFL-CV-1SE | 0.9269 | 1.3825 | 1.0000 | 0.1337 | 0.8663 | 1.0000 | 0.9826 |
| 1.0 | 1.0 | AdaptiveLassoCV-1SE | 0.8614 | 68.3858 | 0.7930 | 0.0390 | 0.9610 | 0.7930 | 0.1526 |
| 1.5 | 0.67 | AFL-CV-1SE | 0.8955 | 3.0341 | 0.9990 | 0.1851 | 0.8149 | 0.9990 | 0.9623 |
| 1.5 | 0.67 | AdaptiveLassoCV-1SE | 0.7889 | 69.8222 | 0.7050 | 0.0832 | 0.9168 | 0.7050 | 0.1461 |
| 2.0 | 0.5 | AFL-CV-1SE | 0.8492 | 5.4668 | 0.9770 | 0.2449 | 0.7551 | 0.9770 | 0.9332 |
| 2.0 | 0.5 | AdaptiveLassoCV-1SE | 0.7092 | 71.9614 | 0.6060 | 0.1195 | 0.8805 | 0.6060 | 0.1360 |
| 3.0 | 0.33 | AFL-CV-1SE | 0.7376 | 12.5561 | 0.8410 | 0.3352 | 0.6648 | 0.8410 | 0.8548 |
| 3.0 | 0.33 | AdaptiveLassoCV-1SE | 0.5787 | 77.4151 | 0.4680 | 0.2099 | 0.7901 | 0.4680 | 0.1183 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-1SE (F1=0.9107)
- rank_2: AdaptiveLassoCV-1SE (F1=0.8255)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-1SE (MSE=3.0062)
- rank_2: AdaptiveLassoCV-1SE (MSE=69.6385)

## 7. Key Findings

1. **Best F1**: AFL-CV-1SE with F1=0.9107
2. **Best MSE**: AFL-CV-1SE with MSE=3.0062

3. **SNR Sensitivity**:
   - AFL-CV-1SE: F1 drop = 0.1838 (high SNR to low SNR)
   - AdaptiveLassoCV-1SE: F1 drop = 0.2791 (high SNR to low SNR)

---
*Report generated: 2026-03-28T15:18:49.168341*
