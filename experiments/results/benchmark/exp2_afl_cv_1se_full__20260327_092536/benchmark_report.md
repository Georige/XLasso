# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-27
**Experiment**: exp2_afl_cv_1se_full
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AFL-CV-1SE** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AFL-CV-1SE | 0.8744 | 0.0937 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-1SE | 3.9159 | 4.4803 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.9657 | 0.0710 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.1938 | 0.1191 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.1 |       0.9913 |
|     0.5 |       0.9456 |
|     1   |       0.9013 |
|     1.5 |       0.8633 |
|     2   |       0.8346 |
|     3   |       0.7103 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.1 |       0.1377 |
|     0.5 |       0.4299 |
|     1   |       1.403  |
|     1.5 |       3.0924 |
|     2   |       5.6097 |
|     3   |      12.8229 |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.1 |        1     |
|     0.5 |        1     |
|     1   |        1     |
|     1.5 |        0.998 |
|     2   |        0.976 |
|     3   |        0.82  |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.1 |       0.0168 |
|     0.5 |       0.1012 |
|     1   |       0.1781 |
|     1.5 |       0.237  |
|     2   |       0.2678 |
|     3   |       0.362  |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AFL-CV-1SE | 0.9913 | 0.1377 | 1.0000 | 0.0168 | 0.9832 | 1.0000 | 0.9982 |
| 0.5 | 2.0 | AFL-CV-1SE | 0.9456 | 0.4299 | 1.0000 | 0.1012 | 0.8988 | 1.0000 | 0.9944 |
| 1.0 | 1.0 | AFL-CV-1SE | 0.9013 | 1.4030 | 1.0000 | 0.1781 | 0.8219 | 1.0000 | 0.9818 |
| 1.5 | 0.67 | AFL-CV-1SE | 0.8633 | 3.0924 | 0.9980 | 0.2370 | 0.7630 | 0.9980 | 0.9604 |
| 2.0 | 0.5 | AFL-CV-1SE | 0.8346 | 5.6097 | 0.9760 | 0.2678 | 0.7322 | 0.9760 | 0.9295 |
| 3.0 | 0.33 | AFL-CV-1SE | 0.7103 | 12.8229 | 0.8200 | 0.3620 | 0.6380 | 0.8200 | 0.8472 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-1SE (F1=0.8744)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-1SE (MSE=3.9159)

## 7. Key Findings

1. **Best F1**: AFL-CV-1SE with F1=0.8744
2. **Best MSE**: AFL-CV-1SE with MSE=3.9159

3. **SNR Sensitivity**:
   - AFL-CV-1SE: F1 drop = 0.1960 (high SNR to low SNR)

---
*Report generated: 2026-03-27T09:30:56.916803*
