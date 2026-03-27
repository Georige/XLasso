# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-27
**Experiment**: exp2_afl_cv_1se_compare
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
| AFL-CV-1SE | 0.9685 | 0.0259 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.2838 | 0.1559 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-1SE | 1.0000 | 0.0000 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.0590 | 0.0477 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.1 |       0.9913 |
|     0.5 |       0.9456 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.1 |       0.1377 |
|     0.5 |       0.4299 |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.1 |            1 |
|     0.5 |            1 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.1 |       0.0168 |
|     0.5 |       0.1012 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AFL-CV-1SE | 0.9913 | 0.1377 | 1.0000 | 0.0168 | 0.9832 | 1.0000 | 0.9982 |
| 0.5 | 2.0 | AFL-CV-1SE | 0.9456 | 0.4299 | 1.0000 | 0.1012 | 0.8988 | 1.0000 | 0.9944 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-1SE (F1=0.9685)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-1SE (MSE=0.2838)

## 7. Key Findings

1. **Best F1**: AFL-CV-1SE with F1=0.9685
2. **Best MSE**: AFL-CV-1SE with MSE=0.2838

---
*Report generated: 2026-03-27T09:12:01.167302*
