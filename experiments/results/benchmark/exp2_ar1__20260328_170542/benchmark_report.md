# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-28
**Experiment**: exp2_ar1
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 10 per configuration

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
| AFL-CV-1SE | 0.7312 | 0.2444 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-1SE | 33.5600 | 48.7892 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.7640 | 0.3022 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.2667 | 0.1721 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       0.9621 |
|     1   |       0.9269 |
|     2   |       0.8492 |
|     5   |       0.581  |
|    10   |       0.337  |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       0.4469 |
|     1   |       1.3825 |
|     2   |       5.4668 |
|     5   |      33.9387 |
|    10   |     126.565  |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |        1     |
|     1   |        1     |
|     2   |        0.977 |
|     5   |        0.582 |
|    10   |        0.261 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       0.0712 |
|     1   |       0.1337 |
|     2   |       0.2449 |
|     5   |       0.4024 |
|    10   |       0.4813 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AFL-CV-1SE | 0.9621 | 0.4469 | 1.0000 | 0.0712 | 0.9288 | 1.0000 | 0.9943 |
| 1.0 | 1.0 | AFL-CV-1SE | 0.9269 | 1.3825 | 1.0000 | 0.1337 | 0.8663 | 1.0000 | 0.9826 |
| 2.0 | 0.5 | AFL-CV-1SE | 0.8492 | 5.4668 | 0.9770 | 0.2449 | 0.7551 | 0.9770 | 0.9332 |
| 5.0 | 0.2 | AFL-CV-1SE | 0.5810 | 33.9387 | 0.5820 | 0.4024 | 0.5976 | 0.5820 | 0.6668 |
| 10.0 | 0.1 | AFL-CV-1SE | 0.3370 | 126.5652 | 0.2610 | 0.4813 | 0.5187 | 0.2610 | 0.2696 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-1SE (F1=0.7312)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-1SE (MSE=33.5600)

## 7. Key Findings

1. **Best F1**: AFL-CV-1SE with F1=0.7312
2. **Best MSE**: AFL-CV-1SE with MSE=33.5600

3. **SNR Sensitivity**:
   - AFL-CV-1SE: F1 drop = 0.3731 (high SNR to low SNR)

---
*Report generated: 2026-03-28T20:16:52.607607*
