# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-28
**Experiment**: exp7_ar1_rho5
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
| AFL-CV-1SE | 0.6773 | 0.3105 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-1SE | 35.2825 | 50.6356 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.7602 | 0.3630 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.2415 | 0.1436 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       0.9324 |
|     1   |       0.8908 |
|     2   |       0.8149 |
|     5   |       0.64   |
|    10   |       0.1084 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       0.3711 |
|     1   |       1.33   |
|     2   |       5.4338 |
|     5   |      37.9888 |
|    10   |     131.289  |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |        1     |
|     1   |        1     |
|     2   |        0.998 |
|     5   |        0.719 |
|    10   |        0.084 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       0.1223 |
|     1   |       0.1908 |
|     2   |       0.3015 |
|     5   |       0.3861 |
|    10   |       0.2067 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AFL-CV-1SE | 0.9324 | 0.3711 | 1.0000 | 0.1223 | 0.8777 | 1.0000 | 0.9882 |
| 1.0 | 1.0 | AFL-CV-1SE | 0.8908 | 1.3300 | 1.0000 | 0.1908 | 0.8092 | 1.0000 | 0.9591 |
| 2.0 | 0.5 | AFL-CV-1SE | 0.8149 | 5.4338 | 0.9980 | 0.3015 | 0.6985 | 0.9980 | 0.8486 |
| 5.0 | 0.2 | AFL-CV-1SE | 0.6400 | 37.9888 | 0.7190 | 0.3861 | 0.6139 | 0.7190 | 0.3402 |
| 10.0 | 0.1 | AFL-CV-1SE | 0.1084 | 131.2888 | 0.0840 | 0.2067 | 0.2333 | 0.0840 | 0.0016 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-1SE (F1=0.6773)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-1SE (MSE=35.2825)

## 7. Key Findings

1. **Best F1**: AFL-CV-1SE with F1=0.6773
2. **Best MSE**: AFL-CV-1SE with MSE=35.2825

3. **SNR Sensitivity**:
   - AFL-CV-1SE: F1 drop = 0.4113 (high SNR to low SNR)

---
*Report generated: 2026-03-28T19:52:46.200640*
