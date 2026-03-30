# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-28
**Experiment**: exp3_twin
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
| AFL-CV-1SE | 0.7001 | 0.1393 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-1SE | 65.6143 | 95.2024 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.8847 | 0.2305 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.3634 | 0.1122 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       0.8414 |
|     1   |       0.786  |
|     2   |       0.7164 |
|     5   |       0.6725 |
|    10   |       0.4843 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       1.8682 |
|     1   |       3.351  |
|     2   |       9.9943 |
|     5   |      65.6662 |
|    10   |     247.192  |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       1      |
|     1   |       1      |
|     2   |       0.9995 |
|     5   |       0.9735 |
|    10   |       0.4505 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       0.2659 |
|     1   |       0.3461 |
|     2   |       0.4369 |
|     5   |       0.4804 |
|    10   |       0.2875 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AFL-CV-1SE | 0.8414 | 1.8682 | 1.0000 | 0.2659 | 0.7341 | 1.0000 | 0.9904 |
| 1.0 | 1.0 | AFL-CV-1SE | 0.7860 | 3.3510 | 1.0000 | 0.3461 | 0.6539 | 1.0000 | 0.9829 |
| 2.0 | 0.5 | AFL-CV-1SE | 0.7164 | 9.9943 | 0.9995 | 0.4369 | 0.5631 | 0.9995 | 0.9496 |
| 5.0 | 0.2 | AFL-CV-1SE | 0.6725 | 65.6662 | 0.9735 | 0.4804 | 0.5196 | 0.9735 | 0.7031 |
| 10.0 | 0.1 | AFL-CV-1SE | 0.4843 | 247.1920 | 0.4505 | 0.2875 | 0.6925 | 0.4505 | 0.1582 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-1SE (F1=0.7001)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-1SE (MSE=65.6143)

## 7. Key Findings

1. **Best F1**: AFL-CV-1SE with F1=0.7001
2. **Best MSE**: AFL-CV-1SE with MSE=65.6143

3. **SNR Sensitivity**:
   - AFL-CV-1SE: F1 drop = 0.2170 (high SNR to low SNR)

---
*Report generated: 2026-03-28T19:19:28.359054*
