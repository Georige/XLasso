# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-28
**Experiment**: exp4_block
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
| AFL-CV-1SE | 0.8689 | 0.1815 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-1SE | 32.1258 | 47.0602 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.8244 | 0.2429 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.0295 | 0.0522 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       1      |
|     1   |       0.999  |
|     2   |       0.993  |
|     5   |       0.8104 |
|    10   |       0.5421 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       0.6304 |
|     1   |       1.4906 |
|     2   |       5.1536 |
|     5   |      31.4526 |
|    10   |     121.902  |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |        1     |
|     1   |        1     |
|     2   |        0.998 |
|     5   |        0.726 |
|    10   |        0.398 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       0      |
|     1   |       0.0018 |
|     2   |       0.0111 |
|     5   |       0.0569 |
|    10   |       0.0778 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AFL-CV-1SE | 1.0000 | 0.6304 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9954 |
| 1.0 | 1.0 | AFL-CV-1SE | 0.9990 | 1.4906 | 1.0000 | 0.0018 | 0.9982 | 1.0000 | 0.9890 |
| 2.0 | 0.5 | AFL-CV-1SE | 0.9930 | 5.1536 | 0.9980 | 0.0111 | 0.9889 | 0.9980 | 0.9627 |
| 5.0 | 0.2 | AFL-CV-1SE | 0.8104 | 31.4526 | 0.7260 | 0.0569 | 0.9431 | 0.7260 | 0.8013 |
| 10.0 | 0.1 | AFL-CV-1SE | 0.5421 | 121.9019 | 0.3980 | 0.0778 | 0.9222 | 0.3980 | 0.4720 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-1SE (F1=0.8689)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-1SE (MSE=32.1258)

## 7. Key Findings

1. **Best F1**: AFL-CV-1SE with F1=0.8689
2. **Best MSE**: AFL-CV-1SE with MSE=32.1258

3. **SNR Sensitivity**:
   - AFL-CV-1SE: F1 drop = 0.2182 (high SNR to low SNR)

---
*Report generated: 2026-03-28T20:08:44.184104*
