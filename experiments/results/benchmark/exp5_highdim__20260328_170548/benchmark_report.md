# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-28
**Experiment**: exp5_highdim
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
| AFL-CV-1SE | 0.8259 | 0.0369 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-1SE | 2.1938 | 0.3034 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.9993 | 0.0021 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.2941 | 0.0562 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|       1 |       0.8259 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|       1 |       2.1938 |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|       1 |       0.9993 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|       1 |       0.2941 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 1.0 | 1.0 | AFL-CV-1SE | 0.8259 | 2.1938 | 0.9993 | 0.2941 | 0.7059 | 0.9993 | 0.9813 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-1SE (F1=0.8259)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-1SE (MSE=2.1938)

## 7. Key Findings

1. **Best F1**: AFL-CV-1SE with F1=0.8259
2. **Best MSE**: AFL-CV-1SE with MSE=2.1938

---
*Report generated: 2026-03-28T17:58:02.490645*
