# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-28
**Experiment**: exp6_pairwise_rho8_test
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 3 per configuration

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
| AFL-CV-1SE | 0.6324 | 0.0721 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-1SE | 6.9741 | 0.7657 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.7500 | 0.0889 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.4425 | 0.0611 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|       2 |       0.6324 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|       2 |       6.9741 |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|       2 |         0.75 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|       2 |       0.4425 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 2.0 | 0.5 | AFL-CV-1SE | 0.6324 | 6.9741 | 0.7500 | 0.4425 | 0.5575 | 0.7500 | 0.9800 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-1SE (F1=0.6324)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-1SE (MSE=6.9741)

## 7. Key Findings

1. **Best F1**: AFL-CV-1SE with F1=0.6324
2. **Best MSE**: AFL-CV-1SE with MSE=6.9741

---
*Report generated: 2026-03-28T23:40:39.516206*
