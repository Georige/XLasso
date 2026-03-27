# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-27
**Experiment**: exp3_elasticnet_relaxed
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **ElasticNet-1SE** (CV-tuned)
- **RelaxedLasso-1SE** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| RelaxedLasso-1SE | 0.9662 | 0.0245 |
| ElasticNet-1SE | 0.9608 | 0.0269 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| RelaxedLasso-1SE | 3.0248 | 3.4868 |
| ElasticNet-1SE | 3.4072 | 3.9368 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| ElasticNet-1SE | 1.0000 | 0.0000 |
| RelaxedLasso-1SE | 1.0000 | 0.0000 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| RelaxedLasso-1SE | 0.0625 | 0.0439 |
| ElasticNet-1SE | 0.0717 | 0.0472 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   ElasticNet-1SE |   RelaxedLasso-1SE |
|--------:|-----------------:|-------------------:|
|     0.1 |           0.9646 |             0.9646 |
|     0.5 |           0.9637 |             0.9637 |
|     1   |           0.9643 |             0.9643 |
|     1.5 |           0.9677 |             0.9696 |
|     2   |           0.9608 |             0.9662 |
|     3   |           0.9437 |             0.9687 |

### 4.2 MSE by Sigma

|   sigma |   ElasticNet-1SE |   RelaxedLasso-1SE |
|--------:|-----------------:|-------------------:|
|     0.1 |           0.0124 |             0.0109 |
|     0.5 |           0.3089 |             0.2739 |
|     1   |           1.2377 |             1.103  |
|     1.5 |           2.7832 |             2.4605 |
|     2   |           4.9572 |             4.4051 |
|     3   |          11.1441 |             9.8955 |

### 4.3 TPR by Sigma

|   sigma |   ElasticNet-1SE |   RelaxedLasso-1SE |
|--------:|-----------------:|-------------------:|
|     0.1 |                1 |                  1 |
|     0.5 |                1 |                  1 |
|     1   |                1 |                  1 |
|     1.5 |                1 |                  1 |
|     2   |                1 |                  1 |
|     3   |                1 |                  1 |

### 4.4 FDR by Sigma

|   sigma |   ElasticNet-1SE |   RelaxedLasso-1SE |
|--------:|-----------------:|-------------------:|
|     0.1 |           0.0652 |             0.0652 |
|     0.5 |           0.0669 |             0.0669 |
|     1   |           0.0659 |             0.0659 |
|     1.5 |           0.0601 |             0.0565 |
|     2   |           0.0721 |             0.0625 |
|     3   |           0.1002 |             0.0579 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | ElasticNet-1SE | 0.9646 | 0.0124 | 1.0000 | 0.0652 | 0.9348 | 1.0000 | 0.9999 |
| 0.1 | 10.0 | RelaxedLasso-1SE | 0.9646 | 0.0109 | 1.0000 | 0.0652 | 0.9348 | 1.0000 | 0.9999 |
| 0.5 | 2.0 | ElasticNet-1SE | 0.9637 | 0.3089 | 1.0000 | 0.0669 | 0.9331 | 1.0000 | 0.9973 |
| 0.5 | 2.0 | RelaxedLasso-1SE | 0.9637 | 0.2739 | 1.0000 | 0.0669 | 0.9331 | 1.0000 | 0.9976 |
| 1.0 | 1.0 | ElasticNet-1SE | 0.9643 | 1.2377 | 1.0000 | 0.0659 | 0.9341 | 1.0000 | 0.9893 |
| 1.0 | 1.0 | RelaxedLasso-1SE | 0.9643 | 1.1030 | 1.0000 | 0.0659 | 0.9341 | 1.0000 | 0.9904 |
| 1.5 | 0.67 | ElasticNet-1SE | 0.9677 | 2.7832 | 1.0000 | 0.0601 | 0.9399 | 1.0000 | 0.9763 |
| 1.5 | 0.67 | RelaxedLasso-1SE | 0.9696 | 2.4605 | 1.0000 | 0.0565 | 0.9435 | 1.0000 | 0.9789 |
| 2.0 | 0.5 | ElasticNet-1SE | 0.9608 | 4.9572 | 1.0000 | 0.0721 | 0.9279 | 1.0000 | 0.9586 |
| 2.0 | 0.5 | RelaxedLasso-1SE | 0.9662 | 4.4051 | 1.0000 | 0.0625 | 0.9375 | 1.0000 | 0.9629 |
| 3.0 | 0.33 | ElasticNet-1SE | 0.9437 | 11.1441 | 1.0000 | 0.1002 | 0.8998 | 1.0000 | 0.9116 |
| 3.0 | 0.33 | RelaxedLasso-1SE | 0.9687 | 9.8955 | 1.0000 | 0.0579 | 0.9421 | 1.0000 | 0.9206 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: RelaxedLasso-1SE (F1=0.9662)
- rank_2: ElasticNet-1SE (F1=0.9608)

### 6.2 By MSE (lower is better)

- rank_1: RelaxedLasso-1SE (MSE=3.0248)
- rank_2: ElasticNet-1SE (MSE=3.4072)

## 7. Key Findings

1. **Best F1**: RelaxedLasso-1SE with F1=0.9662
2. **Best MSE**: RelaxedLasso-1SE with MSE=3.0248

3. **SNR Sensitivity**:
   - ElasticNet-1SE: F1 drop = 0.0119 (high SNR to low SNR)
   - RelaxedLasso-1SE: F1 drop = -0.0033 (high SNR to low SNR)

---
*Report generated: 2026-03-27T13:37:50.820452*
