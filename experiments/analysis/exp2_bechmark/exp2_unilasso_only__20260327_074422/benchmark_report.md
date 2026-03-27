# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-27
**Experiment**: exp2_unilasso_only
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **UniLassoCV** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| UniLassoCV | 0.7922 | 0.0648 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| UniLassoCV | 6.4762 | 7.0607 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| UniLassoCV | 0.9233 | 0.1072 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| UniLassoCV | 0.3008 | 0.0550 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   UniLassoCV |
|--------:|-------------:|
|     0.1 |       0.8265 |
|     0.5 |       0.8329 |
|     1   |       0.8263 |
|     1.5 |       0.8119 |
|     2   |       0.7741 |
|     3   |       0.6813 |

### 4.2 MSE by Sigma

|   sigma |   UniLassoCV |
|--------:|-------------:|
|     0.1 |       0.2879 |
|     0.5 |       0.6433 |
|     1   |       2.4573 |
|     1.5 |       5.61   |
|     2   |       9.6964 |
|     3   |      20.1626 |

### 4.3 TPR by Sigma

|   sigma |   UniLassoCV |
|--------:|-------------:|
|     0.1 |        1     |
|     0.5 |        1     |
|     1   |        0.988 |
|     1.5 |        0.958 |
|     2   |        0.886 |
|     3   |        0.708 |

### 4.4 FDR by Sigma

|   sigma |   UniLassoCV |
|--------:|-------------:|
|     0.1 |       0.2929 |
|     0.5 |       0.2846 |
|     1   |       0.2882 |
|     1.5 |       0.2934 |
|     2   |       0.3095 |
|     3   |       0.3361 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | UniLassoCV | 0.8265 | 0.2879 | 1.0000 | 0.2929 | 0.7071 | 1.0000 | 0.9962 |
| 0.5 | 2.0 | UniLassoCV | 0.8329 | 0.6433 | 1.0000 | 0.2846 | 0.7154 | 1.0000 | 0.9915 |
| 1.0 | 1.0 | UniLassoCV | 0.8263 | 2.4573 | 0.9880 | 0.2882 | 0.7118 | 0.9880 | 0.9678 |
| 1.5 | 0.67 | UniLassoCV | 0.8119 | 5.6100 | 0.9580 | 0.2934 | 0.7066 | 0.9580 | 0.9280 |
| 2.0 | 0.5 | UniLassoCV | 0.7741 | 9.6964 | 0.8860 | 0.3095 | 0.6905 | 0.8860 | 0.8780 |
| 3.0 | 0.33 | UniLassoCV | 0.6813 | 20.1626 | 0.7080 | 0.3361 | 0.6639 | 0.7080 | 0.7604 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: UniLassoCV (F1=0.7922)

### 6.2 By MSE (lower is better)

- rank_1: UniLassoCV (MSE=6.4762)

## 7. Key Findings

1. **Best F1**: UniLassoCV with F1=0.7922
2. **Best MSE**: UniLassoCV with MSE=6.4762

3. **SNR Sensitivity**:
   - UniLassoCV: F1 drop = 0.1020 (high SNR to low SNR)

---
*Report generated: 2026-03-27T07:51:46.113178*
