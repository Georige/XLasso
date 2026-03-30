# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp4_afl_cv_sigma
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 10 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AdaptiveFlippedLassoCV** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AdaptiveFlippedLassoCV | 0.8229 | 0.1358 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 34.3363 | 51.6215 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.9504 | 0.0966 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.2576 | 0.1562 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8985 |
|     1   |                   0.9144 |
|     2   |                   0.8895 |
|     5   |                   0.8224 |
|    10   |                   0.5896 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.6253 |
|     1   |                   1.4857 |
|     2   |                   4.9768 |
|     5   |                  30.8847 |
|    10   |                 133.709  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                    1     |
|     1   |                    1     |
|     2   |                    1     |
|     5   |                    0.986 |
|    10   |                    0.766 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.1673 |
|     1   |                   0.1486 |
|     2   |                   0.1906 |
|     5   |                   0.2839 |
|    10   |                   0.4973 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8985 | 0.6253 | 1.0000 | 0.1673 | 0.8327 | 1.0000 | 0.9963 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.9144 | 1.4857 | 1.0000 | 0.1486 | 0.8514 | 1.0000 | 0.9914 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.8895 | 4.9768 | 1.0000 | 0.1906 | 0.8094 | 1.0000 | 0.9716 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.8224 | 30.8847 | 0.9860 | 0.2839 | 0.7161 | 0.9860 | 0.8414 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.5896 | 133.7093 | 0.7660 | 0.4973 | 0.5027 | 0.7660 | 0.4963 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.8229)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=34.3363)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.8229
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=34.3363

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.1313 (high SNR to low SNR)

---
*Report generated: 2026-03-29T04:00:01.453835*
