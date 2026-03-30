# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp5_afl_cv_sigma
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

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
| AdaptiveFlippedLassoCV | 0.0092 | 0.0190 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 32.3461 | 45.2977 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.0100 | 0.0207 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.3915 | 0.4965 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.0098 |
|     2   |                   0.0089 |
|    10   |                   0.0089 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.4399 |
|     2   |                   3.9813 |
|    10   |                  92.6171 |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                     0.01 |
|     2   |                     0.01 |
|    10   |                     0.01 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.3905 |
|     2   |                   0.392  |
|    10   |                   0.392  |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.0098 | 0.4399 | 0.0100 | 0.3905 | 0.0095 | 0.0100 | -1.1740 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.0089 | 3.9813 | 0.0100 | 0.3920 | 0.0080 | 0.0100 | -0.1246 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.0089 | 92.6171 | 0.0100 | 0.3920 | 0.0080 | 0.0100 | -0.0340 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.0092)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=32.3461)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.0092
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=32.3461

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.0009 (high SNR to low SNR)

---
*Report generated: 2026-03-29T18:27:12.061200*
