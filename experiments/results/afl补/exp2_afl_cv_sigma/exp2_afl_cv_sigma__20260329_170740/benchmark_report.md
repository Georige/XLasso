# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp2_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.6287 | 0.2190 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 31.2615 | 46.0756 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.7880 | 0.2747 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.4640 | 0.1929 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.856  |
|     1   |                   0.8076 |
|     2   |                   0.6762 |
|     5   |                   0.4716 |
|    10   |                   0.3319 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.3408 |
|     1   |                   1.2411 |
|     2   |                   5.4311 |
|     5   |                  31.8804 |
|    10   |                 117.414  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                     1    |
|     1   |                     1    |
|     2   |                     0.97 |
|     5   |                     0.62 |
|    10   |                     0.35 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2503 |
|     1   |                   0.3216 |
|     2   |                   0.4697 |
|     5   |                   0.6148 |
|    10   |                   0.6635 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8560 | 0.3408 | 1.0000 | 0.2503 | 0.7497 | 1.0000 | 0.9955 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.8076 | 1.2411 | 1.0000 | 0.3216 | 0.6784 | 1.0000 | 0.9837 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.6762 | 5.4311 | 0.9700 | 0.4697 | 0.5303 | 0.9700 | 0.9318 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.4716 | 31.8804 | 0.6200 | 0.6148 | 0.3852 | 0.6200 | 0.6876 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.3319 | 117.4139 | 0.3500 | 0.6635 | 0.3365 | 0.3500 | 0.3515 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.6287)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=31.2615)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.6287
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=31.2615

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.3628 (high SNR to low SNR)

---
*Report generated: 2026-03-29T17:12:07.205456*
