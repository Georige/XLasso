# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp6_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.6005 | 0.2725 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 26.0297 | 40.1245 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.8000 | 0.3202 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.4521 | 0.2593 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8347 |
|     1   |                   0.7681 |
|     2   |                   0.7072 |
|     5   |                   0.4837 |
|    10   |                   0.2087 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.276  |
|     1   |                   1.0119 |
|     2   |                   4.0262 |
|     5   |                  26.6414 |
|    10   |                  98.193  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                     1    |
|     1   |                     1    |
|     2   |                     1    |
|     5   |                     0.78 |
|    10   |                     0.22 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2539 |
|     1   |                   0.3539 |
|     2   |                   0.4439 |
|     5   |                   0.6301 |
|    10   |                   0.5789 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8347 | 0.2760 | 1.0000 | 0.2539 | 0.7461 | 1.0000 | 0.9811 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.7681 | 1.0119 | 1.0000 | 0.3539 | 0.6461 | 1.0000 | 0.9333 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.7072 | 4.0262 | 1.0000 | 0.4439 | 0.5561 | 1.0000 | 0.7750 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.4837 | 26.6414 | 0.7800 | 0.6301 | 0.3699 | 0.7800 | 0.2750 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.2087 | 98.1930 | 0.2200 | 0.5789 | 0.2211 | 0.2200 | 0.0462 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.6005)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=26.0297)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.6005
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=26.0297

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.3682 (high SNR to low SNR)

---
*Report generated: 2026-03-29T18:29:34.781291*
