# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp6_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.7049 | 0.2994 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 30.0362 | 43.9874 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.7948 | 0.3106 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.2841 | 0.2252 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.9696 |
|     1   |                   0.9401 |
|     2   |                   0.8546 |
|     5   |                   0.5537 |
|    10   |                   0.2067 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2862 |
|     1   |                   1.0873 |
|     2   |                   4.4585 |
|     5   |                  30.3296 |
|    10   |                 114.019  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                    1     |
|     1   |                    1     |
|     2   |                    0.998 |
|     5   |                    0.752 |
|    10   |                    0.224 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.0532 |
|     1   |                   0.1023 |
|     2   |                   0.2319 |
|     5   |                   0.5113 |
|    10   |                   0.522  |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.9696 | 0.2862 | 1.0000 | 0.0532 | 0.9468 | 1.0000 | 0.9801 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.9401 | 1.0873 | 1.0000 | 0.1023 | 0.8977 | 1.0000 | 0.9282 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.8546 | 4.4585 | 0.9980 | 0.2319 | 0.7681 | 0.9980 | 0.7543 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.5537 | 30.3296 | 0.7520 | 0.5113 | 0.4887 | 0.7520 | 0.2208 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.2067 | 114.0191 | 0.2240 | 0.5220 | 0.2980 | 0.2240 | -0.0151 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.7049)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=30.0362)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.7049
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=30.0362

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.4313 (high SNR to low SNR)

---
*Report generated: 2026-03-29T03:47:53.772381*
