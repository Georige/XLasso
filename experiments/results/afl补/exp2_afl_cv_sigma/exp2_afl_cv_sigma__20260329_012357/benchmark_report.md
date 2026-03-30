# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp2_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.6535 | 0.2085 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 31.1160 | 45.2224 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.8024 | 0.2620 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.4343 | 0.1783 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8559 |
|     1   |                   0.8411 |
|     2   |                   0.7244 |
|     5   |                   0.5116 |
|    10   |                   0.3345 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.343  |
|     1   |                   1.2098 |
|     2   |                   5.0006 |
|     5   |                  31.8425 |
|    10   |                 117.184  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                    1     |
|     1   |                    1     |
|     2   |                    0.989 |
|     5   |                    0.666 |
|    10   |                    0.357 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2378 |
|     1   |                   0.267  |
|     2   |                   0.4227 |
|     5   |                   0.5764 |
|    10   |                   0.6676 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8559 | 0.3430 | 1.0000 | 0.2378 | 0.7622 | 1.0000 | 0.9956 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.8411 | 1.2098 | 1.0000 | 0.2670 | 0.7330 | 1.0000 | 0.9847 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.7244 | 5.0006 | 0.9890 | 0.4227 | 0.5773 | 0.9890 | 0.9389 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.5116 | 31.8425 | 0.6660 | 0.5764 | 0.4236 | 0.6660 | 0.6861 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.3345 | 117.1841 | 0.3570 | 0.6676 | 0.3324 | 0.3570 | 0.3197 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.6535)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=31.1160)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.6535
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=31.1160

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.3325 (high SNR to low SNR)

---
*Report generated: 2026-03-29T01:38:23.229492*
