# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-27
**Experiment**: exp2_afl_cv_tuned
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
| AdaptiveFlippedLassoCV | 0.7071 | 0.0961 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 3.6498 | 4.2326 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.9813 | 0.0420 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.4285 | 0.1117 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.1 |                   0.7481 |
|     0.5 |                   0.8225 |
|     1   |                   0.7524 |
|     1.5 |                   0.7097 |
|     2   |                   0.6349 |
|     3   |                   0.5753 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.1 |                   0.0665 |
|     0.5 |                   0.3537 |
|     1   |                   1.2753 |
|     1.5 |                   2.8821 |
|     2   |                   5.2595 |
|     3   |                  12.0618 |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.1 |                    1     |
|     0.5 |                    1     |
|     1   |                    1     |
|     1.5 |                    0.998 |
|     2   |                    0.99  |
|     3   |                    0.9   |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.1 |                   0.3483 |
|     0.5 |                   0.2965 |
|     1   |                   0.3893 |
|     1.5 |                   0.4421 |
|     2   |                   0.5263 |
|     3   |                   0.5686 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AdaptiveFlippedLassoCV | 0.7481 | 0.0665 | 1.0000 | 0.3483 | 0.6517 | 1.0000 | 0.9991 |
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8225 | 0.3537 | 1.0000 | 0.2965 | 0.7035 | 1.0000 | 0.9954 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.7524 | 1.2753 | 1.0000 | 0.3893 | 0.6107 | 1.0000 | 0.9835 |
| 1.5 | 0.67 | AdaptiveFlippedLassoCV | 0.7097 | 2.8821 | 0.9980 | 0.4421 | 0.5579 | 0.9980 | 0.9631 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.6349 | 5.2595 | 0.9900 | 0.5263 | 0.4737 | 0.9900 | 0.9340 |
| 3.0 | 0.33 | AdaptiveFlippedLassoCV | 0.5753 | 12.0618 | 0.9000 | 0.5686 | 0.4314 | 0.9000 | 0.8563 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.7071)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=3.6498)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.7071
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=3.6498

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.1802 (high SNR to low SNR)

---
*Report generated: 2026-03-27T08:40:58.552104*
