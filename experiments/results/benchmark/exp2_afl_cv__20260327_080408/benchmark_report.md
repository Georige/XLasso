# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-27
**Experiment**: exp2_afl_cv
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
| AdaptiveFlippedLassoCV | 0.7202 | 0.1115 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 3.8940 | 4.1858 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.9773 | 0.0445 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.3952 | 0.1300 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.1 |                   0.7718 |
|     0.5 |                   0.8236 |
|     1   |                   0.6979 |
|     1.5 |                   0.7508 |
|     2   |                   0.6878 |
|     3   |                   0.5894 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.1 |                   0.066  |
|     0.5 |                   1.668  |
|     1   |                   1.5008 |
|     1.5 |                   2.8706 |
|     2   |                   5.2068 |
|     3   |                  12.052  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.1 |                    1     |
|     0.5 |                    0.98  |
|     1   |                    1     |
|     1.5 |                    0.998 |
|     2   |                    0.986 |
|     3   |                    0.9   |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.1 |                   0.3117 |
|     0.5 |                   0.2489 |
|     1   |                   0.4138 |
|     1.5 |                   0.3824 |
|     2   |                   0.464  |
|     3   |                   0.5502 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AdaptiveFlippedLassoCV | 0.7718 | 0.0660 | 1.0000 | 0.3117 | 0.6883 | 1.0000 | 0.9991 |
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8236 | 1.6680 | 0.9800 | 0.2489 | 0.7511 | 0.9800 | 0.9807 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.6979 | 1.5008 | 1.0000 | 0.4138 | 0.5862 | 1.0000 | 0.9804 |
| 1.5 | 0.67 | AdaptiveFlippedLassoCV | 0.7508 | 2.8706 | 0.9980 | 0.3824 | 0.6176 | 0.9980 | 0.9631 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.6878 | 5.2068 | 0.9860 | 0.4640 | 0.5360 | 0.9860 | 0.9345 |
| 3.0 | 0.33 | AdaptiveFlippedLassoCV | 0.5894 | 12.0520 | 0.9000 | 0.5502 | 0.4498 | 0.9000 | 0.8565 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.7202)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=3.8940)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.7202
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=3.8940

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.1591 (high SNR to low SNR)

---
*Report generated: 2026-03-27T08:11:13.269000*
