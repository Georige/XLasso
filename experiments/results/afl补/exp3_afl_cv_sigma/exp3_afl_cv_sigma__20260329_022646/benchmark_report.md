# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp3_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.7966 | 0.1772 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 29.8625 | 43.8881 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.8498 | 0.2082 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.2289 | 0.1504 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.9157 |
|     1   |                   0.9319 |
|     2   |                   0.922  |
|     5   |                   0.7082 |
|    10   |                   0.5053 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.37   |
|     1   |                   1.2008 |
|     2   |                   4.576  |
|     5   |                  29.4235 |
|    10   |                 113.742  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                    1     |
|     1   |                    1     |
|     2   |                    0.999 |
|     5   |                    0.769 |
|    10   |                    0.481 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.1344 |
|     1   |                   0.1156 |
|     2   |                   0.1357 |
|     5   |                   0.3223 |
|    10   |                   0.4364 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.9157 | 0.3700 | 1.0000 | 0.1344 | 0.8656 | 1.0000 | 0.9972 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.9319 | 1.2008 | 1.0000 | 0.1156 | 0.8844 | 1.0000 | 0.9911 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.9220 | 4.5760 | 0.9990 | 0.1357 | 0.8643 | 0.9990 | 0.9667 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.7082 | 29.4235 | 0.7690 | 0.3223 | 0.6777 | 0.7690 | 0.8151 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.5053 | 113.7424 | 0.4810 | 0.4364 | 0.5636 | 0.4810 | 0.5124 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.7966)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=29.8625)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.7966
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=29.8625

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.2039 (high SNR to low SNR)

---
*Report generated: 2026-03-29T02:38:13.962487*
