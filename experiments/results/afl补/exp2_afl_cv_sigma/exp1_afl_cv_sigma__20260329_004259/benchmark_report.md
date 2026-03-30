# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp1_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.5793 | 0.2692 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 32.1783 | 45.6373 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.7724 | 0.3050 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.5188 | 0.2554 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8911 |
|     1   |                   0.8195 |
|     2   |                   0.6151 |
|     5   |                   0.372  |
|    10   |                   0.1988 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.5383 |
|     1   |                   1.4385 |
|     2   |                   5.7268 |
|     5   |                  34.6029 |
|    10   |                 118.585  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                    1     |
|     1   |                    1     |
|     2   |                    0.989 |
|     5   |                    0.624 |
|    10   |                    0.249 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.1864 |
|     1   |                   0.2965 |
|     2   |                   0.5481 |
|     5   |                   0.7317 |
|    10   |                   0.8315 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8911 | 0.5383 | 1.0000 | 0.1864 | 0.8136 | 1.0000 | 0.9974 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.8195 | 1.4385 | 1.0000 | 0.2965 | 0.7035 | 1.0000 | 0.9930 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.6151 | 5.7268 | 0.9890 | 0.5481 | 0.4519 | 0.9890 | 0.9724 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.3720 | 34.6029 | 0.6240 | 0.7317 | 0.2683 | 0.6240 | 0.8469 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.1988 | 118.5850 | 0.2490 | 0.8315 | 0.1685 | 0.2490 | 0.6024 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.5793)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=32.1783)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.5793
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=32.1783

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.4959 (high SNR to low SNR)

---
*Report generated: 2026-03-29T01:14:57.122453*
