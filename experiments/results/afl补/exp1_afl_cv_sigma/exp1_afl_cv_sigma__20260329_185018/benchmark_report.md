# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp1_afl_cv_sigma
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 2 per configuration

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
| AdaptiveFlippedLassoCV | 0.5722 | 0.3583 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 37.1806 | 53.9671 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.7667 | 0.3615 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.5064 | 0.3596 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.9424 |
|     2   |                   0.5807 |
|    10   |                   0.1936 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.3272 |
|     2   |                   5.2349 |
|    10   |                 105.98   |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                      1   |
|     2   |                      1   |
|    10   |                      0.3 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.1071 |
|     2   |                   0.5655 |
|    10   |                   0.8466 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.9424 | 0.3272 | 1.0000 | 0.1071 | 0.8929 | 1.0000 | 0.9984 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.5807 | 5.2349 | 1.0000 | 0.5655 | 0.4345 | 1.0000 | 0.9741 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.1936 | 105.9797 | 0.3000 | 0.8466 | 0.1534 | 0.3000 | 0.6189 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.5722)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=37.1806)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.5722
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=37.1806

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.5552 (high SNR to low SNR)

---
*Report generated: 2026-03-29T18:53:42.478406*
