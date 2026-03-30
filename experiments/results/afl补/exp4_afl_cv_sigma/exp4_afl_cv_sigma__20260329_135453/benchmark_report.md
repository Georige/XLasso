# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp4_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.7723 | 0.1317 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 35.1193 | 52.4153 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.9504 | 0.0966 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.3329 | 0.1481 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8554 |
|     1   |                   0.8612 |
|     2   |                   0.8261 |
|     5   |                   0.7557 |
|    10   |                   0.563  |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.6405 |
|     1   |                   1.5478 |
|     2   |                   5.2198 |
|     5   |                  32.3465 |
|    10   |                 135.842  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                    1     |
|     1   |                    1     |
|     2   |                    1     |
|     5   |                    0.987 |
|    10   |                    0.765 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2331 |
|     1   |                   0.2332 |
|     2   |                   0.2852 |
|     5   |                   0.378  |
|    10   |                   0.5349 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8554 | 0.6405 | 1.0000 | 0.2331 | 0.7669 | 1.0000 | 0.9962 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.8612 | 1.5478 | 1.0000 | 0.2332 | 0.7668 | 1.0000 | 0.9910 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.8261 | 5.2198 | 1.0000 | 0.2852 | 0.7148 | 1.0000 | 0.9702 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.7557 | 32.3465 | 0.9870 | 0.3780 | 0.6220 | 0.9870 | 0.8344 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.5630 | 135.8421 | 0.7650 | 0.5349 | 0.4651 | 0.7650 | 0.4895 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.7723)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=35.1193)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.7723
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=35.1193

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.1404 (high SNR to low SNR)

---
*Report generated: 2026-03-29T14:11:15.714487*
