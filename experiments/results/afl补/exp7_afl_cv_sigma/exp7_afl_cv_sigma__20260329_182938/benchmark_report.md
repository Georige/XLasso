# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp7_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.1338 | 0.1219 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 28.9726 | 41.9663 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.1580 | 0.2357 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.5803 | 0.3808 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2787 |
|     1   |                   0.212  |
|     2   |                   0.149  |
|     5   |                   0.0292 |
|    10   |                   0      |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   1.9246 |
|     1   |                   3.0082 |
|     2   |                   6.0916 |
|     5   |                  28.1013 |
|    10   |                 105.737  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                     0.49 |
|     1   |                     0.18 |
|     2   |                     0.1  |
|     5   |                     0.02 |
|    10   |                     0    |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.6782 |
|     1   |                   0.5613 |
|     2   |                   0.5178 |
|     5   |                   0.5444 |
|    10   |                   0.6    |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.2787 | 1.9246 | 0.4900 | 0.6782 | 0.3218 | 0.4900 | 0.3354 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.2120 | 3.0082 | 0.1800 | 0.5613 | 0.4387 | 0.1800 | 0.1668 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.1490 | 6.0916 | 0.1000 | 0.5178 | 0.4822 | 0.1000 | 0.0736 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.0292 | 28.1013 | 0.0200 | 0.5444 | 0.0556 | 0.0200 | -0.0193 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.0000 | 105.7372 | 0.0000 | 0.6000 | 0.0000 | 0.0000 | -0.0256 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.1338)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=28.9726)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.1338
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=28.9726

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.2193 (high SNR to low SNR)

---
*Report generated: 2026-03-29T18:44:43.596963*
