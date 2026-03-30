# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-29
**Experiment**: exp7_afl_cv_sigma
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
| AdaptiveFlippedLassoCV | 0.1476 | 0.1177 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 27.6191 | 38.3854 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.1706 | 0.2256 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.4443 | 0.2191 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.3202 |
|     1   |                   0.2092 |
|     2   |                   0.123  |
|     5   |                   0.0704 |
|    10   |                   0.015  |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   1.4789 |
|     1   |                   2.7882 |
|     2   |                   5.7959 |
|     5   |                  27.1661 |
|    10   |                 100.866  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                    0.56  |
|     1   |                    0.162 |
|     2   |                    0.075 |
|     5   |                    0.046 |
|    10   |                    0.01  |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.6124 |
|     1   |                   0.3465 |
|     2   |                   0.2977 |
|     5   |                   0.5102 |
|    10   |                   0.4547 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.3202 | 1.4789 | 0.5600 | 0.6124 | 0.3876 | 0.5600 | 0.5171 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.2092 | 2.7882 | 0.1620 | 0.3465 | 0.6535 | 0.1620 | 0.2580 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.1230 | 5.7959 | 0.0750 | 0.2977 | 0.7023 | 0.0750 | 0.1201 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.0704 | 27.1661 | 0.0460 | 0.5102 | 0.2698 | 0.0460 | -0.0216 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.0150 | 100.8662 | 0.0100 | 0.4547 | 0.0453 | 0.0100 | -0.0249 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.1476)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=27.6191)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.1476
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=27.6191

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.2507 (high SNR to low SNR)

---
*Report generated: 2026-03-29T04:36:28.709675*
