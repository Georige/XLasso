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
| AdaptiveFlippedLassoCV | 0.6434 | 0.1880 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 30.8622 | 44.8349 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.8180 | 0.2490 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.4559 | 0.1553 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8277 |
|     1   |                   0.8071 |
|     2   |                   0.7053 |
|     5   |                   0.5267 |
|    10   |                   0.3501 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.3479 |
|     1   |                   1.2241 |
|     2   |                   5.0283 |
|     5   |                  31.4746 |
|    10   |                 116.236  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                    1     |
|     1   |                    1     |
|     2   |                    0.995 |
|     5   |                    0.713 |
|    10   |                    0.382 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2817 |
|     1   |                   0.3165 |
|     2   |                   0.4456 |
|     5   |                   0.574  |
|    10   |                   0.6616 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8277 | 0.3479 | 1.0000 | 0.2817 | 0.7183 | 1.0000 | 0.9956 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.8071 | 1.2241 | 1.0000 | 0.3165 | 0.6835 | 1.0000 | 0.9845 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.7053 | 5.0283 | 0.9950 | 0.4456 | 0.5544 | 0.9950 | 0.9387 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.5267 | 31.4746 | 0.7130 | 0.5740 | 0.4260 | 0.7130 | 0.6897 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.3501 | 116.2359 | 0.3820 | 0.6616 | 0.3384 | 0.3820 | 0.3256 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.6434)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=30.8622)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.6434
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=30.8622

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.3004 (high SNR to low SNR)

---
*Report generated: 2026-03-29T13:41:13.929166*
