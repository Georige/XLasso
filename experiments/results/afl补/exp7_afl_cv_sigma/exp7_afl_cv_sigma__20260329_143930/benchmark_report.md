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
| AdaptiveFlippedLassoCV | 0.1362 | 0.1048 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 27.6241 | 38.4187 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.1808 | 0.2501 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.4627 | 0.2334 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.2773 |
|     1   |                   0.2031 |
|     2   |                   0.124  |
|     5   |                   0.0607 |
|    10   |                   0.0161 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   1.4145 |
|     1   |                   2.7961 |
|     2   |                   5.8267 |
|     5   |                  27.1518 |
|    10   |                 100.932  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                    0.616 |
|     1   |                    0.158 |
|     2   |                    0.077 |
|     5   |                    0.042 |
|    10   |                    0.011 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.6737 |
|     1   |                   0.3898 |
|     2   |                   0.406  |
|     5   |                   0.5093 |
|    10   |                   0.3347 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.2773 | 1.4145 | 0.6160 | 0.6737 | 0.3263 | 0.6160 | 0.5389 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.2031 | 2.7961 | 0.1580 | 0.3898 | 0.6102 | 0.1580 | 0.2570 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.1240 | 5.8267 | 0.0770 | 0.4060 | 0.5940 | 0.0770 | 0.1157 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.0607 | 27.1518 | 0.0420 | 0.5093 | 0.1707 | 0.0420 | -0.0214 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.0161 | 100.9316 | 0.0110 | 0.3347 | 0.0453 | 0.0110 | -0.0251 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.1362)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=27.6241)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.1362
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=27.6241

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.2104 (high SNR to low SNR)

---
*Report generated: 2026-03-29T15:46:35.102871*
