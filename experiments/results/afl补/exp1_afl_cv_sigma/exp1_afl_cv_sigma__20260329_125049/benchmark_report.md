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
| AdaptiveFlippedLassoCV | 0.5788 | 0.2690 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 31.8774 | 45.1682 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.7846 | 0.2916 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveFlippedLassoCV | 0.5217 | 0.2613 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.8986 |
|     1   |                   0.8201 |
|     2   |                   0.5949 |
|     5   |                   0.3697 |
|    10   |                   0.2108 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.541  |
|     1   |                   1.4523 |
|     2   |                   5.7461 |
|     5   |                  34.1997 |
|    10   |                 117.448  |

### 4.3 TPR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                    1     |
|     1   |                    1     |
|     2   |                    0.992 |
|     5   |                    0.65  |
|    10   |                    0.281 |

### 4.4 FDR by Sigma

|   sigma |   AdaptiveFlippedLassoCV |
|--------:|-------------------------:|
|     0.5 |                   0.1775 |
|     1   |                   0.2941 |
|     2   |                   0.5688 |
|     5   |                   0.7396 |
|    10   |                   0.8287 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AdaptiveFlippedLassoCV | 0.8986 | 0.5410 | 1.0000 | 0.1775 | 0.8225 | 1.0000 | 0.9974 |
| 1.0 | 1.0 | AdaptiveFlippedLassoCV | 0.8201 | 1.4523 | 1.0000 | 0.2941 | 0.7059 | 1.0000 | 0.9929 |
| 2.0 | 0.5 | AdaptiveFlippedLassoCV | 0.5949 | 5.7461 | 0.9920 | 0.5688 | 0.4312 | 0.9920 | 0.9723 |
| 5.0 | 0.2 | AdaptiveFlippedLassoCV | 0.3697 | 34.1997 | 0.6500 | 0.7396 | 0.2604 | 0.6500 | 0.8489 |
| 10.0 | 0.1 | AdaptiveFlippedLassoCV | 0.2108 | 117.4480 | 0.2810 | 0.8287 | 0.1713 | 0.2810 | 0.6061 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveFlippedLassoCV (F1=0.5788)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveFlippedLassoCV (MSE=31.8774)

## 7. Key Findings

1. **Best F1**: AdaptiveFlippedLassoCV with F1=0.5788
2. **Best MSE**: AdaptiveFlippedLassoCV with MSE=31.8774

3. **SNR Sensitivity**:
   - AdaptiveFlippedLassoCV: F1 drop = 0.5068 (high SNR to low SNR)

---
*Report generated: 2026-03-29T13:25:17.001820*
