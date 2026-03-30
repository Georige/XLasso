# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-30
**Experiment**: exp2_afl_cv_en_vs_adaptive
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AFL-CV-EN (ElasticNet)** (CV-tuned)
- **AFL-EBIC-Simple** (fixed params)
- **AdaptiveLassoCV (fixed)** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AFL-CV-EN (ElasticNet) | 0.6941 | 0.2824 |
| AdaptiveLassoCV (fixed) | 0.6516 | 0.3001 |
| AFL-EBIC-Simple | 0.3926 | 0.4060 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-EN (ElasticNet) | 42.1598 | 58.2678 |
| AdaptiveLassoCV (fixed) | 44.1720 | 60.3546 |
| AFL-EBIC-Simple | 108.4980 | 150.5892 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-EBIC-Simple | 0.8667 | 0.1970 |
| AFL-CV-EN (ElasticNet) | 0.7433 | 0.3375 |
| AdaptiveLassoCV (fixed) | 0.7133 | 0.3573 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV-EN (ElasticNet) | 0.3112 | 0.2461 |
| AdaptiveLassoCV (fixed) | 0.3770 | 0.2493 |
| AFL-EBIC-Simple | 0.6585 | 0.4107 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AFL-EBIC-Simple |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|------------------:|--------------------------:|
|     0.5 |                   0.9674 |            0.9452 |                    0.9355 |
|     2   |                   0.7778 |            0.1461 |                    0.7518 |
|    10   |                   0.3371 |            0.0867 |                    0.2675 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AFL-EBIC-Simple |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|------------------:|--------------------------:|
|     0.5 |                   0.333  |            0.8295 |                    0.9056 |
|     2   |                   5.4329 |           14.4073 |                    6.5673 |
|    10   |                 120.714  |          310.257  |                  125.043  |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AFL-EBIC-Simple |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|------------------:|--------------------------:|
|     0.5 |                     1    |              1    |                      1    |
|     2   |                     0.94 |              0.99 |                      0.9  |
|    10   |                     0.29 |              0.61 |                      0.24 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AFL-EBIC-Simple |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|------------------:|--------------------------:|
|     0.5 |                   0.061  |            0.101  |                    0.1199 |
|     2   |                   0.3221 |            0.9211 |                    0.3469 |
|    10   |                   0.5506 |            0.9533 |                    0.6643 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AFL-CV-EN (ElasticNet) | 0.9674 | 0.3330 | 1.0000 | 0.0610 | 0.9390 | 1.0000 | 0.9956 |
| 0.5 | 2.0 | AFL-EBIC-Simple | 0.9452 | 0.8295 | 1.0000 | 0.1010 | 0.8990 | 1.0000 | 0.9882 |
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.9355 | 0.9056 | 1.0000 | 0.1199 | 0.8801 | 1.0000 | 0.9872 |
| 2.0 | 0.5 | AFL-CV-EN (ElasticNet) | 0.7778 | 5.4329 | 0.9400 | 0.3221 | 0.6779 | 0.9400 | 0.9320 |
| 2.0 | 0.5 | AFL-EBIC-Simple | 0.1461 | 14.4073 | 0.9900 | 0.9211 | 0.0789 | 0.9900 | 0.8143 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.7518 | 6.5673 | 0.9000 | 0.3469 | 0.6531 | 0.9000 | 0.9170 |
| 10.0 | 0.1 | AFL-CV-EN (ElasticNet) | 0.3371 | 120.7136 | 0.2900 | 0.5506 | 0.4494 | 0.2900 | 0.3325 |
| 10.0 | 0.1 | AFL-EBIC-Simple | 0.0867 | 310.2573 | 0.6100 | 0.9533 | 0.0467 | 0.6100 | -0.7462 |
| 10.0 | 0.1 | AdaptiveLassoCV (fixed) | 0.2675 | 125.0432 | 0.2400 | 0.6643 | 0.3357 | 0.2400 | 0.3101 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-EN (ElasticNet) (F1=0.6941)
- rank_2: AdaptiveLassoCV (fixed) (F1=0.6516)
- rank_3: AFL-EBIC-Simple (F1=0.3926)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-EN (ElasticNet) (MSE=42.1598)
- rank_2: AdaptiveLassoCV (fixed) (MSE=44.1720)
- rank_3: AFL-EBIC-Simple (MSE=108.4980)

## 7. Key Findings

1. **Best F1**: AFL-CV-EN (ElasticNet) with F1=0.6941
2. **Best MSE**: AFL-CV-EN (ElasticNet) with MSE=42.1598

3. **SNR Sensitivity**:
   - AFL-CV-EN (ElasticNet): F1 drop = 0.4100 (high SNR to low SNR)
   - AFL-EBIC-Simple: F1 drop = 0.8288 (high SNR to low SNR)
   - AdaptiveLassoCV (fixed): F1 drop = 0.4258 (high SNR to low SNR)

---
*Report generated: 2026-03-30T17:07:33.319810*
