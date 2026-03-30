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
- **AdaptiveLassoCV (fixed)** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AFL-CV-EN (ElasticNet) | 0.6753 | 0.2742 |
| AdaptiveLassoCV (fixed) | 0.6649 | 0.2992 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-EN (ElasticNet) | 41.8243 | 57.6095 |
| AdaptiveLassoCV (fixed) | 44.9089 | 61.3704 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-EN (ElasticNet) | 0.7500 | 0.3349 |
| AdaptiveLassoCV (fixed) | 0.7133 | 0.3637 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveLassoCV (fixed) | 0.3416 | 0.2424 |
| AFL-CV-EN (ElasticNet) | 0.3431 | 0.2435 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   0.9546 |                    0.9539 |
|     2   |                   0.7334 |                    0.7548 |
|    10   |                   0.338  |                    0.2859 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   0.3305 |                    0.9106 |
|     2   |                   5.4052 |                    6.3838 |
|    10   |                 119.737  |                  127.432  |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                     1    |                      1    |
|     2   |                     0.95 |                      0.91 |
|    10   |                     0.3  |                      0.23 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   0.0834 |                    0.0855 |
|     2   |                   0.3881 |                    0.3392 |
|    10   |                   0.5577 |                    0.6    |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AFL-CV-EN (ElasticNet) | 0.9546 | 0.3305 | 1.0000 | 0.0834 | 0.9166 | 1.0000 | 0.9956 |
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.9539 | 0.9106 | 1.0000 | 0.0855 | 0.9145 | 1.0000 | 0.9872 |
| 2.0 | 0.5 | AFL-CV-EN (ElasticNet) | 0.7334 | 5.4052 | 0.9500 | 0.3881 | 0.6119 | 0.9500 | 0.9325 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.7548 | 6.3838 | 0.9100 | 0.3392 | 0.6608 | 0.9100 | 0.9192 |
| 10.0 | 0.1 | AFL-CV-EN (ElasticNet) | 0.3380 | 119.7372 | 0.3000 | 0.5577 | 0.4423 | 0.3000 | 0.3354 |
| 10.0 | 0.1 | AdaptiveLassoCV (fixed) | 0.2859 | 127.4323 | 0.2300 | 0.6000 | 0.4000 | 0.2300 | 0.2969 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-EN (ElasticNet) (F1=0.6753)
- rank_2: AdaptiveLassoCV (fixed) (F1=0.6649)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-EN (ElasticNet) (MSE=41.8243)
- rank_2: AdaptiveLassoCV (fixed) (MSE=44.9089)

## 7. Key Findings

1. **Best F1**: AFL-CV-EN (ElasticNet) with F1=0.6753
2. **Best MSE**: AFL-CV-EN (ElasticNet) with MSE=41.8243

3. **SNR Sensitivity**:
   - AFL-CV-EN (ElasticNet): F1 drop = 0.4189 (high SNR to low SNR)
   - AdaptiveLassoCV (fixed): F1 drop = 0.4336 (high SNR to low SNR)

---
*Report generated: 2026-03-30T19:22:26.232215*
