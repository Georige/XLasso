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
| AFL-CV-EN (ElasticNet) | 0.6810 | 0.3012 |
| AdaptiveLassoCV (fixed) | 0.6649 | 0.2992 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveLassoCV (fixed) | 44.9089 | 61.3704 |
| AFL-CV-EN (ElasticNet) | 46.8190 | 56.9200 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveLassoCV (fixed) | 0.7133 | 0.3637 |
| AFL-CV-EN (ElasticNet) | 0.6267 | 0.2884 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV-EN (ElasticNet) | 0.2351 | 0.3143 |
| AdaptiveLassoCV (fixed) | 0.3416 | 0.2424 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   0.951  |                    0.9539 |
|     2   |                   0.797  |                    0.7548 |
|    10   |                   0.2949 |                    0.2859 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   6.311  |                    0.9106 |
|     2   |                  10.9621 |                    6.3838 |
|    10   |                 123.184  |                  127.432  |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                     0.91 |                      1    |
|     2   |                     0.7  |                      0.91 |
|    10   |                     0.27 |                      0.23 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   0      |                    0.0855 |
|     2   |                   0.0713 |                    0.3392 |
|    10   |                   0.6342 |                    0.6    |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AFL-CV-EN (ElasticNet) | 0.9510 | 6.3110 | 0.9100 | 0.0000 | 1.0000 | 0.9100 | 0.9181 |
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.9539 | 0.9106 | 1.0000 | 0.0855 | 0.9145 | 1.0000 | 0.9872 |
| 2.0 | 0.5 | AFL-CV-EN (ElasticNet) | 0.7970 | 10.9621 | 0.7000 | 0.0713 | 0.9287 | 0.7000 | 0.8627 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.7548 | 6.3838 | 0.9100 | 0.3392 | 0.6608 | 0.9100 | 0.9192 |
| 10.0 | 0.1 | AFL-CV-EN (ElasticNet) | 0.2949 | 123.1839 | 0.2700 | 0.6342 | 0.3658 | 0.2700 | 0.3167 |
| 10.0 | 0.1 | AdaptiveLassoCV (fixed) | 0.2859 | 127.4323 | 0.2300 | 0.6000 | 0.4000 | 0.2300 | 0.2969 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-EN (ElasticNet) (F1=0.6810)
- rank_2: AdaptiveLassoCV (fixed) (F1=0.6649)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveLassoCV (fixed) (MSE=44.9089)
- rank_2: AFL-CV-EN (ElasticNet) (MSE=46.8190)

## 7. Key Findings

1. **Best F1**: AFL-CV-EN (ElasticNet) with F1=0.6810
2. **Best MSE**: AdaptiveLassoCV (fixed) with MSE=44.9089

3. **SNR Sensitivity**:
   - AFL-CV-EN (ElasticNet): F1 drop = 0.4051 (high SNR to low SNR)
   - AdaptiveLassoCV (fixed): F1 drop = 0.4336 (high SNR to low SNR)

---
*Report generated: 2026-03-30T21:17:48.294205*
