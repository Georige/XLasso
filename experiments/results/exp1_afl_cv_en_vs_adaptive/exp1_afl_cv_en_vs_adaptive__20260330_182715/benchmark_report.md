# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-30
**Experiment**: exp1_afl_cv_en_vs_adaptive
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
| AFL-CV-EN (ElasticNet) | 0.6627 | 0.3632 |
| AdaptiveLassoCV (fixed) | 0.6345 | 0.3486 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-EN (ElasticNet) | 40.5453 | 55.9096 |
| AdaptiveLassoCV (fixed) | 41.2379 | 56.5522 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-EN (ElasticNet) | 0.7500 | 0.3606 |
| AdaptiveLassoCV (fixed) | 0.7200 | 0.3990 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV-EN (ElasticNet) | 0.3822 | 0.3722 |
| AdaptiveLassoCV (fixed) | 0.4078 | 0.3296 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   1      |                    0.9765 |
|     2   |                   0.7909 |                    0.7257 |
|    10   |                   0.1974 |                    0.2011 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   0.5162 |                    0.7479 |
|     2   |                   5.4415 |                    5.8217 |
|    10   |                 115.678  |                  117.144  |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                     1    |                      1    |
|     2   |                     0.99 |                      0.98 |
|    10   |                     0.26 |                      0.18 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   0      |                    0.0443 |
|     2   |                   0.3212 |                    0.4087 |
|    10   |                   0.8255 |                    0.7704 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AFL-CV-EN (ElasticNet) | 1.0000 | 0.5162 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9976 |
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.9765 | 0.7479 | 1.0000 | 0.0443 | 0.9557 | 1.0000 | 0.9964 |
| 2.0 | 0.5 | AFL-CV-EN (ElasticNet) | 0.7909 | 5.4415 | 0.9900 | 0.3212 | 0.6788 | 0.9900 | 0.9745 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.7257 | 5.8217 | 0.9800 | 0.4087 | 0.5913 | 0.9800 | 0.9726 |
| 10.0 | 0.1 | AFL-CV-EN (ElasticNet) | 0.1974 | 115.6781 | 0.2600 | 0.8255 | 0.1745 | 0.2600 | 0.6140 |
| 10.0 | 0.1 | AdaptiveLassoCV (fixed) | 0.2011 | 117.1442 | 0.1800 | 0.7704 | 0.2296 | 0.1800 | 0.6075 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-EN (ElasticNet) (F1=0.6627)
- rank_2: AdaptiveLassoCV (fixed) (F1=0.6345)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-EN (ElasticNet) (MSE=40.5453)
- rank_2: AdaptiveLassoCV (fixed) (MSE=41.2379)

## 7. Key Findings

1. **Best F1**: AFL-CV-EN (ElasticNet) with F1=0.6627
2. **Best MSE**: AFL-CV-EN (ElasticNet) with MSE=40.5453

3. **SNR Sensitivity**:
   - AFL-CV-EN (ElasticNet): F1 drop = 0.5059 (high SNR to low SNR)
   - AdaptiveLassoCV (fixed): F1 drop = 0.5131 (high SNR to low SNR)

---
*Report generated: 2026-03-30T20:05:28.941630*
