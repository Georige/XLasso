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
| AFL-CV-EN (ElasticNet) | 0.6959 | 0.2811 |
| AdaptiveLassoCV (fixed) | 0.6516 | 0.3001 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-EN (ElasticNet) | 41.9082 | 57.7662 |
| AdaptiveLassoCV (fixed) | 44.1720 | 60.3546 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-EN (ElasticNet) | 0.7467 | 0.3341 |
| AdaptiveLassoCV (fixed) | 0.7133 | 0.3573 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV-EN (ElasticNet) | 0.3107 | 0.2455 |
| AdaptiveLassoCV (fixed) | 0.3770 | 0.2493 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   0.9674 |                    0.9355 |
|     2   |                   0.7782 |                    0.7518 |
|    10   |                   0.3422 |                    0.2675 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   0.333  |                    0.9056 |
|     2   |                   5.3537 |                    6.5673 |
|    10   |                 120.038  |                  125.043  |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                     1    |                      1    |
|     2   |                     0.94 |                      0.9  |
|    10   |                     0.3  |                      0.24 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-EN (ElasticNet) |   AdaptiveLassoCV (fixed) |
|--------:|-------------------------:|--------------------------:|
|     0.5 |                   0.061  |                    0.1199 |
|     2   |                   0.3223 |                    0.3469 |
|    10   |                   0.5489 |                    0.6643 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AFL-CV-EN (ElasticNet) | 0.9674 | 0.3330 | 1.0000 | 0.0610 | 0.9390 | 1.0000 | 0.9956 |
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.9355 | 0.9056 | 1.0000 | 0.1199 | 0.8801 | 1.0000 | 0.9872 |
| 2.0 | 0.5 | AFL-CV-EN (ElasticNet) | 0.7782 | 5.3537 | 0.9400 | 0.3223 | 0.6777 | 0.9400 | 0.9331 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.7518 | 6.5673 | 0.9000 | 0.3469 | 0.6531 | 0.9000 | 0.9170 |
| 10.0 | 0.1 | AFL-CV-EN (ElasticNet) | 0.3422 | 120.0377 | 0.3000 | 0.5489 | 0.4511 | 0.3000 | 0.3340 |
| 10.0 | 0.1 | AdaptiveLassoCV (fixed) | 0.2675 | 125.0432 | 0.2400 | 0.6643 | 0.3357 | 0.2400 | 0.3101 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-EN (ElasticNet) (F1=0.6959)
- rank_2: AdaptiveLassoCV (fixed) (F1=0.6516)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-EN (ElasticNet) (MSE=41.9082)
- rank_2: AdaptiveLassoCV (fixed) (MSE=44.1720)

## 7. Key Findings

1. **Best F1**: AFL-CV-EN (ElasticNet) with F1=0.6959
2. **Best MSE**: AFL-CV-EN (ElasticNet) with MSE=41.9082

3. **SNR Sensitivity**:
   - AFL-CV-EN (ElasticNet): F1 drop = 0.4072 (high SNR to low SNR)
   - AdaptiveLassoCV (fixed): F1 drop = 0.4258 (high SNR to low SNR)

---
*Report generated: 2026-03-30T18:21:37.608613*
