# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-30
**Experiment**: exp2_afl_cv_sigma_eps
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AFL-CV (eps=1e-5)** (CV-tuned)
- **AFL-CV (eps=1e-10)** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AFL-CV (eps=1e-10) | 0.6194 | 0.2499 |
| AFL-CV (eps=1e-5) | 0.6194 | 0.2499 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV (eps=1e-5) | 41.1546 | 57.0565 |
| AFL-CV (eps=1e-10) | 41.1546 | 57.0565 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV (eps=1e-10) | 0.7567 | 0.3262 |
| AFL-CV (eps=1e-5) | 0.7567 | 0.3262 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV (eps=1e-10) | 0.4616 | 0.2138 |
| AFL-CV (eps=1e-5) | 0.4616 | 0.2138 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV (eps=1e-10) |   AFL-CV (eps=1e-5) |
|--------:|---------------------:|--------------------:|
|     0.5 |               0.8474 |              0.8474 |
|     2   |               0.7018 |              0.7018 |
|    10   |               0.309  |              0.309  |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV (eps=1e-10) |   AFL-CV (eps=1e-5) |
|--------:|---------------------:|--------------------:|
|     0.5 |               0.3475 |              0.3475 |
|     2   |               5.3979 |              5.3979 |
|    10   |             117.718  |            117.718  |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV (eps=1e-10) |   AFL-CV (eps=1e-5) |
|--------:|---------------------:|--------------------:|
|     0.5 |                 1    |                1    |
|     2   |                 0.95 |                0.95 |
|    10   |                 0.32 |                0.32 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV (eps=1e-10) |   AFL-CV (eps=1e-5) |
|--------:|---------------------:|--------------------:|
|     0.5 |               0.2612 |              0.2612 |
|     2   |               0.4369 |              0.4369 |
|    10   |               0.6867 |              0.6867 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AFL-CV (eps=1e-5) | 0.8474 | 0.3475 | 1.0000 | 0.2612 | 0.7388 | 1.0000 | 0.9954 |
| 0.5 | 2.0 | AFL-CV (eps=1e-10) | 0.8474 | 0.3475 | 1.0000 | 0.2612 | 0.7388 | 1.0000 | 0.9954 |
| 2.0 | 0.5 | AFL-CV (eps=1e-5) | 0.7018 | 5.3979 | 0.9500 | 0.4369 | 0.5631 | 0.9500 | 0.9322 |
| 2.0 | 0.5 | AFL-CV (eps=1e-10) | 0.7018 | 5.3979 | 0.9500 | 0.4369 | 0.5631 | 0.9500 | 0.9322 |
| 10.0 | 0.1 | AFL-CV (eps=1e-5) | 0.3090 | 117.7183 | 0.3200 | 0.6867 | 0.3133 | 0.3200 | 0.3515 |
| 10.0 | 0.1 | AFL-CV (eps=1e-10) | 0.3090 | 117.7183 | 0.3200 | 0.6867 | 0.3133 | 0.3200 | 0.3515 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV (eps=1e-10) (F1=0.6194)
- rank_2: AFL-CV (eps=1e-5) (F1=0.6194)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV (eps=1e-5) (MSE=41.1546)
- rank_2: AFL-CV (eps=1e-10) (MSE=41.1546)

## 7. Key Findings

1. **Best F1**: AFL-CV (eps=1e-10) with F1=0.6194
2. **Best MSE**: AFL-CV (eps=1e-5) with MSE=41.1546

3. **SNR Sensitivity**:
   - AFL-CV (eps=1e-5): F1 drop = 0.3419 (high SNR to low SNR)
   - AFL-CV (eps=1e-10): F1 drop = 0.3419 (high SNR to low SNR)

---
*Report generated: 2026-03-30T00:12:38.293490*
