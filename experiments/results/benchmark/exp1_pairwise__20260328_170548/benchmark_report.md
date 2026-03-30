# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-28
**Experiment**: exp1_pairwise
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 10 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AFL-CV-1SE** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AFL-CV-1SE | 0.6475 | 0.3266 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV-1SE | 34.7109 | 49.0382 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.7236 | 0.3454 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV-1SE | 0.3989 | 0.3181 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       0.9893 |
|     1   |       0.938  |
|     2   |       0.7578 |
|     5   |       0.3872 |
|    10   |       0.1654 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       0.6903 |
|     1   |       1.6267 |
|     2   |       6.2823 |
|     5   |      37.311  |
|    10   |     127.644  |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |        1     |
|     1   |        1     |
|     2   |        0.96  |
|     5   |        0.498 |
|    10   |        0.16  |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-1SE |
|--------:|-------------:|
|     0.5 |       0.0202 |
|     1   |       0.1117 |
|     2   |       0.3662 |
|     5   |       0.6765 |
|    10   |       0.8199 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.5 | 2.0 | AFL-CV-1SE | 0.9893 | 0.6903 | 1.0000 | 0.0202 | 0.9798 | 1.0000 | 0.9966 |
| 1.0 | 1.0 | AFL-CV-1SE | 0.9380 | 1.6267 | 1.0000 | 0.1117 | 0.8883 | 1.0000 | 0.9920 |
| 2.0 | 0.5 | AFL-CV-1SE | 0.7578 | 6.2823 | 0.9600 | 0.3662 | 0.6338 | 0.9600 | 0.9696 |
| 5.0 | 0.2 | AFL-CV-1SE | 0.3872 | 37.3110 | 0.4980 | 0.6765 | 0.3235 | 0.4980 | 0.8352 |
| 10.0 | 0.1 | AFL-CV-1SE | 0.1654 | 127.6440 | 0.1600 | 0.8199 | 0.1801 | 0.1600 | 0.5736 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-1SE (F1=0.6475)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV-1SE (MSE=34.7109)

## 7. Key Findings

1. **Best F1**: AFL-CV-1SE with F1=0.6475
2. **Best MSE**: AFL-CV-1SE with MSE=34.7109

3. **SNR Sensitivity**:
   - AFL-CV-1SE: F1 drop = 0.5525 (high SNR to low SNR)

---
*Report generated: 2026-03-28T21:03:38.150435*
