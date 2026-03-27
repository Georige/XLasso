# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-27
**Experiment**: exp1_afl_fixed_only
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 10.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AFL-Fixed** (fixed params)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AFL-Fixed | 0.8614 | 0.1462 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-Fixed | 3.9592 | 4.3352 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-Fixed | 0.9900 | 0.0242 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-Fixed | 0.2145 | 0.2123 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-Fixed |
|--------:|------------:|
|     0.1 |      1      |
|     0.5 |      1      |
|     1   |      0.9579 |
|     1.5 |      0.8385 |
|     2   |      0.7447 |
|     3   |      0.6273 |

### 4.2 MSE by Sigma

|   sigma |   AFL-Fixed |
|--------:|------------:|
|     0.1 |      0.4962 |
|     0.5 |      0.7535 |
|     1   |      1.5535 |
|     1.5 |      3.0674 |
|     2   |      5.3926 |
|     3   |     12.4921 |

### 4.3 TPR by Sigma

|   sigma |   AFL-Fixed |
|--------:|------------:|
|     0.1 |        1    |
|     0.5 |        1    |
|     1   |        1    |
|     1.5 |        1    |
|     2   |        0.99 |
|     3   |        0.95 |

### 4.4 FDR by Sigma

|   sigma |   AFL-Fixed |
|--------:|------------:|
|     0.1 |      0      |
|     0.5 |      0      |
|     1   |      0.0792 |
|     1.5 |      0.2763 |
|     2   |      0.4009 |
|     3   |      0.5307 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AFL-Fixed | 1.0000 | 0.4962 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9977 |
| 0.5 | 2.0 | AFL-Fixed | 1.0000 | 0.7535 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9965 |
| 1.0 | 1.0 | AFL-Fixed | 0.9579 | 1.5535 | 1.0000 | 0.0792 | 0.9208 | 1.0000 | 0.9929 |
| 1.5 | 0.67 | AFL-Fixed | 0.8385 | 3.0674 | 1.0000 | 0.2763 | 0.7237 | 1.0000 | 0.9861 |
| 2.0 | 0.5 | AFL-Fixed | 0.7447 | 5.3926 | 0.9900 | 0.4009 | 0.5991 | 0.9900 | 0.9760 |
| 3.0 | 0.33 | AFL-Fixed | 0.6273 | 12.4921 | 0.9500 | 0.5307 | 0.4693 | 0.9500 | 0.9465 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-Fixed (F1=0.8614)

### 6.2 By MSE (lower is better)

- rank_1: AFL-Fixed (MSE=3.9592)

## 7. Key Findings

1. **Best F1**: AFL-Fixed with F1=0.8614
2. **Best MSE**: AFL-Fixed with MSE=3.9592

3. **SNR Sensitivity**:
   - AFL-Fixed: F1 drop = 0.3140 (high SNR to low SNR)

---
*Report generated: 2026-03-27T12:25:16.158526*
