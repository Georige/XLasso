# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-27
**Experiment**: exp1_afl_cv_compare
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 10.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AFL-CV** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AFL-CV | 0.8815 | 0.1308 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AFL-CV | 4.1933 | 4.6225 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AFL-CV | 0.9663 | 0.0596 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFL-CV | 0.1755 | 0.1780 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV |
|--------:|---------:|
|     0.1 |   1      |
|     0.5 |   0.9961 |
|     1   |   0.9516 |
|     1.5 |   0.8939 |
|     2   |   0.8051 |
|     3   |   0.6425 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV |
|--------:|---------:|
|     0.1 |   0.3611 |
|     0.5 |   0.6431 |
|     1   |   1.5777 |
|     1.5 |   3.2714 |
|     2   |   5.8995 |
|     3   |  13.4069 |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV |
|--------:|---------:|
|     0.1 |    1     |
|     0.5 |    1     |
|     1   |    0.998 |
|     1.5 |    0.994 |
|     2   |    0.96  |
|     3   |    0.846 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV |
|--------:|---------:|
|     0.1 |   0      |
|     0.5 |   0.0076 |
|     1   |   0.0853 |
|     1.5 |   0.1823 |
|     2   |   0.2999 |
|     3   |   0.4777 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AFL-CV | 1.0000 | 0.3611 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9983 |
| 0.5 | 2.0 | AFL-CV | 0.9961 | 0.6431 | 1.0000 | 0.0076 | 0.9924 | 1.0000 | 0.9970 |
| 1.0 | 1.0 | AFL-CV | 0.9516 | 1.5777 | 0.9980 | 0.0853 | 0.9147 | 0.9980 | 0.9927 |
| 1.5 | 0.67 | AFL-CV | 0.8939 | 3.2714 | 0.9940 | 0.1823 | 0.8177 | 0.9940 | 0.9849 |
| 2.0 | 0.5 | AFL-CV | 0.8051 | 5.8995 | 0.9600 | 0.2999 | 0.7001 | 0.9600 | 0.9729 |
| 3.0 | 0.33 | AFL-CV | 0.6425 | 13.4069 | 0.8460 | 0.4777 | 0.5223 | 0.8460 | 0.9400 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV (F1=0.8815)

### 6.2 By MSE (lower is better)

- rank_1: AFL-CV (MSE=4.1933)

## 7. Key Findings

1. **Best F1**: AFL-CV with F1=0.8815
2. **Best MSE**: AFL-CV with MSE=4.1933

3. **SNR Sensitivity**:
   - AFL-CV: F1 drop = 0.2743 (high SNR to low SNR)

---
*Report generated: 2026-03-27T11:52:12.001343*
