# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-27
**Experiment**: exp3_lasso_compare
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 10.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AFL-CV-1SE** (CV-tuned)
- **Lasso-CV** (CV-tuned)
- **AdaptiveLasso-CV** (CV-tuned)
- **UniLasso-CV** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AFL-CV-1SE | 0.9934 | 0.0130 |
| UniLasso-CV | 0.9895 | 0.0309 |
| AdaptiveLasso-CV | 0.8047 | 0.1152 |
| Lasso-CV | 0.7517 | 0.0712 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| Lasso-CV | 3.1502 | 3.6340 |
| AFL-CV-1SE | 3.3322 | 3.7093 |
| UniLasso-CV | 4.7121 | 5.1874 |
| AdaptiveLasso-CV | 102.7299 | 11.5262 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| Lasso-CV | 1.0000 | 0.0000 |
| AdaptiveLasso-CV | 0.9997 | 0.0018 |
| AFL-CV-1SE | 0.9983 | 0.0053 |
| UniLasso-CV | 0.9917 | 0.0193 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| UniLasso-CV | 0.0088 | 0.0424 |
| AFL-CV-1SE | 0.0105 | 0.0195 |
| AdaptiveLasso-CV | 0.3056 | 0.1581 |
| Lasso-CV | 0.3839 | 0.0973 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLasso-CV |   Lasso-CV |   UniLasso-CV |
|--------:|-------------:|-------------------:|-----------:|--------------:|
|     0.1 |       1      |             0.8184 |     0.7586 |        0.9671 |
|     0.5 |       1      |             0.9491 |     0.7682 |        0.9964 |
|     1   |       1      |             0.8812 |     0.7426 |        1      |
|     1.5 |       0.999  |             0.8109 |     0.7434 |        1      |
|     2   |       0.9942 |             0.75   |     0.7517 |        0.999  |
|     3   |       0.9672 |             0.6185 |     0.7458 |        0.9748 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLasso-CV |   Lasso-CV |   UniLasso-CV |
|--------:|-------------:|-------------------:|-----------:|--------------:|
|     0.1 |       0.1919 |            98.4513 |     0.0116 |        0.4666 |
|     0.5 |       0.455  |            99.1241 |     0.2885 |        0.6823 |
|     1   |       1.274  |           100.45   |     1.1436 |        1.7073 |
|     1.5 |       2.6653 |           102.331  |     2.5715 |        3.7203 |
|     2   |       4.7042 |           104.768  |     4.5812 |        6.7168 |
|     3   |      10.7027 |           111.255  |    10.3047 |       14.9792 |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLasso-CV |   Lasso-CV |   UniLasso-CV |
|--------:|-------------:|-------------------:|-----------:|--------------:|
|     0.1 |         1    |              1     |          1 |         1     |
|     0.5 |         1    |              1     |          1 |         1     |
|     1   |         1    |              1     |          1 |         1     |
|     1.5 |         1    |              1     |          1 |         1     |
|     2   |         1    |              1     |          1 |         0.998 |
|     3   |         0.99 |              0.998 |          1 |         0.952 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLasso-CV |   Lasso-CV |   UniLasso-CV |
|--------:|-------------:|-------------------:|-----------:|--------------:|
|     0.1 |       0      |             0.3026 |     0.3697 |        0.0462 |
|     0.5 |       0      |             0.0779 |     0.3548 |        0.0067 |
|     1   |       0      |             0.2056 |     0.3991 |        0      |
|     1.5 |       0.0019 |             0.3121 |     0.3987 |        0      |
|     2   |       0.0111 |             0.3908 |     0.3866 |        0      |
|     3   |       0.0501 |             0.5443 |     0.3943 |        0      |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AFL-CV-1SE | 1.0000 | 0.1919 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9983 |
| 0.1 | 10.0 | Lasso-CV | 0.7586 | 0.0116 | 1.0000 | 0.3697 | 0.6303 | 1.0000 | 0.9999 |
| 0.1 | 10.0 | AdaptiveLasso-CV | 0.8184 | 98.4513 | 1.0000 | 0.3026 | 0.6974 | 1.0000 | 0.1510 |
| 0.1 | 10.0 | UniLasso-CV | 0.9671 | 0.4666 | 1.0000 | 0.0462 | 0.9538 | 1.0000 | 0.9958 |
| 0.5 | 2.0 | AFL-CV-1SE | 1.0000 | 0.4550 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9960 |
| 0.5 | 2.0 | Lasso-CV | 0.7682 | 0.2885 | 1.0000 | 0.3548 | 0.6452 | 1.0000 | 0.9975 |
| 0.5 | 2.0 | AdaptiveLasso-CV | 0.9491 | 99.1241 | 1.0000 | 0.0779 | 0.9221 | 1.0000 | 0.1498 |
| 0.5 | 2.0 | UniLasso-CV | 0.9964 | 0.6823 | 1.0000 | 0.0067 | 0.9933 | 1.0000 | 0.9939 |
| 1.0 | 1.0 | AFL-CV-1SE | 1.0000 | 1.2740 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9890 |
| 1.0 | 1.0 | Lasso-CV | 0.7426 | 1.1436 | 1.0000 | 0.3991 | 0.6009 | 1.0000 | 0.9901 |
| 1.0 | 1.0 | AdaptiveLasso-CV | 0.8812 | 100.4502 | 1.0000 | 0.2056 | 0.7944 | 1.0000 | 0.1475 |
| 1.0 | 1.0 | UniLasso-CV | 1.0000 | 1.7073 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9851 |
| 1.5 | 0.67 | AFL-CV-1SE | 0.9990 | 2.6653 | 1.0000 | 0.0019 | 0.9981 | 1.0000 | 0.9773 |
| 1.5 | 0.67 | Lasso-CV | 0.7434 | 2.5715 | 1.0000 | 0.3987 | 0.6013 | 1.0000 | 0.9780 |
| 1.5 | 0.67 | AdaptiveLasso-CV | 0.8109 | 102.3313 | 1.0000 | 0.3121 | 0.6879 | 1.0000 | 0.1440 |
| 1.5 | 0.67 | UniLasso-CV | 1.0000 | 3.7203 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9682 |
| 2.0 | 0.5 | AFL-CV-1SE | 0.9942 | 4.7042 | 1.0000 | 0.0111 | 0.9889 | 1.0000 | 0.9606 |
| 2.0 | 0.5 | Lasso-CV | 0.7517 | 4.5812 | 1.0000 | 0.3866 | 0.6134 | 1.0000 | 0.9616 |
| 2.0 | 0.5 | AdaptiveLasso-CV | 0.7500 | 104.7676 | 1.0000 | 0.3908 | 0.6092 | 1.0000 | 0.1395 |
| 2.0 | 0.5 | UniLasso-CV | 0.9990 | 6.7168 | 0.9980 | 0.0000 | 1.0000 | 0.9980 | 0.9438 |
| 3.0 | 0.33 | AFL-CV-1SE | 0.9672 | 10.7027 | 0.9900 | 0.0501 | 0.9499 | 0.9900 | 0.9146 |
| 3.0 | 0.33 | Lasso-CV | 0.7458 | 10.3047 | 1.0000 | 0.3943 | 0.6057 | 1.0000 | 0.9179 |
| 3.0 | 0.33 | AdaptiveLasso-CV | 0.6185 | 111.2549 | 0.9980 | 0.5443 | 0.4557 | 0.9980 | 0.1284 |
| 3.0 | 0.33 | UniLasso-CV | 0.9748 | 14.9792 | 0.9520 | 0.0000 | 1.0000 | 0.9520 | 0.8805 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-1SE (F1=0.9934)
- rank_2: UniLasso-CV (F1=0.9895)
- rank_3: AdaptiveLasso-CV (F1=0.8047)
- rank_4: Lasso-CV (F1=0.7517)

### 6.2 By MSE (lower is better)

- rank_1: Lasso-CV (MSE=3.1502)
- rank_2: AFL-CV-1SE (MSE=3.3322)
- rank_3: UniLasso-CV (MSE=4.7121)
- rank_4: AdaptiveLasso-CV (MSE=102.7299)

## 7. Key Findings

1. **Best F1**: AFL-CV-1SE with F1=0.9934
2. **Best MSE**: Lasso-CV with MSE=3.1502

3. **SNR Sensitivity**:
   - AFL-CV-1SE: F1 drop = 0.0193 (high SNR to low SNR)
   - Lasso-CV: F1 drop = 0.0146 (high SNR to low SNR)
   - AdaptiveLasso-CV: F1 drop = 0.1995 (high SNR to low SNR)
   - UniLasso-CV: F1 drop = -0.0051 (high SNR to low SNR)

---
*Report generated: 2026-03-27T12:58:13.544070*
