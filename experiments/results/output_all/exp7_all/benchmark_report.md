# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-04-02
**Experiment**: exp7_pfl_vs_adaptive
**Family**: gaussian
**Repeats**: 20 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **PFLRegressorCV (BAFL, gamma=1.0, cap=10.0)** (CV-tuned)
- **AdaptiveLassoCV (fixed)** (CV-tuned)
- **LassoCV** (CV-tuned)
- **UnilassoCV** (CV-tuned)
- **RelaxedLassoCV** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AdaptiveLassoCV (fixed) | 0.1561 | 0.0933 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.1490 | 0.0712 |
| LassoCV | 0.1220 | 0.0946 |
| RelaxedLassoCV | 0.1130 | 0.0867 |
| UnilassoCV | 0.1121 | 0.0489 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| LassoCV | 5.7933 | 4.3266 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 5.8129 | 3.9966 |
| UnilassoCV | 5.8758 | 3.9904 |
| RelaxedLassoCV | 5.8916 | 4.2861 |
| AdaptiveLassoCV (fixed) | 12.1344 | 11.8073 |

### 4. TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveLassoCV (fixed) | 0.5344 | 0.2753 |
| LassoCV | 0.2225 | 0.3582 |
| RelaxedLassoCV | 0.1756 | 0.3020 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.1075 | 0.0883 |
| UnilassoCV | 0.0613 | 0.0297 |

### 5. FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| UnilassoCV | 0.1419 | 0.2386 |
| RelaxedLassoCV | 0.2385 | 0.3497 |
| LassoCV | 0.3190 | 0.3686 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.4362 | 0.3293 |
| AdaptiveLassoCV (fixed) | 0.7190 | 0.3637 |

### 6. Sign Accuracy (higher is better)

| Model | Sign Acc Mean | Sign Acc Std |
|-------|---------------|--------------|
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.8488 | 0.0742 |
| AdaptiveLassoCV (fixed) | 0.7338 | 0.1488 |
| LassoCV | 0.6013 | 0.1866 |
| RelaxedLassoCV | 0.5756 | 0.1603 |
| UnilassoCV | 0.5069 | 0.0191 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.2069 |    0.2237 |                                       0.2042 |           0.2093 |       0.1236 |
|     1   |                    0.2001 |    0.1346 |                                       0.1564 |           0.1239 |       0.1155 |
|     2   |                    0.1247 |    0.0923 |                                       0.1186 |           0.0828 |       0.1139 |
|     3   |                    0.0928 |    0.0376 |                                       0.1166 |           0.0362 |       0.0952 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    2.1678 |    1.341  |                                       2.1031 |           1.6668 |       2.2015 |
|     1   |                    3.3121 |    3.1027 |                                       2.9959 |           3.0777 |       3.0549 |
|     2   |                   12.6615 |    6.5664 |                                       6.3088 |           6.5492 |       6.359  |
|     3   |                   30.396  |   12.1633 |                                      11.8439 |          12.2726 |      11.8878 |

### 5. TPR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.2625 |    0.735  |                                       0.1625 |           0.5675 |       0.0675 |
|     1   |                    0.6575 |    0.08   |                                       0.1175 |           0.07   |       0.0625 |
|     2   |                    0.5925 |    0.0525 |                                       0.075  |           0.045  |       0.0625 |
|     3   |                    0.625  |    0.0225 |                                       0.075  |           0.02   |       0.0525 |

### 6. FDR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.2733 |    0.7335 |                                       0.5269 |           0.6654 |       0.0667 |
|     1   |                    0.7243 |    0.3175 |                                       0.4583 |           0.15   |       0.0917 |
|     2   |                    0.9286 |    0.1458 |                                       0.3373 |           0.0792 |       0.1583 |
|     3   |                    0.9498 |    0.0792 |                                       0.4222 |           0.0595 |       0.2508 |

### 7. Ridge Prior Sign Accuracy by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.6275 |    0.8675 |                                       0.8975 |           0.785  |       0.5125 |
|     1   |                    0.83   |    0.5225 |                                       0.8775 |           0.515  |       0.5075 |
|     2   |                    0.7575 |    0.51   |                                       0.835  |           0.5025 |       0.505  |
|     3   |                    0.72   |    0.505  |                                       0.785  |           0.5    |       0.5025 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 | Sign Acc | Sel F1 | Sel Acc |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|-----------|--------|--------|
| 0.5 | 2.0 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.2042 | 2.1031 | 0.1625 | 0.5269 | 0.4731 | 0.1625 | 0.2957 | 0.8975 | 0.2042 | 0.9538 |
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.2069 | 2.1678 | 0.2625 | 0.2733 | 0.6767 | 0.2625 | 0.2969 | 0.6275 | 0.2069 | 0.9390 |
| 0.5 | 2.0 | LassoCV | 0.2237 | 1.3410 | 0.7350 | 0.7335 | 0.2665 | 0.7350 | 0.5421 | 0.8675 | 0.2237 | 0.8203 |
| 0.5 | 2.0 | UnilassoCV | 0.1236 | 2.2015 | 0.0675 | 0.0667 | 0.9333 | 0.0675 | 0.2685 | 0.5125 | 0.1236 | 0.9624 |
| 0.5 | 2.0 | RelaxedLassoCV | 0.2093 | 1.6668 | 0.5675 | 0.6654 | 0.3346 | 0.5675 | 0.4242 | 0.7850 | 0.2093 | 0.8414 |
| 1.0 | 1.0 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.1564 | 2.9959 | 0.1175 | 0.4583 | 0.5417 | 0.1175 | 0.2078 | 0.8775 | 0.1564 | 0.9540 |
| 1.0 | 1.0 | AdaptiveLassoCV (fixed) | 0.2001 | 3.3121 | 0.6575 | 0.7243 | 0.2757 | 0.6575 | 0.1201 | 0.8300 | 0.2001 | 0.7619 |
| 1.0 | 1.0 | LassoCV | 0.1346 | 3.1027 | 0.0800 | 0.3175 | 0.6825 | 0.0800 | 0.1881 | 0.5225 | 0.1346 | 0.9600 |
| 1.0 | 1.0 | UnilassoCV | 0.1155 | 3.0549 | 0.0625 | 0.0917 | 0.9083 | 0.0625 | 0.1971 | 0.5075 | 0.1155 | 0.9621 |
| 1.0 | 1.0 | RelaxedLassoCV | 0.1239 | 3.0777 | 0.0700 | 0.1500 | 0.8500 | 0.0700 | 0.1849 | 0.5150 | 0.1239 | 0.9616 |
| 2.0 | 0.5 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.1186 | 6.3088 | 0.0750 | 0.3373 | 0.6627 | 0.0750 | 0.0895 | 0.8350 | 0.1186 | 0.9579 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.1247 | 12.6615 | 0.5925 | 0.9286 | 0.0714 | 0.5925 | -0.8294 | 0.7575 | 0.1247 | 0.6406 |
| 2.0 | 0.5 | LassoCV | 0.0923 | 6.5664 | 0.0525 | 0.1458 | 0.6542 | 0.0525 | 0.0576 | 0.5100 | 0.0923 | 0.9608 |
| 2.0 | 0.5 | UnilassoCV | 0.1139 | 6.3590 | 0.0625 | 0.1583 | 0.8417 | 0.0625 | 0.0862 | 0.5050 | 0.1139 | 0.9616 |
| 2.0 | 0.5 | RelaxedLassoCV | 0.0828 | 6.5492 | 0.0450 | 0.0792 | 0.7208 | 0.0450 | 0.0556 | 0.5025 | 0.0828 | 0.9611 |
| 3.0 | 0.33 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.1166 | 11.8439 | 0.0750 | 0.4222 | 0.5278 | 0.0750 | 0.0261 | 0.7850 | 0.1166 | 0.9574 |
| 3.0 | 0.33 | AdaptiveLassoCV (fixed) | 0.0928 | 30.3960 | 0.6250 | 0.9498 | 0.0502 | 0.6250 | -1.5072 | 0.7200 | 0.0928 | 0.5115 |
| 3.0 | 0.33 | LassoCV | 0.0376 | 12.1633 | 0.0225 | 0.0792 | 0.2208 | 0.0225 | 0.0017 | 0.5050 | 0.0376 | 0.9597 |
| 3.0 | 0.33 | UnilassoCV | 0.0952 | 11.8878 | 0.0525 | 0.2508 | 0.6992 | 0.0525 | 0.0235 | 0.5025 | 0.0952 | 0.9605 |
| 3.0 | 0.33 | RelaxedLassoCV | 0.0362 | 12.2726 | 0.0200 | 0.0595 | 0.2905 | 0.0200 | -0.0086 | 0.5000 | 0.0362 | 0.9601 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveLassoCV (fixed) (F1=0.1561)
- rank_2: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (F1=0.1490)
- rank_3: LassoCV (F1=0.1220)
- rank_4: RelaxedLassoCV (F1=0.1130)
- rank_5: UnilassoCV (F1=0.1121)

### 6.2 By MSE (lower is better)

- rank_1: LassoCV (MSE=5.7933)
- rank_2: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (MSE=5.8129)
- rank_3: UnilassoCV (MSE=5.8758)
- rank_4: RelaxedLassoCV (MSE=5.8916)
- rank_5: AdaptiveLassoCV (fixed) (MSE=12.1344)

### 6.4 By Sign Accuracy (higher is better)

- rank_1: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (Sign Acc=0.8488)
- rank_2: AdaptiveLassoCV (fixed) (Sign Acc=0.7338)
- rank_3: LassoCV (Sign Acc=0.6013)
- rank_4: RelaxedLassoCV (Sign Acc=0.5756)
- rank_5: UnilassoCV (Sign Acc=0.5069)

## 7. Key Findings

1. **Best F1**: AdaptiveLassoCV (fixed) with F1=0.1561
2. **Best MSE**: LassoCV with MSE=5.7933

6. **SNR Sensitivity**:
   - PFLRegressorCV (BAFL, gamma=1.0, cap=10.0): F1 drop = 0.0865 (high SNR to low SNR)
   - AdaptiveLassoCV (fixed): F1 drop = 0.0981 (high SNR to low SNR)
   - LassoCV: F1 drop = 0.1588 (high SNR to low SNR)
   - UnilassoCV: F1 drop = 0.0191 (high SNR to low SNR)
   - RelaxedLassoCV: F1 drop = 0.1498 (high SNR to low SNR)

7. **Ridge Prior Sign Accuracy**:
   - PFLRegressorCV (BAFL, gamma=1.0, cap=10.0): 0.8488
   - AdaptiveLassoCV (fixed): 0.7338
   - LassoCV: 0.6013
   - UnilassoCV: 0.5069
   - RelaxedLassoCV: 0.5756

---
*Report generated: 2026-04-02T17:55:12.716758*
