# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-04-03
**Experiment**: exp6_pfl_cv_vs_adaptive
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
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9720 | 0.0583 |
| UnilassoCV | 0.8840 | 0.1033 |
| RelaxedLassoCV | 0.8233 | 0.1515 |
| AdaptiveLassoCV (fixed) | 0.7932 | 0.2850 |
| LassoCV | 0.7101 | 0.1843 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| UnilassoCV | 3.9431 | 4.0274 |
| RelaxedLassoCV | 4.1401 | 4.2682 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 4.2642 | 4.3991 |
| LassoCV | 4.4713 | 4.5187 |
| AdaptiveLassoCV (fixed) | 4.6288 | 4.3610 |

### 4. TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| UnilassoCV | 0.9988 | 0.0112 |
| RelaxedLassoCV | 0.9975 | 0.0157 |
| LassoCV | 0.9975 | 0.0157 |
| AdaptiveLassoCV (fixed) | 0.9912 | 0.0326 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9850 | 0.0480 |

### 5. FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.0362 | 0.0842 |
| UnilassoCV | 0.1931 | 0.1578 |
| AdaptiveLassoCV (fixed) | 0.2594 | 0.3391 |
| RelaxedLassoCV | 0.2720 | 0.2097 |
| LassoCV | 0.4164 | 0.2256 |

### 6. Sign Accuracy (higher is better)

| Model | Sign Acc Mean | Sign Acc Std |
|-------|---------------|--------------|
| AdaptiveLassoCV (fixed) | 1.0000 | 0.0000 |
| LassoCV | 1.0000 | 0.0000 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 1.0000 | 0.0000 |
| RelaxedLassoCV | 1.0000 | 0.0000 |
| UnilassoCV | 1.0000 | 0.0000 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.9704 |    0.6841 |                                       1      |           0.8096 |       0.9563 |
|     1   |                    0.9785 |    0.7274 |                                       0.9976 |           0.8253 |       0.9251 |
|     2   |                    0.8383 |    0.7229 |                                       0.9792 |           0.8308 |       0.8637 |
|     3   |                    0.3857 |    0.706  |                                       0.9112 |           0.8276 |       0.7908 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    1.1989 |    0.3092 |                                       0.2942 |           0.2945 |       0.326  |
|     1   |                    1.6352 |    1.2536 |                                       1.1406 |           1.1567 |       1.0926 |
|     2   |                    4.5035 |    5.0357 |                                       4.6528 |           4.6107 |       4.3318 |
|     3   |                   11.1776 |   11.2867 |                                      10.9692 |          10.4985 |      10.0219 |

### 5. TPR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                     0.985 |      1    |                                         1    |             1    |        1     |
|     1   |                     0.995 |      1    |                                         1    |             1    |        1     |
|     2   |                     1     |      1    |                                         1    |             1    |        1     |
|     3   |                     0.985 |      0.99 |                                         0.94 |             0.99 |        0.995 |

### 6. FDR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.03   |    0.4589 |                                       0      |           0.2902 |       0.0779 |
|     1   |                    0.0306 |    0.3965 |                                       0.0045 |           0.2715 |       0.1317 |
|     2   |                    0.2273 |    0.3964 |                                       0.0362 |           0.2633 |       0.2283 |
|     3   |                    0.7497 |    0.4138 |                                       0.1039 |           0.263  |       0.3345 |

### 7. Ridge Prior Sign Accuracy by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                         1 |         1 |                                            1 |                1 |            1 |
|     1   |                         1 |         1 |                                            1 |                1 |            1 |
|     2   |                         1 |         1 |                                            1 |                1 |            1 |
|     3   |                         1 |         1 |                                            1 |                1 |            1 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 | Sign Acc | Sel F1 | Sel Acc |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|-----------|--------|--------|
| 0.5 | 2.0 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 1.0000 | 0.2942 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9794 | 1.0000 | 1.0000 | 1.0000 |
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.9704 | 1.1989 | 0.9850 | 0.0300 | 0.9700 | 0.9850 | 0.9174 | 1.0000 | 0.9704 | 0.9982 |
| 0.5 | 2.0 | LassoCV | 0.6841 | 0.3092 | 1.0000 | 0.4589 | 0.5411 | 1.0000 | 0.9783 | 1.0000 | 0.6841 | 0.9771 |
| 0.5 | 2.0 | UnilassoCV | 0.9563 | 0.3260 | 1.0000 | 0.0779 | 0.9221 | 1.0000 | 0.9771 | 1.0000 | 0.9563 | 0.9980 |
| 0.5 | 2.0 | RelaxedLassoCV | 0.8096 | 0.2945 | 1.0000 | 0.2902 | 0.7098 | 1.0000 | 0.9792 | 1.0000 | 0.8096 | 0.9880 |
| 1.0 | 1.0 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9976 | 1.1406 | 1.0000 | 0.0045 | 0.9955 | 1.0000 | 0.9242 | 1.0000 | 0.9976 | 0.9999 |
| 1.0 | 1.0 | AdaptiveLassoCV (fixed) | 0.9785 | 1.6352 | 0.9950 | 0.0306 | 0.9694 | 0.9950 | 0.8919 | 1.0000 | 0.9785 | 0.9989 |
| 1.0 | 1.0 | LassoCV | 0.7274 | 1.2536 | 1.0000 | 0.3965 | 0.6035 | 1.0000 | 0.9167 | 1.0000 | 0.7274 | 0.9806 |
| 1.0 | 1.0 | UnilassoCV | 0.9251 | 1.0926 | 1.0000 | 0.1317 | 0.8683 | 1.0000 | 0.9271 | 1.0000 | 0.9251 | 0.9965 |
| 1.0 | 1.0 | RelaxedLassoCV | 0.8253 | 1.1567 | 1.0000 | 0.2715 | 0.7285 | 1.0000 | 0.9223 | 1.0000 | 0.8253 | 0.9896 |
| 2.0 | 0.5 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9792 | 4.6528 | 1.0000 | 0.0362 | 0.9638 | 1.0000 | 0.7422 | 1.0000 | 0.9792 | 0.9990 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.8383 | 4.5035 | 1.0000 | 0.2273 | 0.7727 | 1.0000 | 0.7498 | 1.0000 | 0.8383 | 0.9764 |
| 2.0 | 0.5 | LassoCV | 0.7229 | 5.0357 | 1.0000 | 0.3964 | 0.6036 | 1.0000 | 0.7207 | 1.0000 | 0.7229 | 0.9800 |
| 2.0 | 0.5 | UnilassoCV | 0.8637 | 4.3318 | 1.0000 | 0.2283 | 0.7717 | 1.0000 | 0.7590 | 1.0000 | 0.8637 | 0.9931 |
| 2.0 | 0.5 | RelaxedLassoCV | 0.8308 | 4.6107 | 1.0000 | 0.2633 | 0.7367 | 1.0000 | 0.7422 | 1.0000 | 0.8308 | 0.9900 |
| 3.0 | 0.33 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9112 | 10.9692 | 0.9400 | 0.1039 | 0.8961 | 0.9400 | 0.5212 | 1.0000 | 0.9112 | 0.9962 |
| 3.0 | 0.33 | AdaptiveLassoCV (fixed) | 0.3857 | 11.1776 | 0.9850 | 0.7497 | 0.2503 | 0.9850 | 0.5084 | 1.0000 | 0.3857 | 0.9248 |
| 3.0 | 0.33 | LassoCV | 0.7060 | 11.2867 | 0.9900 | 0.4138 | 0.5862 | 0.9900 | 0.5073 | 1.0000 | 0.7060 | 0.9787 |
| 3.0 | 0.33 | UnilassoCV | 0.7908 | 10.0219 | 0.9950 | 0.3345 | 0.6655 | 0.9950 | 0.5615 | 1.0000 | 0.7908 | 0.9887 |
| 3.0 | 0.33 | RelaxedLassoCV | 0.8276 | 10.4985 | 0.9900 | 0.2630 | 0.7370 | 0.9900 | 0.5382 | 1.0000 | 0.8276 | 0.9900 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (F1=0.9720)
- rank_2: UnilassoCV (F1=0.8840)
- rank_3: RelaxedLassoCV (F1=0.8233)
- rank_4: AdaptiveLassoCV (fixed) (F1=0.7932)
- rank_5: LassoCV (F1=0.7101)

### 6.2 By MSE (lower is better)

- rank_1: UnilassoCV (MSE=3.9431)
- rank_2: RelaxedLassoCV (MSE=4.1401)
- rank_3: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (MSE=4.2642)
- rank_4: LassoCV (MSE=4.4713)
- rank_5: AdaptiveLassoCV (fixed) (MSE=4.6288)

### 6.4 By Sign Accuracy (higher is better)

- rank_1: AdaptiveLassoCV (fixed) (Sign Acc=1.0000)
- rank_2: LassoCV (Sign Acc=1.0000)
- rank_3: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (Sign Acc=1.0000)
- rank_4: RelaxedLassoCV (Sign Acc=1.0000)
- rank_5: UnilassoCV (Sign Acc=1.0000)

## 7. Key Findings

1. **Best F1**: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) with F1=0.9720
2. **Best MSE**: UnilassoCV with MSE=3.9431

6. **SNR Sensitivity**:
   - PFLRegressorCV (BAFL, gamma=1.0, cap=10.0): F1 drop = 0.0548 (high SNR to low SNR)
   - AdaptiveLassoCV (fixed): F1 drop = 0.3584 (high SNR to low SNR)
   - LassoCV: F1 drop = -0.0304 (high SNR to low SNR)
   - UnilassoCV: F1 drop = 0.1291 (high SNR to low SNR)
   - RelaxedLassoCV: F1 drop = -0.0196 (high SNR to low SNR)

7. **Ridge Prior Sign Accuracy**:
   - PFLRegressorCV (BAFL, gamma=1.0, cap=10.0): 1.0000
   - AdaptiveLassoCV (fixed): 1.0000
   - LassoCV: 1.0000
   - UnilassoCV: 1.0000
   - RelaxedLassoCV: 1.0000

---
*Report generated: 2026-04-03T22:02:18.717266*
