# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-04-02
**Experiment**: exp1_pfl_vs_adaptive
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
- **ElasticNetCV** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.8689 | 0.1395 |
| AdaptiveLassoCV (fixed) | 0.6156 | 0.3277 |
| RelaxedLassoCV | 0.4721 | 0.0353 |
| ElasticNetCV | 0.4638 | 0.0423 |
| LassoCV | 0.4589 | 0.0328 |
| UnilassoCV | 0.4294 | 0.0309 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| RelaxedLassoCV | 5.0274 | 5.0577 |
| LassoCV | 5.1444 | 5.1249 |
| UnilassoCV | 5.2966 | 4.9765 |
| ElasticNetCV | 5.4670 | 5.5105 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 5.5540 | 5.7013 |
| AdaptiveLassoCV (fixed) | 7.6863 | 7.2048 |

### 4. TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| UnilassoCV | 0.9950 | 0.0171 |
| LassoCV | 0.9938 | 0.0201 |
| ElasticNetCV | 0.9912 | 0.0222 |
| RelaxedLassoCV | 0.9850 | 0.0431 |
| AdaptiveLassoCV (fixed) | 0.9556 | 0.0601 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9550 | 0.0818 |

### 5. FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.1908 | 0.1841 |
| AdaptiveLassoCV (fixed) | 0.4509 | 0.3743 |
| RelaxedLassoCV | 0.6890 | 0.0288 |
| ElasticNetCV | 0.6964 | 0.0342 |
| LassoCV | 0.7012 | 0.0272 |
| UnilassoCV | 0.7258 | 0.0247 |

### 6. Sign Accuracy (higher is better)

| Model | Sign Acc Mean | Sign Acc Std |
|-------|---------------|--------------|
| AdaptiveLassoCV (fixed) | 1.0000 | 0.0000 |
| ElasticNetCV | 1.0000 | 0.0000 |
| LassoCV | 1.0000 | 0.0000 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 1.0000 | 0.0000 |
| RelaxedLassoCV | 1.0000 | 0.0000 |
| UnilassoCV | 1.0000 | 0.0000 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   ElasticNetCV |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|---------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.8308 |         0.4719 |    0.4595 |                                       0.9964 |           0.4764 |       0.4224 |
|     1   |                    0.868  |         0.473  |    0.4606 |                                       0.966  |           0.4769 |       0.429  |
|     2   |                    0.5001 |         0.4729 |    0.4626 |                                       0.8325 |           0.478  |       0.4359 |
|     3   |                    0.2636 |         0.4375 |    0.4529 |                                       0.6808 |           0.4572 |       0.4302 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   ElasticNetCV |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|---------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    2.0635 |         0.3871 |    0.3634 |                                       0.4644 |           0.3507 |       0.7069 |
|     1   |                    3.0106 |         1.5381 |    1.4542 |                                       1.4897 |           1.401  |       1.7445 |
|     2   |                    7.1134 |         6.1473 |    5.8045 |                                       5.8741 |           5.6133 |       5.9094 |
|     3   |                   18.5575 |        13.7955 |   12.9555 |                                      14.3876 |          12.7445 |      12.8256 |

### 5. TPR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   ElasticNetCV |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|---------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.965  |          1     |     1     |                                         1    |             1    |         1    |
|     1   |                    0.965  |          1     |     1     |                                         1    |             1    |         1    |
|     2   |                    0.9675 |          1     |     1     |                                         0.99 |             1    |         1    |
|     3   |                    0.925  |          0.965 |     0.975 |                                         0.83 |             0.94 |         0.98 |

### 6. FDR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   ElasticNetCV |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|---------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.1938 |         0.6907 |    0.7013 |                                       0.0069 |           0.6869 |       0.7318 |
|     1   |                    0.1682 |         0.6897 |    0.7003 |                                       0.0643 |           0.6864 |       0.7265 |
|     2   |                    0.6049 |         0.6898 |    0.6985 |                                       0.2763 |           0.6854 |       0.7209 |
|     3   |                    0.8368 |         0.7155 |    0.7046 |                                       0.4156 |           0.6975 |       0.7241 |

### 7. Ridge Prior Sign Accuracy by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   ElasticNetCV |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|---------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                         1 |              1 |         1 |                                            1 |                1 |            1 |
|     1   |                         1 |              1 |         1 |                                            1 |                1 |            1 |
|     2   |                         1 |              1 |         1 |                                            1 |                1 |            1 |
|     3   |                         1 |              1 |         1 |                                            1 |                1 |            1 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 | Sign Acc | Sel F1 | Sel Acc |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|-----------|--------|--------|
| 0.5 | 2.0 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9964 | 0.4644 | 1.0000 | 0.0069 | 0.9931 | 1.0000 | 0.9977 | 1.0000 | 0.9964 | 0.9997 |
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.8308 | 2.0635 | 0.9650 | 0.1938 | 0.8062 | 0.9650 | 0.9900 | 1.0000 | 0.8308 | 0.9602 |
| 0.5 | 2.0 | LassoCV | 0.4595 | 0.3634 | 1.0000 | 0.7013 | 0.2987 | 1.0000 | 0.9982 | 1.0000 | 0.4595 | 0.9052 |
| 0.5 | 2.0 | UnilassoCV | 0.4224 | 0.7069 | 1.0000 | 0.7318 | 0.2682 | 1.0000 | 0.9965 | 1.0000 | 0.4224 | 0.8896 |
| 0.5 | 2.0 | RelaxedLassoCV | 0.4764 | 0.3507 | 1.0000 | 0.6869 | 0.3131 | 1.0000 | 0.9983 | 1.0000 | 0.4764 | 0.9115 |
| 0.5 | 2.0 | ElasticNetCV | 0.4719 | 0.3871 | 1.0000 | 0.6907 | 0.3093 | 1.0000 | 0.9981 | 1.0000 | 0.4719 | 0.9098 |
| 1.0 | 1.0 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9660 | 1.4897 | 1.0000 | 0.0643 | 0.9357 | 1.0000 | 0.9926 | 1.0000 | 0.9660 | 0.9971 |
| 1.0 | 1.0 | AdaptiveLassoCV (fixed) | 0.8680 | 3.0106 | 0.9650 | 0.1682 | 0.8318 | 0.9650 | 0.9857 | 1.0000 | 0.8680 | 0.9836 |
| 1.0 | 1.0 | LassoCV | 0.4606 | 1.4542 | 1.0000 | 0.7003 | 0.2997 | 1.0000 | 0.9928 | 1.0000 | 0.4606 | 0.9056 |
| 1.0 | 1.0 | UnilassoCV | 0.4290 | 1.7445 | 1.0000 | 0.7265 | 0.2735 | 1.0000 | 0.9914 | 1.0000 | 0.4290 | 0.8926 |
| 1.0 | 1.0 | RelaxedLassoCV | 0.4769 | 1.4010 | 1.0000 | 0.6864 | 0.3136 | 1.0000 | 0.9931 | 1.0000 | 0.4769 | 0.9116 |
| 1.0 | 1.0 | ElasticNetCV | 0.4730 | 1.5381 | 1.0000 | 0.6897 | 0.3103 | 1.0000 | 0.9924 | 1.0000 | 0.4730 | 0.9102 |
| 2.0 | 0.5 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.8325 | 5.8741 | 0.9900 | 0.2763 | 0.7237 | 0.9900 | 0.9714 | 1.0000 | 0.8325 | 0.9835 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.5001 | 7.1134 | 0.9675 | 0.6049 | 0.3951 | 0.9675 | 0.9653 | 1.0000 | 0.5001 | 0.8612 |
| 2.0 | 0.5 | LassoCV | 0.4626 | 5.8045 | 1.0000 | 0.6985 | 0.3015 | 1.0000 | 0.9718 | 1.0000 | 0.4626 | 0.9062 |
| 2.0 | 0.5 | UnilassoCV | 0.4359 | 5.9094 | 1.0000 | 0.7209 | 0.2791 | 1.0000 | 0.9712 | 1.0000 | 0.4359 | 0.8956 |
| 2.0 | 0.5 | RelaxedLassoCV | 0.4780 | 5.6133 | 1.0000 | 0.6854 | 0.3146 | 1.0000 | 0.9727 | 1.0000 | 0.4780 | 0.9120 |
| 2.0 | 0.5 | ElasticNetCV | 0.4729 | 6.1473 | 1.0000 | 0.6898 | 0.3102 | 1.0000 | 0.9702 | 1.0000 | 0.4729 | 0.9102 |
| 3.0 | 0.33 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.6808 | 14.3876 | 0.8300 | 0.4156 | 0.5844 | 0.8300 | 0.9322 | 1.0000 | 0.6808 | 0.9676 |
| 3.0 | 0.33 | AdaptiveLassoCV (fixed) | 0.2636 | 18.5575 | 0.9250 | 0.8368 | 0.1632 | 0.9250 | 0.9110 | 1.0000 | 0.2636 | 0.7340 |
| 3.0 | 0.33 | LassoCV | 0.4529 | 12.9555 | 0.9750 | 0.7046 | 0.2954 | 0.9750 | 0.9389 | 1.0000 | 0.4529 | 0.9049 |
| 3.0 | 0.33 | UnilassoCV | 0.4302 | 12.8256 | 0.9800 | 0.7241 | 0.2759 | 0.9800 | 0.9392 | 1.0000 | 0.4302 | 0.8954 |
| 3.0 | 0.33 | RelaxedLassoCV | 0.4572 | 12.7445 | 0.9400 | 0.6975 | 0.3025 | 0.9400 | 0.9397 | 1.0000 | 0.4572 | 0.9101 |
| 3.0 | 0.33 | ElasticNetCV | 0.4375 | 13.7955 | 0.9650 | 0.7155 | 0.2845 | 0.9650 | 0.9352 | 1.0000 | 0.4375 | 0.8970 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (F1=0.8689)
- rank_2: AdaptiveLassoCV (fixed) (F1=0.6156)
- rank_3: RelaxedLassoCV (F1=0.4721)
- rank_4: ElasticNetCV (F1=0.4638)
- rank_5: LassoCV (F1=0.4589)
- rank_6: UnilassoCV (F1=0.4294)

### 6.2 By MSE (lower is better)

- rank_1: RelaxedLassoCV (MSE=5.0274)
- rank_2: LassoCV (MSE=5.1444)
- rank_3: UnilassoCV (MSE=5.2966)
- rank_4: ElasticNetCV (MSE=5.4670)
- rank_5: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (MSE=5.5540)
- rank_6: AdaptiveLassoCV (fixed) (MSE=7.6863)

### 6.4 By Sign Accuracy (higher is better)

- rank_1: AdaptiveLassoCV (fixed) (Sign Acc=1.0000)
- rank_2: ElasticNetCV (Sign Acc=1.0000)
- rank_3: LassoCV (Sign Acc=1.0000)
- rank_4: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (Sign Acc=1.0000)
- rank_5: RelaxedLassoCV (Sign Acc=1.0000)
- rank_6: UnilassoCV (Sign Acc=1.0000)

## 7. Key Findings

1. **Best F1**: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) with F1=0.8689
2. **Best MSE**: RelaxedLassoCV with MSE=5.0274

6. **SNR Sensitivity**:
   - PFLRegressorCV (BAFL, gamma=1.0, cap=10.0): F1 drop = 0.2398 (high SNR to low SNR)
   - AdaptiveLassoCV (fixed): F1 drop = 0.4490 (high SNR to low SNR)
   - LassoCV: F1 drop = 0.0018 (high SNR to low SNR)
   - UnilassoCV: F1 drop = -0.0106 (high SNR to low SNR)
   - RelaxedLassoCV: F1 drop = 0.0088 (high SNR to low SNR)
   - ElasticNetCV: F1 drop = 0.0167 (high SNR to low SNR)

7. **Ridge Prior Sign Accuracy**:
   - PFLRegressorCV (BAFL, gamma=1.0, cap=10.0): 1.0000
   - AdaptiveLassoCV (fixed): 1.0000
   - LassoCV: 1.0000
   - UnilassoCV: 1.0000
   - RelaxedLassoCV: 1.0000
   - ElasticNetCV: 1.0000

---
*Report generated: 2026-04-02T07:48:09.220603*
