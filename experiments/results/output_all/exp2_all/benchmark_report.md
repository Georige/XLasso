# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-04-02
**Experiment**: exp2_pfl_vs_adaptive
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
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.8839 | 0.0914 |
| RelaxedLassoCV | 0.7570 | 0.0710 |
| ElasticNetCV | 0.7106 | 0.0968 |
| UnilassoCV | 0.7057 | 0.0632 |
| LassoCV | 0.6918 | 0.0901 |
| AdaptiveLassoCV (fixed) | 0.6171 | 0.3715 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| RelaxedLassoCV | 4.7059 | 4.6699 |
| LassoCV | 4.8650 | 4.8281 |
| UnilassoCV | 4.9417 | 4.5279 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 4.9979 | 4.9422 |
| ElasticNetCV | 5.0246 | 4.9890 |
| AdaptiveLassoCV (fixed) | 9.5233 | 10.6084 |

### 4. TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| UnilassoCV | 0.9950 | 0.0219 |
| ElasticNetCV | 0.9925 | 0.0329 |
| LassoCV | 0.9881 | 0.0349 |
| RelaxedLassoCV | 0.9825 | 0.0458 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9631 | 0.0715 |
| AdaptiveLassoCV (fixed) | 0.9606 | 0.0655 |

### 5. FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.1764 | 0.1233 |
| RelaxedLassoCV | 0.3792 | 0.0856 |
| ElasticNetCV | 0.4381 | 0.1097 |
| AdaptiveLassoCV (fixed) | 0.4411 | 0.4035 |
| UnilassoCV | 0.4499 | 0.0732 |
| LassoCV | 0.4611 | 0.1021 |

### 6. Sign Accuracy (higher is better)

| Model | Sign Acc Mean | Sign Acc Std |
|-------|---------------|--------------|
| AdaptiveLassoCV (fixed) | 1.0000 | 0.0000 |
| ElasticNetCV | 1.0000 | 0.0000 |
| LassoCV | 1.0000 | 0.0000 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 1.0000 | 0.0000 |
| UnilassoCV | 1.0000 | 0.0000 |
| RelaxedLassoCV | 0.9975 | 0.0110 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   ElasticNetCV |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|---------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.979  |         0.7412 |    0.6929 |                                       0.9718 |           0.7581 |       0.727  |
|     1   |                    0.8934 |         0.7382 |    0.7074 |                                       0.9303 |           0.7637 |       0.7126 |
|     2   |                    0.4238 |         0.7139 |    0.6917 |                                       0.8492 |           0.7637 |       0.7029 |
|     3   |                    0.172  |         0.6493 |    0.6752 |                                       0.7843 |           0.7425 |       0.6804 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   ElasticNetCV |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|---------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    1.0361 |         0.3588 |    0.352  |                                       0.4396 |           0.3438 |       0.7483 |
|     1   |                    2.3577 |         1.4266 |    1.4103 |                                       1.4285 |           1.3628 |       1.6791 |
|     2   |                    8.4652 |         5.675  |    5.4703 |                                       5.5501 |           5.3416 |       5.4674 |
|     3   |                   26.2341 |        12.6379 |   12.2273 |                                      12.5735 |          11.7755 |      11.872  |

### 5. TPR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   ElasticNetCV |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|---------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.9975 |           1    |    1      |                                       1      |           1      |       1      |
|     1   |                    0.9875 |           1    |    1      |                                       1      |           1      |       1      |
|     2   |                    0.9625 |           1    |    0.9975 |                                       0.98   |           0.9975 |       0.9975 |
|     3   |                    0.895  |           0.97 |    0.955  |                                       0.8725 |           0.9325 |       0.9825 |

### 6. FDR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   ElasticNetCV |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|---------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.0368 |         0.4051 |    0.4594 |                                       0.0533 |           0.3839 |       0.4254 |
|     1   |                    0.1511 |         0.4097 |    0.445  |                                       0.126  |           0.3766 |       0.4436 |
|     2   |                    0.6724 |         0.4375 |    0.4667 |                                       0.2465 |           0.3771 |       0.4543 |
|     3   |                    0.9041 |         0.5    |    0.4731 |                                       0.2797 |           0.3793 |       0.4761 |

### 7. Ridge Prior Sign Accuracy by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   ElasticNetCV |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|---------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                         1 |              1 |         1 |                                            1 |             1    |            1 |
|     1   |                         1 |              1 |         1 |                                            1 |             1    |            1 |
|     2   |                         1 |              1 |         1 |                                            1 |             1    |            1 |
|     3   |                         1 |              1 |         1 |                                            1 |             0.99 |            1 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 | Sign Acc | Sel F1 | Sel Acc |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|-----------|--------|--------|
| 0.5 | 2.0 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9718 | 0.4396 | 1.0000 | 0.0533 | 0.9467 | 1.0000 | 0.9941 | 1.0000 | 0.9718 | 0.9976 |
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.9790 | 1.0361 | 0.9975 | 0.0368 | 0.9632 | 0.9975 | 0.9859 | 1.0000 | 0.9790 | 0.9982 |
| 0.5 | 2.0 | LassoCV | 0.6929 | 0.3520 | 1.0000 | 0.4594 | 0.5406 | 1.0000 | 0.9953 | 1.0000 | 0.6929 | 0.9613 |
| 0.5 | 2.0 | UnilassoCV | 0.7270 | 0.7483 | 1.0000 | 0.4254 | 0.5746 | 1.0000 | 0.9900 | 1.0000 | 0.7270 | 0.9691 |
| 0.5 | 2.0 | RelaxedLassoCV | 0.7581 | 0.3438 | 1.0000 | 0.3839 | 0.6161 | 1.0000 | 0.9954 | 1.0000 | 0.7581 | 0.9733 |
| 0.5 | 2.0 | ElasticNetCV | 0.7412 | 0.3588 | 1.0000 | 0.4051 | 0.5949 | 1.0000 | 0.9952 | 1.0000 | 0.7412 | 0.9705 |
| 1.0 | 1.0 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9303 | 1.4285 | 1.0000 | 0.1260 | 0.8740 | 1.0000 | 0.9810 | 1.0000 | 0.9303 | 0.9937 |
| 1.0 | 1.0 | AdaptiveLassoCV (fixed) | 0.8934 | 2.3577 | 0.9875 | 0.1511 | 0.8489 | 0.9875 | 0.9685 | 1.0000 | 0.8934 | 0.9859 |
| 1.0 | 1.0 | LassoCV | 0.7074 | 1.4103 | 1.0000 | 0.4450 | 0.5550 | 1.0000 | 0.9814 | 1.0000 | 0.7074 | 0.9647 |
| 1.0 | 1.0 | UnilassoCV | 0.7126 | 1.6791 | 1.0000 | 0.4436 | 0.5564 | 1.0000 | 0.9777 | 1.0000 | 0.7126 | 0.9670 |
| 1.0 | 1.0 | RelaxedLassoCV | 0.7637 | 1.3628 | 1.0000 | 0.3766 | 0.6234 | 1.0000 | 0.9819 | 1.0000 | 0.7637 | 0.9741 |
| 1.0 | 1.0 | ElasticNetCV | 0.7382 | 1.4266 | 1.0000 | 0.4097 | 0.5903 | 1.0000 | 0.9811 | 1.0000 | 0.7382 | 0.9703 |
| 2.0 | 0.5 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.8492 | 5.5501 | 0.9800 | 0.2465 | 0.7535 | 0.9800 | 0.9296 | 1.0000 | 0.8492 | 0.9857 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.4238 | 8.4652 | 0.9625 | 0.6724 | 0.3276 | 0.9625 | 0.8905 | 1.0000 | 0.4238 | 0.8103 |
| 2.0 | 0.5 | LassoCV | 0.6917 | 5.4703 | 0.9975 | 0.4667 | 0.5333 | 0.9975 | 0.9306 | 1.0000 | 0.6917 | 0.9632 |
| 2.0 | 0.5 | UnilassoCV | 0.7029 | 5.4674 | 0.9975 | 0.4543 | 0.5457 | 0.9975 | 0.9301 | 1.0000 | 0.7029 | 0.9654 |
| 2.0 | 0.5 | RelaxedLassoCV | 0.7637 | 5.3416 | 0.9975 | 0.3771 | 0.6229 | 0.9975 | 0.9320 | 1.0000 | 0.7637 | 0.9745 |
| 2.0 | 0.5 | ElasticNetCV | 0.7139 | 5.6750 | 1.0000 | 0.4375 | 0.5625 | 1.0000 | 0.9282 | 1.0000 | 0.7139 | 0.9661 |
| 3.0 | 0.33 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.7843 | 12.5735 | 0.8725 | 0.2797 | 0.7203 | 0.8725 | 0.8506 | 1.0000 | 0.7843 | 0.9806 |
| 3.0 | 0.33 | AdaptiveLassoCV (fixed) | 0.1720 | 26.2341 | 0.8950 | 0.9041 | 0.0959 | 0.8950 | 0.6853 | 1.0000 | 0.1720 | 0.6403 |
| 3.0 | 0.33 | LassoCV | 0.6752 | 12.2273 | 0.9550 | 0.4731 | 0.5269 | 0.9550 | 0.8551 | 1.0000 | 0.6752 | 0.9619 |
| 3.0 | 0.33 | UnilassoCV | 0.6804 | 11.8720 | 0.9825 | 0.4761 | 0.5239 | 0.9825 | 0.8579 | 1.0000 | 0.6804 | 0.9621 |
| 3.0 | 0.33 | RelaxedLassoCV | 0.7425 | 11.7755 | 0.9325 | 0.3793 | 0.6207 | 0.9325 | 0.8597 | 0.9900 | 0.7425 | 0.9737 |
| 3.0 | 0.33 | ElasticNetCV | 0.6493 | 12.6379 | 0.9700 | 0.5000 | 0.5000 | 0.9700 | 0.8506 | 1.0000 | 0.6493 | 0.9535 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (F1=0.8839)
- rank_2: RelaxedLassoCV (F1=0.7570)
- rank_3: ElasticNetCV (F1=0.7106)
- rank_4: UnilassoCV (F1=0.7057)
- rank_5: LassoCV (F1=0.6918)
- rank_6: AdaptiveLassoCV (fixed) (F1=0.6171)

### 6.2 By MSE (lower is better)

- rank_1: RelaxedLassoCV (MSE=4.7059)
- rank_2: LassoCV (MSE=4.8650)
- rank_3: UnilassoCV (MSE=4.9417)
- rank_4: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (MSE=4.9979)
- rank_5: ElasticNetCV (MSE=5.0246)
- rank_6: AdaptiveLassoCV (fixed) (MSE=9.5233)

### 6.4 By Sign Accuracy (higher is better)

- rank_1: AdaptiveLassoCV (fixed) (Sign Acc=1.0000)
- rank_2: ElasticNetCV (Sign Acc=1.0000)
- rank_3: LassoCV (Sign Acc=1.0000)
- rank_4: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (Sign Acc=1.0000)
- rank_5: UnilassoCV (Sign Acc=1.0000)
- rank_6: RelaxedLassoCV (Sign Acc=0.9975)

## 7. Key Findings

1. **Best F1**: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) with F1=0.8839
2. **Best MSE**: RelaxedLassoCV with MSE=4.7059

6. **SNR Sensitivity**:
   - PFLRegressorCV (BAFL, gamma=1.0, cap=10.0): F1 drop = 0.1550 (high SNR to low SNR)
   - AdaptiveLassoCV (fixed): F1 drop = 0.6812 (high SNR to low SNR)
   - LassoCV: F1 drop = 0.0095 (high SNR to low SNR)
   - UnilassoCV: F1 drop = 0.0353 (high SNR to low SNR)
   - RelaxedLassoCV: F1 drop = 0.0050 (high SNR to low SNR)
   - ElasticNetCV: F1 drop = 0.0596 (high SNR to low SNR)

7. **Ridge Prior Sign Accuracy**:
   - PFLRegressorCV (BAFL, gamma=1.0, cap=10.0): 1.0000
   - AdaptiveLassoCV (fixed): 1.0000
   - LassoCV: 1.0000
   - UnilassoCV: 1.0000
   - RelaxedLassoCV: 0.9975
   - ElasticNetCV: 1.0000

---
*Report generated: 2026-04-02T09:08:28.579895*
