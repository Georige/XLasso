# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-04-02
**Experiment**: exp5_pfl_vs_adaptive
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
| AdaptiveLassoCV (fixed) | 0.0214 | 0.0123 |
| LassoCV | 0.0000 | 0.0000 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.0000 | 0.0000 |
| RelaxedLassoCV | 0.0000 | 0.0000 |
| UnilassoCV | 0.0000 | 0.0000 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| LassoCV | 3.4339 | 3.4375 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 3.4361 | 3.4403 |
| UnilassoCV | 3.7468 | 3.4066 |
| RelaxedLassoCV | 3.8863 | 3.9032 |
| AdaptiveLassoCV (fixed) | 16.9747 | 24.7302 |

### 4. TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveLassoCV (fixed) | 0.1450 | 0.0833 |
| LassoCV | 0.0000 | 0.0000 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.0000 | 0.0000 |
| RelaxedLassoCV | 0.0000 | 0.0000 |
| UnilassoCV | 0.0000 | 0.0000 |

### 5. FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| LassoCV | 0.0500 | 0.2193 |
| UnilassoCV | 0.0750 | 0.2651 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.4625 | 0.5017 |
| RelaxedLassoCV | 0.9000 | 0.3019 |
| AdaptiveLassoCV (fixed) | 0.9384 | 0.2167 |

### 6. Sign Accuracy (higher is better)

| Model | Sign Acc Mean | Sign Acc Std |
|-------|---------------|--------------|
| AdaptiveLassoCV (fixed) | 0.5163 | 0.0502 |
| LassoCV | 0.5000 | 0.0000 |
| RelaxedLassoCV | 0.5000 | 0.0000 |
| UnilassoCV | 0.5000 | 0.0000 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.4588 | 0.0818 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.0223 |         0 |                                            0 |                0 |            0 |
|     1   |                    0.0204 |         0 |                                            0 |                0 |            0 |
|     2   |                    0.0209 |         0 |                                            0 |                0 |            0 |
|     3   |                    0.0221 |         0 |                                            0 |                0 |            0 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    1.1582 |    0.2405 |                                       0.241  |           0.2711 |       0.6462 |
|     1   |                    4.6308 |    0.9648 |                                       0.9642 |           1.0838 |       1.3364 |
|     2   |                   18.8635 |    3.8556 |                                       3.8579 |           4.3357 |       4.1511 |
|     3   |                   43.2462 |    8.6748 |                                       8.6814 |           9.8547 |       8.8536 |

### 5. TPR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.1425 |         0 |                                            0 |                0 |            0 |
|     1   |                    0.135  |         0 |                                            0 |                0 |            0 |
|     2   |                    0.145  |         0 |                                            0 |                0 |            0 |
|     3   |                    0.1575 |         0 |                                            0 |                0 |            0 |

### 6. FDR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.9379 |      0.05 |                                         0.3  |             0.8  |          0   |
|     1   |                    0.9389 |      0.05 |                                         0.4  |             0.9  |          0   |
|     2   |                    0.9387 |      0.05 |                                         0.55 |             0.95 |          0.1 |
|     3   |                    0.9381 |      0.05 |                                         0.6  |             0.95 |          0.2 |

### 7. Ridge Prior Sign Accuracy by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.5125 |       0.5 |                                        0.45  |              0.5 |          0.5 |
|     1   |                    0.5125 |       0.5 |                                        0.455 |              0.5 |          0.5 |
|     2   |                    0.52   |       0.5 |                                        0.455 |              0.5 |          0.5 |
|     3   |                    0.52   |       0.5 |                                        0.475 |              0.5 |          0.5 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 | Sign Acc | Sel F1 | Sel Acc |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|-----------|--------|--------|
| 0.5 | 2.0 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.0000 | 0.2410 | 0.0000 | 0.3000 | 0.0000 | 0.0000 | -0.0224 | 0.4500 | 0.0000 | 0.9790 |
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.0223 | 1.1582 | 0.1425 | 0.9379 | 0.0121 | 0.1425 | -3.9847 | 0.5125 | 0.0223 | 0.7615 |
| 0.5 | 2.0 | LassoCV | 0.0000 | 0.2405 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | -0.0204 | 0.5000 | 0.0000 | 0.9796 |
| 0.5 | 2.0 | UnilassoCV | 0.0000 | 0.6462 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -1.6950 | 0.5000 | 0.0000 | 0.9800 |
| 0.5 | 2.0 | RelaxedLassoCV | 0.0000 | 0.2711 | 0.0000 | 0.8000 | 0.0000 | 0.0000 | -0.1573 | 0.5000 | 0.0000 | 0.9747 |
| 1.0 | 1.0 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.0000 | 0.9642 | 0.0000 | 0.4000 | 0.0000 | 0.0000 | -0.0229 | 0.4550 | 0.0000 | 0.9789 |
| 1.0 | 1.0 | AdaptiveLassoCV (fixed) | 0.0204 | 4.6308 | 0.1350 | 0.9389 | 0.0111 | 0.1350 | -3.9792 | 0.5125 | 0.0204 | 0.7533 |
| 1.0 | 1.0 | LassoCV | 0.0000 | 0.9648 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | -0.0234 | 0.5000 | 0.0000 | 0.9788 |
| 1.0 | 1.0 | UnilassoCV | 0.0000 | 1.3364 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.4096 | 0.5000 | 0.0000 | 0.9800 |
| 1.0 | 1.0 | RelaxedLassoCV | 0.0000 | 1.0838 | 0.0000 | 0.9000 | 0.0000 | 0.0000 | -0.1564 | 0.5000 | 0.0000 | 0.9747 |
| 2.0 | 0.5 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.0000 | 3.8579 | 0.0000 | 0.5500 | 0.0000 | 0.0000 | -0.0233 | 0.4550 | 0.0000 | 0.9789 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.0209 | 18.8635 | 0.1450 | 0.9387 | 0.0113 | 0.1450 | -4.1058 | 0.5200 | 0.0209 | 0.7414 |
| 2.0 | 0.5 | LassoCV | 0.0000 | 3.8556 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | -0.0226 | 0.5000 | 0.0000 | 0.9792 |
| 2.0 | 0.5 | UnilassoCV | 0.0000 | 4.1511 | 0.0000 | 0.1000 | 0.0000 | 0.0000 | -0.1008 | 0.5000 | 0.0000 | 0.9799 |
| 2.0 | 0.5 | RelaxedLassoCV | 0.0000 | 4.3357 | 0.0000 | 0.9500 | 0.0000 | 0.0000 | -0.1576 | 0.5000 | 0.0000 | 0.9745 |
| 3.0 | 0.33 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.0000 | 8.6814 | 0.0000 | 0.6000 | 0.0000 | 0.0000 | -0.0235 | 0.4750 | 0.0000 | 0.9789 |
| 3.0 | 0.33 | AdaptiveLassoCV (fixed) | 0.0221 | 43.2462 | 0.1575 | 0.9381 | 0.0119 | 0.1575 | -4.2298 | 0.5200 | 0.0221 | 0.7344 |
| 3.0 | 0.33 | LassoCV | 0.0000 | 8.6748 | 0.0000 | 0.0500 | 0.0000 | 0.0000 | -0.0228 | 0.5000 | 0.0000 | 0.9791 |
| 3.0 | 0.33 | UnilassoCV | 0.0000 | 8.8536 | 0.0000 | 0.2000 | 0.0000 | 0.0000 | -0.0446 | 0.5000 | 0.0000 | 0.9793 |
| 3.0 | 0.33 | RelaxedLassoCV | 0.0000 | 9.8547 | 0.0000 | 0.9500 | 0.0000 | 0.0000 | -0.1679 | 0.5000 | 0.0000 | 0.9737 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AdaptiveLassoCV (fixed) (F1=0.0214)
- rank_2: LassoCV (F1=0.0000)
- rank_3: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (F1=0.0000)
- rank_4: RelaxedLassoCV (F1=0.0000)
- rank_5: UnilassoCV (F1=0.0000)

### 6.2 By MSE (lower is better)

- rank_1: LassoCV (MSE=3.4339)
- rank_2: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (MSE=3.4361)
- rank_3: UnilassoCV (MSE=3.7468)
- rank_4: RelaxedLassoCV (MSE=3.8863)
- rank_5: AdaptiveLassoCV (fixed) (MSE=16.9747)

### 6.4 By Sign Accuracy (higher is better)

- rank_1: AdaptiveLassoCV (fixed) (Sign Acc=0.5163)
- rank_2: LassoCV (Sign Acc=0.5000)
- rank_3: RelaxedLassoCV (Sign Acc=0.5000)
- rank_4: UnilassoCV (Sign Acc=0.5000)
- rank_5: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (Sign Acc=0.4588)

## 7. Key Findings

1. **Best F1**: AdaptiveLassoCV (fixed) with F1=0.0214
2. **Best MSE**: LassoCV with MSE=3.4339

6. **SNR Sensitivity**:
   - PFLRegressorCV (BAFL, gamma=1.0, cap=10.0): F1 drop = 0.0000 (high SNR to low SNR)
   - AdaptiveLassoCV (fixed): F1 drop = 0.0008 (high SNR to low SNR)
   - LassoCV: F1 drop = 0.0000 (high SNR to low SNR)
   - UnilassoCV: F1 drop = 0.0000 (high SNR to low SNR)
   - RelaxedLassoCV: F1 drop = 0.0000 (high SNR to low SNR)

7. **Ridge Prior Sign Accuracy**:
   - PFLRegressorCV (BAFL, gamma=1.0, cap=10.0): 0.4588
   - AdaptiveLassoCV (fixed): 0.5163
   - LassoCV: 0.5000
   - UnilassoCV: 0.5000
   - RelaxedLassoCV: 0.5000

---
*Report generated: 2026-04-02T16:29:03.227057*
