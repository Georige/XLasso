# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-04-02
**Experiment**: exp4_pfl_vs_adaptive
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
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9762 | 0.0293 |
| AdaptiveLassoCV (fixed) | 0.7282 | 0.2634 |
| RelaxedLassoCV | 0.6947 | 0.1163 |
| LassoCV | 0.6019 | 0.1168 |
| UnilassoCV | 0.4820 | 0.0675 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 5.0187 | 4.7577 |
| RelaxedLassoCV | 5.3840 | 5.4017 |
| AdaptiveLassoCV (fixed) | 5.4840 | 5.2500 |
| LassoCV | 5.5766 | 5.5508 |
| UnilassoCV | 6.3022 | 4.5737 |

### 4. TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| AdaptiveLassoCV (fixed) | 1.0000 | 0.0000 |
| LassoCV | 1.0000 | 0.0000 |
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 1.0000 | 0.0000 |
| RelaxedLassoCV | 1.0000 | 0.0000 |
| UnilassoCV | 1.0000 | 0.0000 |

### 5. FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.0449 | 0.0545 |
| AdaptiveLassoCV (fixed) | 0.3624 | 0.3205 |
| RelaxedLassoCV | 0.4561 | 0.1334 |
| LassoCV | 0.5591 | 0.1275 |
| UnilassoCV | 0.6798 | 0.0601 |

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
|     0.5 |                    0.8117 |    0.6286 |                                       0.9988 |           0.6497 |       0.4972 |
|     1   |                    0.9844 |    0.6138 |                                       0.994  |           0.6883 |       0.4852 |
|     2   |                    0.715  |    0.5854 |                                       0.9668 |           0.7143 |       0.4784 |
|     3   |                    0.4019 |    0.5797 |                                       0.9453 |           0.7267 |       0.4673 |

### 4.2 MSE by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    1.0616 |    0.4013 |                                       0.6208 |           0.3966 |       2.4197 |
|     1   |                    1.8319 |    1.6083 |                                       1.5694 |           1.5396 |       3.1254 |
|     2   |                    5.3814 |    6.2522 |                                       5.5223 |           6.0581 |       6.611  |
|     3   |                   13.6609 |   14.0445 |                                      12.3622 |          13.5416 |      13.0527 |

### 5. TPR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                         1 |         1 |                                            1 |                1 |            1 |
|     1   |                         1 |         1 |                                            1 |                1 |            1 |
|     2   |                         1 |         1 |                                            1 |                1 |            1 |
|     3   |                         1 |         1 |                                            1 |                1 |            1 |

### 6. FDR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LassoCV |   PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|----------:|---------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.2358 |    0.5291 |                                       0.0024 |           0.5074 |       0.6651 |
|     1   |                    0.0299 |    0.541  |                                       0.0117 |           0.4643 |       0.6774 |
|     2   |                    0.4374 |    0.5801 |                                       0.0633 |           0.4334 |       0.6835 |
|     3   |                    0.7467 |    0.586  |                                       0.1024 |           0.4191 |       0.6934 |

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
| 0.5 | 2.0 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9988 | 0.6208 | 1.0000 | 0.0024 | 0.9976 | 1.0000 | 0.9967 | 1.0000 | 0.9988 | 1.0000 |
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.8117 | 1.0616 | 1.0000 | 0.2358 | 0.7642 | 1.0000 | 0.9943 | 1.0000 | 0.8117 | 0.9743 |
| 0.5 | 2.0 | LassoCV | 0.6286 | 0.4013 | 1.0000 | 0.5291 | 0.4709 | 1.0000 | 0.9979 | 1.0000 | 0.6286 | 0.9735 |
| 0.5 | 2.0 | UnilassoCV | 0.4972 | 2.4197 | 1.0000 | 0.6651 | 0.3349 | 1.0000 | 0.9870 | 1.0000 | 0.4972 | 0.9572 |
| 0.5 | 2.0 | RelaxedLassoCV | 0.6497 | 0.3966 | 1.0000 | 0.5074 | 0.4926 | 1.0000 | 0.9979 | 1.0000 | 0.6497 | 0.9758 |
| 1.0 | 1.0 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9940 | 1.5694 | 1.0000 | 0.0117 | 0.9883 | 1.0000 | 0.9917 | 1.0000 | 0.9940 | 0.9998 |
| 1.0 | 1.0 | AdaptiveLassoCV (fixed) | 0.9844 | 1.8319 | 1.0000 | 0.0299 | 0.9701 | 1.0000 | 0.9902 | 1.0000 | 0.9844 | 0.9993 |
| 1.0 | 1.0 | LassoCV | 0.6138 | 1.6083 | 1.0000 | 0.5410 | 0.4590 | 1.0000 | 0.9916 | 1.0000 | 0.6138 | 0.9718 |
| 1.0 | 1.0 | UnilassoCV | 0.4852 | 3.1254 | 1.0000 | 0.6774 | 0.3226 | 1.0000 | 0.9832 | 1.0000 | 0.4852 | 0.9562 |
| 1.0 | 1.0 | RelaxedLassoCV | 0.6883 | 1.5396 | 1.0000 | 0.4643 | 0.5357 | 1.0000 | 0.9918 | 1.0000 | 0.6883 | 0.9802 |
| 2.0 | 0.5 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9668 | 5.5223 | 1.0000 | 0.0633 | 0.9367 | 1.0000 | 0.9713 | 1.0000 | 0.9668 | 0.9986 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.7150 | 5.3814 | 1.0000 | 0.4374 | 0.5626 | 1.0000 | 0.9715 | 1.0000 | 0.7150 | 0.9833 |
| 2.0 | 0.5 | LassoCV | 0.5854 | 6.2522 | 1.0000 | 0.5801 | 0.4199 | 1.0000 | 0.9676 | 1.0000 | 0.5854 | 0.9698 |
| 2.0 | 0.5 | UnilassoCV | 0.4784 | 6.6110 | 1.0000 | 0.6835 | 0.3165 | 1.0000 | 0.9647 | 1.0000 | 0.4784 | 0.9551 |
| 2.0 | 0.5 | RelaxedLassoCV | 0.7143 | 6.0581 | 1.0000 | 0.4334 | 0.5666 | 1.0000 | 0.9681 | 1.0000 | 0.7143 | 0.9825 |
| 3.0 | 0.33 | PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) | 0.9453 | 12.3622 | 1.0000 | 0.1024 | 0.8976 | 1.0000 | 0.9373 | 1.0000 | 0.9453 | 0.9976 |
| 3.0 | 0.33 | AdaptiveLassoCV (fixed) | 0.4019 | 13.6609 | 1.0000 | 0.7467 | 0.2533 | 1.0000 | 0.9298 | 1.0000 | 0.4019 | 0.9382 |
| 3.0 | 0.33 | LassoCV | 0.5797 | 14.0445 | 1.0000 | 0.5860 | 0.4140 | 1.0000 | 0.9290 | 1.0000 | 0.5797 | 0.9691 |
| 3.0 | 0.33 | UnilassoCV | 0.4673 | 13.0527 | 1.0000 | 0.6934 | 0.3066 | 1.0000 | 0.9320 | 1.0000 | 0.4673 | 0.9533 |
| 3.0 | 0.33 | RelaxedLassoCV | 0.7267 | 13.5416 | 1.0000 | 0.4191 | 0.5809 | 1.0000 | 0.9304 | 1.0000 | 0.7267 | 0.9836 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (F1=0.9762)
- rank_2: AdaptiveLassoCV (fixed) (F1=0.7282)
- rank_3: RelaxedLassoCV (F1=0.6947)
- rank_4: LassoCV (F1=0.6019)
- rank_5: UnilassoCV (F1=0.4820)

### 6.2 By MSE (lower is better)

- rank_1: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (MSE=5.0187)
- rank_2: RelaxedLassoCV (MSE=5.3840)
- rank_3: AdaptiveLassoCV (fixed) (MSE=5.4840)
- rank_4: LassoCV (MSE=5.5766)
- rank_5: UnilassoCV (MSE=6.3022)

### 6.4 By Sign Accuracy (higher is better)

- rank_1: AdaptiveLassoCV (fixed) (Sign Acc=1.0000)
- rank_2: LassoCV (Sign Acc=1.0000)
- rank_3: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) (Sign Acc=1.0000)
- rank_4: RelaxedLassoCV (Sign Acc=1.0000)
- rank_5: UnilassoCV (Sign Acc=1.0000)

## 7. Key Findings

1. **Best F1**: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) with F1=0.9762
2. **Best MSE**: PFLRegressorCV (BAFL, gamma=1.0, cap=10.0) with MSE=5.0187

6. **SNR Sensitivity**:
   - PFLRegressorCV (BAFL, gamma=1.0, cap=10.0): F1 drop = 0.0427 (high SNR to low SNR)
   - AdaptiveLassoCV (fixed): F1 drop = 0.2532 (high SNR to low SNR)
   - LassoCV: F1 drop = 0.0460 (high SNR to low SNR)
   - UnilassoCV: F1 drop = 0.0243 (high SNR to low SNR)
   - RelaxedLassoCV: F1 drop = -0.0708 (high SNR to low SNR)

7. **Ridge Prior Sign Accuracy**:
   - PFLRegressorCV (BAFL, gamma=1.0, cap=10.0): 1.0000
   - AdaptiveLassoCV (fixed): 1.0000
   - LassoCV: 1.0000
   - UnilassoCV: 1.0000
   - RelaxedLassoCV: 1.0000

---
*Report generated: 2026-04-02T15:34:32.192657*
