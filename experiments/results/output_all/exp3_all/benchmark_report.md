# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-04-02
**Experiment**: exp3_pfl_vs_adaptive
**Family**: binomial
**Repeats**: 20 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **PFLClassifierCV (BAFL, gamma=1.0, cap=10.0)** (CV-tuned)
- **AdaptiveLassoCV (fixed)** (CV-tuned)
- **LogisticRegressionCV** (CV-tuned)
- **UnilassoCV** (CV-tuned)
- **RelaxedLassoCV** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| LogisticRegressionCV | 0.9903 | 0.0163 |
| PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) | 0.9902 | 0.0176 |
| UnilassoCV | 0.9867 | 0.0176 |
| RelaxedLassoCV | 0.9802 | 0.0265 |
| AdaptiveLassoCV (fixed) | 0.8769 | 0.0654 |

### 3.2 Accuracy (higher is better)

| Model | Accuracy Mean | Accuracy Std |
|-------|---------------|--------------|
| LogisticRegressionCV | 0.9906 | 0.0154 |
| PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) | 0.9904 | 0.0171 |
| UnilassoCV | 0.9869 | 0.0169 |
| RelaxedLassoCV | 0.9808 | 0.0252 |
| AdaptiveLassoCV (fixed) | 0.8898 | 0.0542 |

### 3.3 AUC (higher is better)

| Model | AUC Mean | AUC Std |
|-------|----------|---------|
| LogisticRegressionCV | 0.9978 | 0.0065 |
| PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) | 0.9977 | 0.0063 |
| RelaxedLassoCV | 0.9966 | 0.0080 |
| AdaptiveLassoCV (fixed) | 0.9739 | 0.0312 |
| UnilassoCV | 0.0000 | 0.0000 |

### 3.4 Selection F1 (higher is better)

| Model | Selection F1 Mean | Selection F1 Std |
|-------|-------------------|------------------|
| PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) | 0.8998 | 0.0992 |
| UnilassoCV | 0.8187 | 0.0770 |
| LogisticRegressionCV | 0.7951 | 0.1123 |
| AdaptiveLassoCV (fixed) | 0.3422 | 0.0813 |
| RelaxedLassoCV | 0.1321 | 0.1234 |

### 3.5 Selection Accuracy (higher is better)

| Model | Selection Acc Mean | Selection Acc Std |
|-------|--------------------|--------------------|
| PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) | 0.9926 | 0.0070 |
| UnilassoCV | 0.9880 | 0.0043 |
| LogisticRegressionCV | 0.9838 | 0.0140 |
| AdaptiveLassoCV (fixed) | 0.9684 | 0.0024 |
| RelaxedLassoCV | 0.2555 | 0.3024 |

### 4. TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| RelaxedLassoCV | 0.9650 | 0.0756 |
| PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) | 0.8875 | 0.1654 |
| LogisticRegressionCV | 0.7344 | 0.1267 |
| UnilassoCV | 0.7000 | 0.1082 |
| AdaptiveLassoCV (fixed) | 0.2094 | 0.0606 |

### 5. FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveLassoCV (fixed) | 0.0000 | 0.0000 |
| UnilassoCV | 0.0000 | 0.0000 |
| PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) | 0.0568 | 0.0952 |
| LogisticRegressionCV | 0.0874 | 0.1778 |
| RelaxedLassoCV | 0.9219 | 0.0971 |

### 6. Sign Accuracy (higher is better)

| Model | Sign Acc Mean | Sign Acc Std |
|-------|---------------|--------------|
| AdaptiveLassoCV (fixed) | 1.0000 | 0.0000 |
| LogisticRegressionCV | 1.0000 | 0.0000 |
| PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) | 1.0000 | 0.0000 |
| RelaxedLassoCV | 1.0000 | 0.0000 |
| UnilassoCV | 1.0000 | 0.0000 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LogisticRegressionCV |   PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|-----------------------:|----------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.8916 |                 0.9932 |                                        0.9937 |           0.9921 |       0.9923 |
|     1   |                    0.8768 |                 0.991  |                                        0.9919 |           0.9837 |       0.9892 |
|     2   |                    0.8846 |                 0.9939 |                                        0.993  |           0.982  |       0.9879 |
|     3   |                    0.8544 |                 0.9832 |                                        0.9821 |           0.9632 |       0.9772 |

### 4.2 Accuracy by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LogisticRegressionCV |   PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|-----------------------:|----------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.9025 |                 0.9933 |                                        0.9942 |           0.9925 |       0.9925 |
|     1   |                    0.8908 |                 0.9917 |                                        0.9925 |           0.985  |       0.99   |
|     2   |                    0.895  |                 0.9942 |                                        0.9933 |           0.9825 |       0.9883 |
|     3   |                    0.8708 |                 0.9833 |                                        0.9817 |           0.9633 |       0.9767 |

### 4.3 AUC by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LogisticRegressionCV |   PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|-----------------------:|----------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.9819 |                 0.9984 |                                        0.9985 |           0.9983 |            0 |
|     1   |                    0.9769 |                 0.9977 |                                        0.998  |           0.9969 |            0 |
|     2   |                    0.9797 |                 0.9981 |                                        0.998  |           0.9979 |            0 |
|     3   |                    0.9571 |                 0.9971 |                                        0.9964 |           0.9931 |            0 |

### 4.4 Selection F1 by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LogisticRegressionCV |   PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|-----------------------:|----------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.3669 |                 0.7991 |                                        0.9276 |           0.107  |       0.8385 |
|     1   |                    0.3357 |                 0.8067 |                                        0.9167 |           0.1529 |       0.819  |
|     2   |                    0.3403 |                 0.806  |                                        0.8881 |           0.1326 |       0.813  |
|     3   |                    0.326  |                 0.7685 |                                        0.8669 |           0.1359 |       0.8042 |

### 4.5 Selection Accuracy by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LogisticRegressionCV |   PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|-----------------------:|----------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.9691 |                 0.9837 |                                        0.994  |           0.1871 |       0.9892 |
|     1   |                    0.9682 |                 0.9867 |                                        0.9942 |           0.283  |       0.988  |
|     2   |                    0.9683 |                 0.9848 |                                        0.9912 |           0.2736 |       0.9877 |
|     3   |                    0.9679 |                 0.9798 |                                        0.9909 |           0.2785 |       0.9871 |

### 5. TPR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LogisticRegressionCV |   PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|-----------------------:|----------------------------------------------:|-----------------:|-------------:|
|     0.5 |                    0.2275 |                 0.7275 |                                        0.9525 |            0.985 |       0.73   |
|     1   |                    0.205  |                 0.7075 |                                        0.8925 |            0.955 |       0.7    |
|     2   |                    0.2075 |                 0.76   |                                        0.89   |            0.965 |       0.6925 |
|     3   |                    0.1975 |                 0.7425 |                                        0.815  |            0.955 |       0.6775 |

### 6. FDR by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LogisticRegressionCV |   PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|-----------------------:|----------------------------------------------:|-----------------:|-------------:|
|     0.5 |                         0 |                 0.0758 |                                        0.0809 |           0.9405 |            0 |
|     1   |                         0 |                 0.028  |                                        0.0313 |           0.9019 |            0 |
|     2   |                         0 |                 0.0917 |                                        0.0762 |           0.9243 |            0 |
|     3   |                         0 |                 0.1542 |                                        0.0388 |           0.9208 |            0 |

### 7. Ridge Prior Sign Accuracy by Sigma

|   sigma |   AdaptiveLassoCV (fixed) |   LogisticRegressionCV |   PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) |   RelaxedLassoCV |   UnilassoCV |
|--------:|--------------------------:|-----------------------:|----------------------------------------------:|-----------------:|-------------:|
|     0.5 |                         1 |                      1 |                                             1 |                1 |            1 |
|     1   |                         1 |                      1 |                                             1 |                1 |            1 |
|     2   |                         1 |                      1 |                                             1 |                1 |            1 |
|     3   |                         1 |                      1 |                                             1 |                1 |            1 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | Accuracy | AUC | TPR | FDR | Precision | Recall | Sign Acc | Sel F1 | Sel Acc |
|-------|-----|-------|-----|----------|-----|-----|-----|----------|--------|----------|--------|--------|
| 0.5 | 2.0 | PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) | 0.9937 | 0.9942 | 0.9985 | 0.9525 | 0.0809 | 0.9191 | 0.9525 | 1.0000 | 0.9276 | 0.9940 |
| 0.5 | 2.0 | AdaptiveLassoCV (fixed) | 0.8916 | 0.9025 | 0.9819 | 0.2275 | 0.0000 | 1.0000 | 0.2275 | 1.0000 | 0.3669 | 0.9691 |
| 0.5 | 2.0 | LogisticRegressionCV | 0.9932 | 0.9933 | 0.9984 | 0.7275 | 0.0758 | 0.9242 | 0.7275 | 1.0000 | 0.7991 | 0.9837 |
| 0.5 | 2.0 | UnilassoCV | 0.9923 | 0.9925 | 0.0000 | 0.7300 | 0.0000 | 1.0000 | 0.7300 | 1.0000 | 0.8385 | 0.9892 |
| 0.5 | 2.0 | RelaxedLassoCV | 0.9921 | 0.9925 | 0.9983 | 0.9850 | 0.9405 | 0.0595 | 0.9850 | 1.0000 | 0.1070 | 0.1871 |
| 1.0 | 1.0 | PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) | 0.9919 | 0.9925 | 0.9980 | 0.8925 | 0.0313 | 0.9687 | 0.8925 | 1.0000 | 0.9167 | 0.9942 |
| 1.0 | 1.0 | AdaptiveLassoCV (fixed) | 0.8768 | 0.8908 | 0.9769 | 0.2050 | 0.0000 | 1.0000 | 0.2050 | 1.0000 | 0.3357 | 0.9682 |
| 1.0 | 1.0 | LogisticRegressionCV | 0.9910 | 0.9917 | 0.9977 | 0.7075 | 0.0280 | 0.9720 | 0.7075 | 1.0000 | 0.8067 | 0.9867 |
| 1.0 | 1.0 | UnilassoCV | 0.9892 | 0.9900 | 0.0000 | 0.7000 | 0.0000 | 1.0000 | 0.7000 | 1.0000 | 0.8190 | 0.9880 |
| 1.0 | 1.0 | RelaxedLassoCV | 0.9837 | 0.9850 | 0.9969 | 0.9550 | 0.9019 | 0.0981 | 0.9550 | 1.0000 | 0.1529 | 0.2830 |
| 2.0 | 0.5 | PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) | 0.9930 | 0.9933 | 0.9980 | 0.8900 | 0.0762 | 0.9238 | 0.8900 | 1.0000 | 0.8881 | 0.9912 |
| 2.0 | 0.5 | AdaptiveLassoCV (fixed) | 0.8846 | 0.8950 | 0.9797 | 0.2075 | 0.0000 | 1.0000 | 0.2075 | 1.0000 | 0.3403 | 0.9683 |
| 2.0 | 0.5 | LogisticRegressionCV | 0.9939 | 0.9942 | 0.9981 | 0.7600 | 0.0917 | 0.9083 | 0.7600 | 1.0000 | 0.8060 | 0.9848 |
| 2.0 | 0.5 | UnilassoCV | 0.9879 | 0.9883 | 0.0000 | 0.6925 | 0.0000 | 1.0000 | 0.6925 | 1.0000 | 0.8130 | 0.9877 |
| 2.0 | 0.5 | RelaxedLassoCV | 0.9820 | 0.9825 | 0.9979 | 0.9650 | 0.9243 | 0.0757 | 0.9650 | 1.0000 | 0.1326 | 0.2736 |
| 3.0 | 0.33 | PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) | 0.9821 | 0.9817 | 0.9964 | 0.8150 | 0.0388 | 0.9612 | 0.8150 | 1.0000 | 0.8669 | 0.9909 |
| 3.0 | 0.33 | AdaptiveLassoCV (fixed) | 0.8544 | 0.8708 | 0.9571 | 0.1975 | 0.0000 | 1.0000 | 0.1975 | 1.0000 | 0.3260 | 0.9679 |
| 3.0 | 0.33 | LogisticRegressionCV | 0.9832 | 0.9833 | 0.9971 | 0.7425 | 0.1542 | 0.8458 | 0.7425 | 1.0000 | 0.7685 | 0.9798 |
| 3.0 | 0.33 | UnilassoCV | 0.9772 | 0.9767 | 0.0000 | 0.6775 | 0.0000 | 1.0000 | 0.6775 | 1.0000 | 0.8042 | 0.9871 |
| 3.0 | 0.33 | RelaxedLassoCV | 0.9632 | 0.9633 | 0.9931 | 0.9550 | 0.9208 | 0.0792 | 0.9550 | 1.0000 | 0.1359 | 0.2785 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: LogisticRegressionCV (F1=0.9903)
- rank_2: PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) (F1=0.9902)
- rank_3: UnilassoCV (F1=0.9867)
- rank_4: RelaxedLassoCV (F1=0.9802)
- rank_5: AdaptiveLassoCV (fixed) (F1=0.8769)

### 6.2 By Accuracy (higher is better)

- rank_1: LogisticRegressionCV (Accuracy=0.9906)
- rank_2: PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) (Accuracy=0.9904)
- rank_3: UnilassoCV (Accuracy=0.9869)
- rank_4: RelaxedLassoCV (Accuracy=0.9808)
- rank_5: AdaptiveLassoCV (fixed) (Accuracy=0.8898)

### 6.3 By AUC (higher is better)

- rank_1: LogisticRegressionCV (AUC=0.9978)
- rank_2: PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) (AUC=0.9977)
- rank_3: RelaxedLassoCV (AUC=0.9966)
- rank_4: AdaptiveLassoCV (fixed) (AUC=0.9739)
- rank_5: UnilassoCV (AUC=0.0000)

### 6.4 By Selection F1 (higher is better)

- rank_1: PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) (Selection F1=0.8998)
- rank_2: UnilassoCV (Selection F1=0.8187)
- rank_3: LogisticRegressionCV (Selection F1=0.7951)
- rank_4: AdaptiveLassoCV (fixed) (Selection F1=0.3422)
- rank_5: RelaxedLassoCV (Selection F1=0.1321)

### 6.5 By Selection Accuracy (higher is better)

- rank_1: PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) (Selection Acc=0.9926)
- rank_2: UnilassoCV (Selection Acc=0.9880)
- rank_3: LogisticRegressionCV (Selection Acc=0.9838)
- rank_4: AdaptiveLassoCV (fixed) (Selection Acc=0.9684)
- rank_5: RelaxedLassoCV (Selection Acc=0.2555)

### 6.4 By Sign Accuracy (higher is better)

- rank_1: AdaptiveLassoCV (fixed) (Sign Acc=1.0000)
- rank_2: LogisticRegressionCV (Sign Acc=1.0000)
- rank_3: PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) (Sign Acc=1.0000)
- rank_4: RelaxedLassoCV (Sign Acc=1.0000)
- rank_5: UnilassoCV (Sign Acc=1.0000)

## 7. Key Findings

1. **Best F1**: LogisticRegressionCV with F1=0.9903
2. **Best Accuracy**: LogisticRegressionCV with Accuracy=0.9906
3. **Best AUC**: LogisticRegressionCV with AUC=0.9978
4. **Best Selection F1**: PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) with Selection F1=0.8998
5. **Best Selection Accuracy**: PFLClassifierCV (BAFL, gamma=1.0, cap=10.0) with Selection Acc=0.9926

6. **SNR Sensitivity**:
   - PFLClassifierCV (BAFL, gamma=1.0, cap=10.0): F1 drop = 0.0062 (high SNR to low SNR)
   - AdaptiveLassoCV (fixed): F1 drop = 0.0221 (high SNR to low SNR)
   - LogisticRegressionCV: F1 drop = 0.0046 (high SNR to low SNR)
   - UnilassoCV: F1 drop = 0.0097 (high SNR to low SNR)
   - RelaxedLassoCV: F1 drop = 0.0195 (high SNR to low SNR)

7. **Ridge Prior Sign Accuracy**:
   - PFLClassifierCV (BAFL, gamma=1.0, cap=10.0): 1.0000
   - AdaptiveLassoCV (fixed): 1.0000
   - LogisticRegressionCV: 1.0000
   - UnilassoCV: 1.0000
   - RelaxedLassoCV: 1.0000

---
*Report generated: 2026-04-02T14:24:10.209717*
