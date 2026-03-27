# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-27
**Experiment**: exp4_lasso_compare
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 10.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **AFL-CV-1SE** (CV-tuned)
- **AdaptiveLasso-CV** (CV-tuned)
- **ElasticNet-1SE** (CV-tuned)
- **RelaxedLasso-1SE** (CV-tuned)
- **UniLasso-CV** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| AFL-CV-1SE | 0.9637 | 0.0367 |
| AdaptiveLasso-CV | 0.8826 | 0.1195 |
| UniLasso-CV | 0.7606 | 0.0684 |
| RelaxedLasso-1SE | 0.7126 | 0.0578 |
| ElasticNet-1SE | 0.7126 | 0.0578 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| RelaxedLasso-1SE | 4.0547 | 4.6661 |
| AFL-CV-1SE | 4.0799 | 4.1776 |
| ElasticNet-1SE | 4.5124 | 5.1790 |
| UniLasso-CV | 4.7405 | 3.3431 |
| AdaptiveLasso-CV | 162.4190 | 14.3911 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| ElasticNet-1SE | 1.0000 | 0.0000 |
| UniLasso-CV | 1.0000 | 0.0000 |
| RelaxedLasso-1SE | 1.0000 | 0.0000 |
| AFL-CV-1SE | 0.9997 | 0.0018 |
| AdaptiveLasso-CV | 0.8113 | 0.1621 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AdaptiveLasso-CV | 0.0003 | 0.0018 |
| AFL-CV-1SE | 0.0653 | 0.0643 |
| UniLasso-CV | 0.3749 | 0.0864 |
| RelaxedLasso-1SE | 0.4354 | 0.0690 |
| ElasticNet-1SE | 0.4355 | 0.0691 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLasso-CV |   ElasticNet-1SE |   RelaxedLasso-1SE |   UniLasso-CV |
|--------:|-------------:|-------------------:|-----------------:|-------------------:|--------------:|
|     0.1 |       1      |             0.9244 |           0.7073 |             0.7068 |        0.7632 |
|     0.5 |       0.9952 |             0.9146 |           0.7172 |             0.7172 |        0.77   |
|     1   |       0.9751 |             0.8987 |           0.7166 |             0.7166 |        0.7803 |
|     1.5 |       0.9477 |             0.8834 |           0.7102 |             0.7102 |        0.7728 |
|     2   |       0.9436 |             0.8649 |           0.7138 |             0.7138 |        0.7465 |
|     3   |       0.9203 |             0.8099 |           0.7103 |             0.7113 |        0.7305 |

### 4.2 MSE by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLasso-CV |   ElasticNet-1SE |   RelaxedLasso-1SE |   UniLasso-CV |
|--------:|-------------:|-------------------:|-----------------:|-------------------:|--------------:|
|     0.1 |       0.5654 |            158.699 |           0.0164 |             0.0147 |        2.1986 |
|     0.5 |       0.8416 |            159.906 |           0.4117 |             0.3691 |        2.2691 |
|     1   |       1.761  |            160.831 |           1.6402 |             1.4774 |        2.8183 |
|     1.5 |       3.2874 |            162.086 |           3.6926 |             3.3283 |        3.9935 |
|     2   |       5.5945 |            163.876 |           6.5725 |             5.9325 |        5.7691 |
|     3   |      12.4297 |            169.116 |          14.741  |            13.2063 |       11.3946 |

### 4.3 TPR by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLasso-CV |   ElasticNet-1SE |   RelaxedLasso-1SE |   UniLasso-CV |
|--------:|-------------:|-------------------:|-----------------:|-------------------:|--------------:|
|     0.1 |        1     |              0.868 |                1 |                  1 |             1 |
|     0.5 |        1     |              0.852 |                1 |                  1 |             1 |
|     1   |        1     |              0.83  |                1 |                  1 |             1 |
|     1.5 |        1     |              0.81  |                1 |                  1 |             1 |
|     2   |        1     |              0.782 |                1 |                  1 |             1 |
|     3   |        0.998 |              0.726 |                1 |                  1 |             1 |

### 4.4 FDR by Sigma

|   sigma |   AFL-CV-1SE |   AdaptiveLasso-CV |   ElasticNet-1SE |   RelaxedLasso-1SE |   UniLasso-CV |
|--------:|-------------:|-------------------:|-----------------:|-------------------:|--------------:|
|     0.1 |       0      |              0     |           0.4415 |             0.4422 |        0.3732 |
|     0.5 |       0.0094 |              0     |           0.4292 |             0.4292 |        0.3606 |
|     1   |       0.0462 |              0     |           0.4303 |             0.4303 |        0.349  |
|     1.5 |       0.0944 |              0     |           0.439  |             0.439  |        0.359  |
|     2   |       0.1026 |              0     |           0.4341 |             0.4341 |        0.3926 |
|     3   |       0.139  |              0.002 |           0.4387 |             0.4376 |        0.4146 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 0.1 | 10.0 | AFL-CV-1SE | 1.0000 | 0.5654 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9966 |
| 0.1 | 10.0 | AdaptiveLasso-CV | 0.9244 | 158.6994 | 0.8680 | 0.0000 | 1.0000 | 0.8680 | 0.0965 |
| 0.1 | 10.0 | ElasticNet-1SE | 0.7073 | 0.0164 | 1.0000 | 0.4415 | 0.5585 | 1.0000 | 0.9999 |
| 0.1 | 10.0 | RelaxedLasso-1SE | 0.7068 | 0.0147 | 1.0000 | 0.4422 | 0.5578 | 1.0000 | 0.9999 |
| 0.1 | 10.0 | UniLasso-CV | 0.7632 | 2.1986 | 1.0000 | 0.3732 | 0.6268 | 1.0000 | 0.9872 |
| 0.5 | 2.0 | AFL-CV-1SE | 0.9952 | 0.8416 | 1.0000 | 0.0094 | 0.9906 | 1.0000 | 0.9950 |
| 0.5 | 2.0 | AdaptiveLasso-CV | 0.9146 | 159.9056 | 0.8520 | 0.0000 | 1.0000 | 0.8520 | 0.0887 |
| 0.5 | 2.0 | ElasticNet-1SE | 0.7172 | 0.4117 | 1.0000 | 0.4292 | 0.5708 | 1.0000 | 0.9976 |
| 0.5 | 2.0 | RelaxedLasso-1SE | 0.7172 | 0.3691 | 1.0000 | 0.4292 | 0.5708 | 1.0000 | 0.9978 |
| 0.5 | 2.0 | UniLasso-CV | 0.7700 | 2.2691 | 1.0000 | 0.3606 | 0.6394 | 1.0000 | 0.9866 |
| 1.0 | 1.0 | AFL-CV-1SE | 0.9751 | 1.7610 | 1.0000 | 0.0462 | 0.9538 | 1.0000 | 0.9897 |
| 1.0 | 1.0 | AdaptiveLasso-CV | 0.8987 | 160.8309 | 0.8300 | 0.0000 | 1.0000 | 0.8300 | 0.0850 |
| 1.0 | 1.0 | ElasticNet-1SE | 0.7166 | 1.6402 | 1.0000 | 0.4303 | 0.5697 | 1.0000 | 0.9906 |
| 1.0 | 1.0 | RelaxedLasso-1SE | 0.7166 | 1.4774 | 1.0000 | 0.4303 | 0.5697 | 1.0000 | 0.9913 |
| 1.0 | 1.0 | UniLasso-CV | 0.7803 | 2.8183 | 1.0000 | 0.3490 | 0.6510 | 1.0000 | 0.9832 |
| 1.5 | 0.67 | AFL-CV-1SE | 0.9477 | 3.2874 | 1.0000 | 0.0944 | 0.9056 | 1.0000 | 0.9809 |
| 1.5 | 0.67 | AdaptiveLasso-CV | 0.8834 | 162.0857 | 0.8100 | 0.0000 | 1.0000 | 0.8100 | 0.0823 |
| 1.5 | 0.67 | ElasticNet-1SE | 0.7102 | 3.6926 | 1.0000 | 0.4390 | 0.5610 | 1.0000 | 0.9789 |
| 1.5 | 0.67 | RelaxedLasso-1SE | 0.7102 | 3.3283 | 1.0000 | 0.4390 | 0.5610 | 1.0000 | 0.9806 |
| 1.5 | 0.67 | UniLasso-CV | 0.7728 | 3.9935 | 1.0000 | 0.3590 | 0.6410 | 1.0000 | 0.9763 |
| 2.0 | 0.5 | AFL-CV-1SE | 0.9436 | 5.5945 | 1.0000 | 0.1026 | 0.8974 | 1.0000 | 0.9678 |
| 2.0 | 0.5 | AdaptiveLasso-CV | 0.8649 | 163.8759 | 0.7820 | 0.0000 | 1.0000 | 0.7820 | 0.0795 |
| 2.0 | 0.5 | ElasticNet-1SE | 0.7138 | 6.5725 | 1.0000 | 0.4341 | 0.5659 | 1.0000 | 0.9627 |
| 2.0 | 0.5 | RelaxedLasso-1SE | 0.7138 | 5.9325 | 1.0000 | 0.4341 | 0.5659 | 1.0000 | 0.9656 |
| 2.0 | 0.5 | UniLasso-CV | 0.7465 | 5.7691 | 1.0000 | 0.3926 | 0.6074 | 1.0000 | 0.9660 |
| 3.0 | 0.33 | AFL-CV-1SE | 0.9203 | 12.4297 | 0.9980 | 0.1390 | 0.8610 | 0.9980 | 0.9300 |
| 3.0 | 0.33 | AdaptiveLasso-CV | 0.8099 | 169.1162 | 0.7260 | 0.0020 | 0.9580 | 0.7260 | 0.0716 |
| 3.0 | 0.33 | ElasticNet-1SE | 0.7103 | 14.7410 | 1.0000 | 0.4387 | 0.5613 | 1.0000 | 0.9181 |
| 3.0 | 0.33 | RelaxedLasso-1SE | 0.7113 | 13.2063 | 1.0000 | 0.4376 | 0.5624 | 1.0000 | 0.9251 |
| 3.0 | 0.33 | UniLasso-CV | 0.7305 | 11.3946 | 1.0000 | 0.4146 | 0.5854 | 1.0000 | 0.9346 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: AFL-CV-1SE (F1=0.9637)
- rank_2: AdaptiveLasso-CV (F1=0.8826)
- rank_3: UniLasso-CV (F1=0.7606)
- rank_4: RelaxedLasso-1SE (F1=0.7126)
- rank_5: ElasticNet-1SE (F1=0.7126)

### 6.2 By MSE (lower is better)

- rank_1: RelaxedLasso-1SE (MSE=4.0547)
- rank_2: AFL-CV-1SE (MSE=4.0799)
- rank_3: ElasticNet-1SE (MSE=4.5124)
- rank_4: UniLasso-CV (MSE=4.7405)
- rank_5: AdaptiveLasso-CV (MSE=162.4190)

## 7. Key Findings

1. **Best F1**: AFL-CV-1SE with F1=0.9637
2. **Best MSE**: RelaxedLasso-1SE with MSE=4.0547

3. **SNR Sensitivity**:
   - AFL-CV-1SE: F1 drop = 0.0656 (high SNR to low SNR)
   - AdaptiveLasso-CV: F1 drop = 0.0821 (high SNR to low SNR)
   - ElasticNet-1SE: F1 drop = 0.0002 (high SNR to low SNR)
   - RelaxedLasso-1SE: F1 drop = -0.0005 (high SNR to low SNR)
   - UniLasso-CV: F1 drop = 0.0281 (high SNR to low SNR)

---
*Report generated: 2026-03-27T15:53:33.660032*
