# Benchmark Report: AdaptiveFlippedLasso Comparison

## Experiment Summary

- **Date**: 2026-03-26 23:17:35
- **Total Trials**: 120
- **Models Compared**: 4
- **Sigma Levels**: 0.1, 0.5, 1.0, 1.5, 2.0, 3.0
- **SNR Range**: 0.3 - 10.0

## Models Compared

- AdaptiveFlippedLasso
- LassoCV
- AdaptiveLassoCV
- GroupLassoCV

## Overall Performance by Model

| Model | F1 Mean | F1 Std | MSE Mean | MSE Std | TPR Mean | TPR Std | FDR Mean | FDR Std | Precision Mean | Recall Mean | R2 Mean |
|-------|---------|--------|----------|---------|----------|---------|----------|---------|----------------|-------------|---------|
| AdaptiveFlippedLasso | 0.8861 | 0.1095 | 3.9721 | 3.8561 | 0.9700 | 0.0677 | 0.1775 | 0.1466 | 0.8225 | 0.9700 | 0.9532 |
| LassoCV | 0.5418 | 0.0278 | 3.6140 | 4.1295 | 0.9937 | 0.0181 | 0.6238 | 0.0265 | 0.3762 | 0.9937 | 0.9557 |
| AdaptiveLassoCV | 0.4738 | 0.1697 | 65.4718 | 3.8595 | 0.9777 | 0.0452 | 0.6670 | 0.1498 | 0.3330 | 0.9777 | 0.1831 |
| GroupLassoCV | 0.1235 | 0.0206 | 65.5971 | 3.9508 | 0.9943 | 0.0117 | 0.9339 | 0.0121 | 0.0661 | 0.9943 | 0.1815 |

## Performance by Sigma Level

### Sigma = 0.1 (SNR = 10.0)

| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.9902 | 0.6448 | 1.0000 | 0.0190 | 0.9810 | 1.0000 | 0.9924 |
| LassoCV | 0.5625 | 0.0134 | 1.0000 | 0.6037 | 0.3963 | 1.0000 | 0.9998 |
| AdaptiveLassoCV | 0.5961 | 63.0694 | 1.0000 | 0.5749 | 0.4251 | 1.0000 | 0.1945 |
| GroupLassoCV | 0.1273 | 63.0793 | 1.0000 | 0.9319 | 0.0681 | 1.0000 | 0.1944 |

### Sigma = 0.5 (SNR = 2.0)

| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.9854 | 0.9293 | 1.0000 | 0.0286 | 0.9714 | 1.0000 | 0.9887 |
| LassoCV | 0.5649 | 0.3342 | 1.0000 | 0.6006 | 0.3994 | 1.0000 | 0.9956 |
| AdaptiveLassoCV | 0.7031 | 63.1367 | 1.0000 | 0.4396 | 0.5604 | 1.0000 | 0.1932 |
| GroupLassoCV | 0.1189 | 63.1916 | 1.0000 | 0.9366 | 0.0634 | 1.0000 | 0.1925 |

### Sigma = 1.0 (SNR = 1.0)

| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.9187 | 1.8064 | 1.0000 | 0.1484 | 0.8516 | 1.0000 | 0.9778 |
| LassoCV | 0.5286 | 1.3273 | 1.0000 | 0.6371 | 0.3629 | 1.0000 | 0.9828 |
| AdaptiveLassoCV | 0.5322 | 63.7051 | 1.0000 | 0.6255 | 0.3745 | 1.0000 | 0.1900 |
| GroupLassoCV | 0.1177 | 63.8119 | 1.0000 | 0.9374 | 0.0626 | 1.0000 | 0.1886 |

### Sigma = 1.5 (SNR = 0.7)

| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.8825 | 3.3821 | 1.0000 | 0.2087 | 0.7913 | 1.0000 | 0.9589 |
| LassoCV | 0.5323 | 2.9843 | 1.0000 | 0.6332 | 0.3668 | 1.0000 | 0.9619 |
| AdaptiveLassoCV | 0.4357 | 64.8418 | 0.9980 | 0.7115 | 0.2885 | 0.9980 | 0.1846 |
| GroupLassoCV | 0.1239 | 65.0217 | 1.0000 | 0.9337 | 0.0663 | 1.0000 | 0.1822 |

### Sigma = 2.0 (SNR = 0.5)

| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.8468 | 5.5380 | 0.9800 | 0.2519 | 0.7481 | 0.9800 | 0.9336 |
| LassoCV | 0.5392 | 5.2719 | 1.0000 | 0.6291 | 0.3709 | 1.0000 | 0.9339 |
| AdaptiveLassoCV | 0.3388 | 66.5431 | 0.9840 | 0.7899 | 0.2101 | 0.9840 | 0.1773 |
| GroupLassoCV | 0.1287 | 66.7093 | 0.9960 | 0.9307 | 0.0693 | 0.9960 | 0.1751 |

### Sigma = 3.0 (SNR = 0.3)

| Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-----|-----|-----|-----------|--------|-----|
| AdaptiveFlippedLasso | 0.6928 | 11.5317 | 0.8400 | 0.4084 | 0.5916 | 0.8400 | 0.8676 |
| LassoCV | 0.5231 | 11.7532 | 0.9620 | 0.6389 | 0.3611 | 0.9620 | 0.8601 |
| AdaptiveLassoCV | 0.2370 | 71.5347 | 0.8840 | 0.8605 | 0.1395 | 0.8840 | 0.1589 |
| GroupLassoCV | 0.1244 | 71.7691 | 0.9700 | 0.9328 | 0.0672 | 0.9700 | 0.1560 |

## Rankings by Metric

### By F1 Score (higher is better)
1. AdaptiveFlippedLasso: 0.8861
2. LassoCV: 0.5418
3. AdaptiveLassoCV: 0.4738
4. GroupLassoCV: 0.1235

### By MSE (lower is better)
1. LassoCV: 3.6140
2. AdaptiveFlippedLasso: 3.9721
3. AdaptiveLassoCV: 65.4718
4. GroupLassoCV: 65.5971

### By FDR (lower is better)
1. AdaptiveFlippedLasso: 0.1775
2. LassoCV: 0.6238
3. AdaptiveLassoCV: 0.6670
4. GroupLassoCV: 0.9339

### By TPR (higher is better)
1. GroupLassoCV: 0.9943
2. LassoCV: 0.9937
3. AdaptiveLassoCV: 0.9777
4. AdaptiveFlippedLasso: 0.9700

## Key Findings

### Best Overall Model
- **By F1**: AdaptiveFlippedLasso with F1=0.8861
- **By MSE**: LassoCV with MSE=3.6140
- **By FDR**: AdaptiveFlippedLasso with FDR=0.1775
- **By TPR**: GroupLassoCV with TPR=0.9943

### Observations by SNR

- Sigma=0.1 (SNR=10.0): Best F1 = 0.9902 (AdaptiveFlippedLasso)
- Sigma=0.5 (SNR=2.0): Best F1 = 0.9854 (AdaptiveFlippedLasso)
- Sigma=1.0 (SNR=1.0): Best F1 = 0.9187 (AdaptiveFlippedLasso)
- Sigma=1.5 (SNR=0.7): Best F1 = 0.8825 (AdaptiveFlippedLasso)
- Sigma=2.0 (SNR=0.5): Best F1 = 0.8468 (AdaptiveFlippedLasso)
- Sigma=3.0 (SNR=0.3): Best F1 = 0.6928 (AdaptiveFlippedLasso)

### Limitations

- **Missing Models**: UniLassoCV, FusedLassoCV were not included in this benchmark
- **Only 4 models compared**: The comprehensive 6-model comparison failed to complete

---
*Report generated: 2026-03-27 07:53:08*
