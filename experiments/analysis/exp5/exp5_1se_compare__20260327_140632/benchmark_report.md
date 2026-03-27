# AdaptiveFlippedLasso Benchmark Report

**Date**: 2026-03-27
**Experiment**: exp5_1se_compare
**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)
**Repeats**: 5 per configuration

## 1. AdaptiveFlippedLasso Optimal Parameters

From Stage1 grid search:
- lambda_ridge: 1.0
- lambda_: 1.0
- gamma: 0.5

## 2. Compared Models

- **UniLassoCV_1SE** (CV-tuned)
- **LassoCV_1SE** (CV-tuned)
- **AFLCV_1SE** (CV-tuned)
- **RelaxedLassoCV_1SE** (CV-tuned)
- **ElasticNetCV_1SE** (CV-tuned)
- **AdaptiveLassoCV_1SE** (CV-tuned)

## 3. Overall Performance by Model

### 3.1 F1 Score (higher is better)

| Model | F1 Mean | F1 Std |
|-------|---------|--------|
| LassoCV_1SE | 0.0016 | 0.0037 |
| UniLassoCV_1SE | 0.0015 | 0.0033 |
| AdaptiveLassoCV_1SE | 0.0000 | 0.0000 |
| AFLCV_1SE | 0.0000 | 0.0000 |
| ElasticNetCV_1SE | 0.0000 | 0.0000 |
| RelaxedLassoCV_1SE | 0.0000 | 0.0000 |

### 3.2 MSE (lower is better)

| Model | MSE Mean | MSE Std |
|-------|----------|--------|
| AdaptiveLassoCV_1SE | 0.9932 | 0.0623 |
| ElasticNetCV_1SE | 0.9932 | 0.0623 |
| LassoCV_1SE | 1.0044 | 0.0568 |
| RelaxedLassoCV_1SE | 1.0140 | 0.0634 |
| AFLCV_1SE | 1.3783 | 0.3404 |
| UniLassoCV_1SE | 1.4041 | 0.3380 |

### 3.3 TPR - True Positive Rate (higher is better)

| Model | TPR Mean | TPR Std |
|-------|----------|--------|
| UniLassoCV_1SE | 0.0020 | 0.0045 |
| LassoCV_1SE | 0.0020 | 0.0045 |
| AdaptiveLassoCV_1SE | 0.0000 | 0.0000 |
| AFLCV_1SE | 0.0000 | 0.0000 |
| ElasticNetCV_1SE | 0.0000 | 0.0000 |
| RelaxedLassoCV_1SE | 0.0000 | 0.0000 |

### 3.4 FDR - False Discovery Rate (lower is better)

| Model | FDR Mean | FDR Std |
|-------|----------|--------|
| AFLCV_1SE | 0.0000 | 0.0000 |
| AdaptiveLassoCV_1SE | 0.0000 | 0.0000 |
| ElasticNetCV_1SE | 0.0000 | 0.0000 |
| LassoCV_1SE | 0.3186 | 0.2275 |
| RelaxedLassoCV_1SE | 0.5600 | 0.2966 |
| UniLassoCV_1SE | 0.7588 | 0.1652 |

## 4. Performance Across SNR Levels

### 4.1 F1 by Sigma

|   sigma |   AFLCV_1SE |   AdaptiveLassoCV_1SE |   ElasticNetCV_1SE |   LassoCV_1SE |   RelaxedLassoCV_1SE |   UniLassoCV_1SE |
|--------:|------------:|----------------------:|-------------------:|--------------:|---------------------:|-----------------:|
|       1 |           0 |                     0 |                  0 |        0.0016 |                    0 |           0.0015 |

### 4.2 MSE by Sigma

|   sigma |   AFLCV_1SE |   AdaptiveLassoCV_1SE |   ElasticNetCV_1SE |   LassoCV_1SE |   RelaxedLassoCV_1SE |   UniLassoCV_1SE |
|--------:|------------:|----------------------:|-------------------:|--------------:|---------------------:|-----------------:|
|       1 |      1.3783 |                0.9932 |             0.9932 |        1.0044 |                1.014 |           1.4041 |

### 4.3 TPR by Sigma

|   sigma |   AFLCV_1SE |   AdaptiveLassoCV_1SE |   ElasticNetCV_1SE |   LassoCV_1SE |   RelaxedLassoCV_1SE |   UniLassoCV_1SE |
|--------:|------------:|----------------------:|-------------------:|--------------:|---------------------:|-----------------:|
|       1 |           0 |                     0 |                  0 |         0.002 |                    0 |            0.002 |

### 4.4 FDR by Sigma

|   sigma |   AFLCV_1SE |   AdaptiveLassoCV_1SE |   ElasticNetCV_1SE |   LassoCV_1SE |   RelaxedLassoCV_1SE |   UniLassoCV_1SE |
|--------:|------------:|----------------------:|-------------------:|--------------:|---------------------:|-----------------:|
|       1 |           0 |                     0 |                  0 |        0.3186 |                 0.56 |           0.7588 |

## 5. Complete Metrics Summary

| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |
|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|
| 1.0 | 1.0 | UniLassoCV_1SE | 0.0015 | 1.4041 | 0.0020 | 0.7588 | 0.0012 | 0.0020 | -0.4603 |
| 1.0 | 1.0 | LassoCV_1SE | 0.0016 | 1.0044 | 0.0020 | 0.3186 | 0.0014 | 0.0020 | -0.0362 |
| 1.0 | 1.0 | AFLCV_1SE | 0.0000 | 1.3783 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.4339 |
| 1.0 | 1.0 | RelaxedLassoCV_1SE | 0.0000 | 1.0140 | 0.0000 | 0.5600 | 0.0000 | 0.0000 | -0.0449 |
| 1.0 | 1.0 | ElasticNetCV_1SE | 0.0000 | 0.9932 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0240 |
| 1.0 | 1.0 | AdaptiveLassoCV_1SE | 0.0000 | 0.9932 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0240 |

## 6. Rankings Summary

### 6.1 By F1 (higher is better)

- rank_1: LassoCV_1SE (F1=0.0016)
- rank_2: UniLassoCV_1SE (F1=0.0015)
- rank_3: AdaptiveLassoCV_1SE (F1=0.0000)
- rank_4: AFLCV_1SE (F1=0.0000)
- rank_5: ElasticNetCV_1SE (F1=0.0000)
- rank_6: RelaxedLassoCV_1SE (F1=0.0000)

### 6.2 By MSE (lower is better)

- rank_1: AdaptiveLassoCV_1SE (MSE=0.9932)
- rank_2: ElasticNetCV_1SE (MSE=0.9932)
- rank_3: LassoCV_1SE (MSE=1.0044)
- rank_4: RelaxedLassoCV_1SE (MSE=1.0140)
- rank_5: AFLCV_1SE (MSE=1.3783)
- rank_6: UniLassoCV_1SE (MSE=1.4041)

## 7. Key Findings

1. **Best F1**: LassoCV_1SE with F1=0.0016
2. **Best MSE**: AdaptiveLassoCV_1SE with MSE=0.9932

---
*Report generated: 2026-03-27T14:20:24.219319*
