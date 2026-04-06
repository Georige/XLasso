# Real Dataset Experiment Report

**Date**: 2026-04-03
**Dataset**: riboflavin - Riboflavin gene expression (n=71, p=4088)
**Mode**: random
**CV Folds**: 10
**Train Ratio**: 0.8
**Iterations**: 100

## Model Comparison (Random Split with Internal CV)

| Model | Test MSE (mean±std) | Model Size (mean±std) |
|-------|---------------------|----------------------|
| pfl_regressor_cv | 0.3471 ± 0.2152 | 16.9 ± 6.7 |
| adaptive_lasso_cv | 0.4030 ± 0.2459 | 45.1 ± 23.2 |
| unilasso_cv | 52.0017 ± 1.9677 | 15.0 ± 5.7 |
| lasso_cv | 0.2912 ± 0.1529 | 42.2 ± 12.7 |
| relaxed_lasso_1se | 0.2819 ± 0.1375 | 20.9 ± 7.3 |

### Key Findings

- **Best Test MSE**: relaxed_lasso_1se (0.2819)
- **Most Sparse**: unilasso_cv (15.0 features)

## Feature Selection Frequency Analysis

### Top 10 Most Frequently Selected Features by Model

**pfl_regressor_cv**:
  1. Feature 2563: 97/100 (97.0%)
  2. Feature 1761: 92/100 (92.0%)
  3. Feature 4002: 81/100 (81.0%)
  4. Feature 623: 80/100 (80.0%)
  5. Feature 2026: 66/100 (66.0%)
  6. Feature 4003: 63/100 (63.0%)
  7. Feature 72: 55/100 (55.0%)
  8. Feature 3513: 53/100 (53.0%)
  9. Feature 1302: 49/100 (49.0%)
  10. Feature 1523: 48/100 (48.0%)

**adaptive_lasso_cv**:
  1. Feature 1761: 79/100 (79.0%)
  2. Feature 2563: 78/100 (78.0%)
  3. Feature 1130: 75/100 (75.0%)
  4. Feature 2026: 74/100 (74.0%)
  5. Feature 72: 70/100 (70.0%)
  6. Feature 623: 64/100 (64.0%)
  7. Feature 4002: 63/100 (63.0%)
  8. Feature 1424: 57/100 (57.0%)
  9. Feature 1826: 52/100 (52.0%)
  10. Feature 2873: 49/100 (49.0%)

**unilasso_cv**:
  1. Feature 2563: 87/100 (87.0%)
  2. Feature 4002: 86/100 (86.0%)
  3. Feature 1515: 86/100 (86.0%)
  4. Feature 623: 74/100 (74.0%)
  5. Feature 1311: 69/100 (69.0%)
  6. Feature 1638: 59/100 (59.0%)
  7. Feature 3513: 55/100 (55.0%)
  8. Feature 1296: 54/100 (54.0%)
  9. Feature 1277: 53/100 (53.0%)
  10. Feature 4003: 48/100 (48.0%)

**lasso_cv**:
  1. Feature 2563: 98/100 (98.0%)
  2. Feature 1761: 92/100 (92.0%)
  3. Feature 4002: 80/100 (80.0%)
  4. Feature 623: 80/100 (80.0%)
  5. Feature 72: 75/100 (75.0%)
  6. Feature 1130: 75/100 (75.0%)
  7. Feature 2026: 75/100 (75.0%)
  8. Feature 1638: 68/100 (68.0%)
  9. Feature 4003: 64/100 (64.0%)
  10. Feature 2873: 60/100 (60.0%)

**relaxed_lasso_1se**:
  1. Feature 2563: 89/100 (89.0%)
  2. Feature 4002: 82/100 (82.0%)
  3. Feature 623: 79/100 (79.0%)
  4. Feature 1761: 69/100 (69.0%)
  5. Feature 1515: 68/100 (68.0%)
  6. Feature 1311: 65/100 (65.0%)
  7. Feature 1638: 63/100 (63.0%)
  8. Feature 4003: 57/100 (57.0%)
  9. Feature 1302: 53/100 (53.0%)
  10. Feature 3513: 52/100 (52.0%)


---
*Report generated: 2026-04-03T17:03:04.520701*
