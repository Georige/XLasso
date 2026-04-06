# Real Dataset Experiment Report

**Date**: 2026-03-31
**Dataset**: riboflavin - Riboflavin gene expression (n=71, p=4088)
**Mode**: random
**CV Folds**: 10
**Train Ratio**: 0.8
**Iterations**: 100

## Model Comparison (Random Split with Internal CV)

| Model | Test MSE (mean±std) | Model Size (mean±std) |
|-------|---------------------|----------------------|
| pfl_regressor_cv | 0.3477 ± 0.2166 | 16.9 ± 7.2 |
| lasso_cv | 0.2912 ± 0.1529 | 42.2 ± 12.7 |
| relaxed_lasso_1se | 0.2819 ± 0.1375 | 20.9 ± 7.3 |

### Key Findings

- **Best Test MSE**: relaxed_lasso_1se (0.2819)
- **Most Sparse**: pfl_regressor_cv (16.9 features)

## Feature Selection Frequency Analysis

### Top 10 Most Frequently Selected Features by Model

**pfl_regressor_cv**:
  1. Feature 2563: 97/100 (97.0%)
  2. Feature 1761: 93/100 (93.0%)
  3. Feature 4002: 81/100 (81.0%)
  4. Feature 623: 80/100 (80.0%)
  5. Feature 2026: 66/100 (66.0%)
  6. Feature 4003: 64/100 (64.0%)
  7. Feature 72: 54/100 (54.0%)
  8. Feature 3513: 53/100 (53.0%)
  9. Feature 1302: 49/100 (49.0%)
  10. Feature 1523: 47/100 (47.0%)

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
*Report generated: 2026-03-31T22:53:38.449309*
