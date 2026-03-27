# Real Dataset Experiment Report

**Date**: 2026-03-27
**Dataset**: tecator - Tecator near-infrared spectra (n=133, p=1024)
**Mode**: random
**Iterations**: 10
**Train Ratio**: 0.7

## Model Comparison (Random Split)

| Model | Test MSE (mean±std) | Model Size (mean±std) |
|-------|---------------------|----------------------|
| adaptive_flipped_lasso_cv | 45.3335 ± 4.1015 | 82.4 ± 29.4 |
| lasso_cv | 0.6793 ± 0.1915 | 25.9 ± 12.6 |

### Key Findings

- **Best Test MSE**: lasso_cv (0.6793)
- **Most Sparse**: lasso_cv (25.9 features)

## Feature Selection Frequency Analysis

### Top 10 Most Frequently Selected Features by Model

**adaptive_flipped_lasso_cv**:
  1. Feature 472: 10/10 (100.0%)
  2. Feature 156: 10/10 (100.0%)
  3. Feature 34: 10/10 (100.0%)
  4. Feature 88: 8/10 (80.0%)
  5. Feature 959: 8/10 (80.0%)
  6. Feature 282: 8/10 (80.0%)
  7. Feature 491: 8/10 (80.0%)
  8. Feature 584: 8/10 (80.0%)
  9. Feature 676: 8/10 (80.0%)
  10. Feature 165: 8/10 (80.0%)

**lasso_cv**:
  1. Feature 984: 10/10 (100.0%)
  2. Feature 156: 10/10 (100.0%)
  3. Feature 34: 10/10 (100.0%)
  4. Feature 472: 10/10 (100.0%)
  5. Feature 676: 9/10 (90.0%)
  6. Feature 1009: 9/10 (90.0%)
  7. Feature 625: 7/10 (70.0%)
  8. Feature 624: 7/10 (70.0%)
  9. Feature 668: 7/10 (70.0%)
  10. Feature 64: 6/10 (60.0%)


---
*Report generated: 2026-03-27T21:41:03.793563*
