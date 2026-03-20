#!/usr/bin/env python3
"""测试修复后的cv_uni"""
import numpy as np
import sys
sys.path.append('..')
from unilasso.uni_lasso import cv_uni, extract_cv
from experiment_ar1 import generate_data

# 生成测试数据
np.random.seed(42)
X_train, y_train, X_test, y_test, beta_true, true_indices = generate_data(
    n=300, p=1000, sigma=1.0, rho=0.8
)

print(f"True non-zero coefficients: {len(true_indices)}")

# 测试软约束XLasso
print("\n测试软约束XLasso：")
cv_result_soft = cv_uni(
    X_train, y_train, family="gaussian", n_folds=2,
    adaptive_weighting=False,
    enable_group_constraint=False,
    alpha=0.0,
    beta=1.0,
    backend="numba",
    verbose=False
)
fit_soft = extract_cv(cv_result_soft)
selected = np.sum(np.abs(fit_soft.coefs) > 1e-8)
print(f"Selected variables: {selected}")
print(f"Best lambda: {cv_result_soft.best_lmda:.10f}")
print(f"Coefficient range: [{fit_soft.coefs.min():.4f}, {fit_soft.coefs.max():.4f}]")
