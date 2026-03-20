#!/usr/bin/env python3
"""简单测试修复后的效果"""
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
print(f"True MSE (oracle): {np.mean((X_test @ beta_true - y_test)**2):.4f}")

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

print(f"cv_result.coefs shape: {cv_result_soft.coefs.shape}")
print(f"Best index: {cv_result_soft.best_idx}")

# 直接取最佳系数预测
beta_pred = cv_result_soft.coefs[cv_result_soft.best_idx]
intercept = cv_result_soft.intercept[cv_result_soft.best_idx]
y_pred = X_test @ beta_pred + intercept
mse = np.mean((y_pred - y_test) ** 2)
selected = np.sum(np.abs(beta_pred) > 1e-8)

print(f"\nTest MSE: {mse:.4f}")
print(f"Selected variables: {selected}")
print(f"Beta range: [{beta_pred.min():.4f}, {beta_pred.max():.4f}]")
