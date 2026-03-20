#!/usr/bin/env python3
"""测试修复后的cv_uni软约束效果"""
import numpy as np
import sys
sys.path.append('..')
from unilasso.uni_lasso import cv_uni
from experiment_ar1 import generate_data

# 生成测试数据
np.random.seed(42)
X_train, y_train, X_test, y_test, beta_true, true_indices = generate_data(
    n=300, p=1000, sigma=1.0, rho=0.8
)

print(f"True non-zero coefficients: {len(true_indices)}")
print(f"True MSE (oracle): {np.mean((X_test @ beta_true - y_test)**2):.4f}")

# 测试软约束XLasso
print("\n=== 测试修复后的cv_uni软约束 ===")
cv_result_soft = cv_uni(
    X_train, y_train, family="gaussian", n_folds=3,
    adaptive_weighting=False,
    enable_group_constraint=False,
    alpha=0.0,
    beta=1.0,
    backend="numba",
    verbose=False
)

# 直接取最佳系数预测
beta_pred = cv_result_soft.coefs[cv_result_soft.best_idx]
intercept = cv_result_soft.intercept[cv_result_soft.best_idx]
y_pred = X_test @ beta_pred + intercept
mse = np.mean((y_pred - y_test) ** 2)
selected = np.sum(np.abs(beta_pred) > 1e-8)

print(f"Test MSE: {mse:.4f}")
print(f"Selected variables: {selected}")
print(f"Best lambda: {cv_result_soft.best_lmda:.6f}")
print(f"Beta range: [{beta_pred.min():.4f}, {beta_pred.max():.4f}]")

# 和基准UniLasso对比
from unilasso.uni_lasso import cv_unilasso, extract_cv
print("\n=== 基准UniLasso结果 ===")
cv_result_uni = cv_unilasso(
    X_train, y_train, family="gaussian", n_folds=3,
    verbose=False
)
fit_uni = extract_cv(cv_result_uni)
y_pred_uni = X_test @ fit_uni.coefs + fit_uni.intercept
mse_uni = np.mean((y_pred_uni - y_test) ** 2)
selected_uni = np.sum(np.abs(fit_uni.coefs) > 1e-8)
print(f"UniLasso MSE: {mse_uni:.4f}")
print(f"Selected variables: {selected_uni}")
