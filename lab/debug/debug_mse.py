#!/usr/bin/env python3
"""调试MSE过大问题"""
import numpy as np
import sys
sys.path.append('..')
from unilasso.uni_lasso import cv_uni, _prepare_unilasso_input
from experiment_ar1 import generate_data

# 生成测试数据
np.random.seed(42)
X_train, y_train, X_test, y_test, beta_true, true_indices = generate_data(
    n=300, p=1000, sigma=1.0, rho=0.8
)

print(f"True non-zero coefficients: {len(true_indices)}")

# 第一步：预处理得到beta_coefs_fit
X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, _, original_lmdas, zero_var_idx = _prepare_unilasso_input(X_train, y_train, "gaussian", None)
print(f"单变量回归系数范围: [{beta_coefs_fit.min():.4f}, {beta_coefs_fit.max():.4f}]")

# 拟合模型
print("\n拟合软约束XLasso：")
cv_result_soft = cv_uni(
    X_train, y_train, family="gaussian", n_folds=2,
    adaptive_weighting=False,
    enable_group_constraint=False,
    alpha=0.0,
    beta=1.0,
    backend="numba",
    verbose=False
)

# 错误用法：直接用gamma系数乘原始X
gamma_coefs = cv_result_soft.coefs[cv_result_soft.best_idx]
intercept = cv_result_soft.intercept[cv_result_soft.best_idx]
y_pred_wrong = X_test @ gamma_coefs + intercept
mse_wrong = np.mean((y_pred_wrong - y_test) ** 2)
print(f"错误用法MSE: {mse_wrong:.4f}")
print(f"gamma系数范围: [{gamma_coefs.min():.4f}, {gamma_coefs.max():.4f}]")

# 正确用法：转换为原始特征系数
beta_pred = gamma_coefs * beta_coefs_fit
y_pred_correct = X_test @ beta_pred + intercept
mse_correct = np.mean((y_pred_correct - y_test) ** 2)
print(f"正确用法MSE: {mse_correct:.4f}")
print(f"原始特征系数范围: [{beta_pred.min():.4f}, {beta_pred.max():.4f}]")
print(f"选到的变量数: {np.sum(np.abs(beta_pred) > 1e-8)}")

# 和基准UniLasso对比
from unilasso.uni_lasso import cv_unilasso, extract_cv
print("\n基准UniLasso：")
cv_result_uni = cv_unilasso(
    X_train, y_train, family="gaussian", n_folds=2,
    verbose=False
)
fit_uni = extract_cv(cv_result_uni)
y_pred_uni = X_test @ fit_uni.coefs + fit_uni.intercept
mse_uni = np.mean((y_pred_uni - y_test) ** 2)
print(f"UniLasso MSE: {mse_uni:.4f}")
print(f"选到的变量数: {np.sum(np.abs(fit_uni.coefs) > 1e-8)}")
