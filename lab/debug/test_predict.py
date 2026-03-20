#!/usr/bin/env python3
"""测试predict方法正确性"""
import numpy as np
import sys
sys.path.append('..')
from unilasso.uni_lasso import cv_uni, predict
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

# 先提取最佳模型再预测
from unilasso.uni_lasso import extract_cv
best_fit = extract_cv(cv_result_soft)
y_pred = predict(best_fit, X_test)
mse = np.mean((y_pred - y_test) ** 2)
selected = np.sum(np.abs(cv_result_soft.coefs[cv_result_soft.best_idx]) > 1e-8)

print(f"Test MSE: {mse:.4f}")
print(f"Selected variables: {selected}")
print(f"Best lambda: {cv_result_soft.best_lmda:.6f}")

# 和基准对比
from unilasso.uni_lasso import cv_unilasso
print("\n基准UniLasso：")
cv_result_uni = cv_unilasso(
    X_train, y_train, family="gaussian", n_folds=2,
    verbose=False
)
y_pred_uni = predict(cv_result_uni, X_test)
mse_uni = np.mean((y_pred_uni - y_test) ** 2)
selected_uni = np.sum(np.abs(cv_result_uni.coefs[cv_result_uni.best_idx]) > 1e-8)
print(f"Test MSE: {mse_uni:.4f}")
print(f"Selected variables: {selected_uni}")
