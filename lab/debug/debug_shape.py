#!/usr/bin/env python3
"""调试维度问题"""
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

# 先看预处理后的形状
X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, _, original_lmdas, zero_var_idx = _prepare_unilasso_input(X_train, y_train, "gaussian", None)
print(f"beta_coefs_fit shape: {beta_coefs_fit.shape}")
print(f"loo_fits shape: {loo_fits.shape}")

# 拟合模型
cv_result_soft = cv_uni(
    X_train, y_train, family="gaussian", n_folds=3,
    adaptive_weighting=False,
    enable_group_constraint=False,
    alpha=0.0,
    beta=1.0,
    backend="numba",
    verbose=False
)

print(f"\ncv_result_soft.coefs shape: {cv_result_soft.coefs.shape}")
print(f"cv_result_soft.intercept shape: {cv_result_soft.intercept.shape}")
print(f"best_idx: {cv_result_soft.best_idx}")

beta_pred = cv_result_soft.coefs[cv_result_soft.best_idx]
intercept = cv_result_soft.intercept[cv_result_soft.best_idx]
print(f"beta_pred shape: {beta_pred.shape}, type: {type(beta_pred)}")
print(f"intercept shape: {intercept.shape}, type: {type(intercept)}")
