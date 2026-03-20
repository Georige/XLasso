#!/usr/bin/env python3
"""专门测试软约束的MSE问题，找到合适的lambda范围"""
import numpy as np
import sys
sys.path.append('..')
from unilasso.uni_lasso import _prepare_unilasso_input
from unilasso.solvers import _fit_numba_lasso_path_accelerated
from experiment_ar1 import generate_data

# 生成测试数据
np.random.seed(42)
X_train, y_train, X_test, y_test, beta_true, true_indices = generate_data(
    n=300, p=1000, sigma=1.0, rho=0.8
)

print(f"True non-zero coefficients: {len(true_indices)}")
print(f"True MSE (oracle): {np.mean((X_test @ beta_true - y_test)**2):.4f}")
print(f"True beta range: [{beta_true.min():.4f}, {beta_true.max():.4f}]")

# 第一步：预处理
X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, _, original_lmdas, zero_var_idx = _prepare_unilasso_input(X_train, y_train, "gaussian", None)
print(f"\n单变量回归系数范围: [{beta_coefs_fit.min():.4f}, {beta_coefs_fit.max():.4f}]")
print(f"LOO矩阵数值范围: [{loo_fits.min():.4f}, {loo_fits.max():.4f}]")

# 第二步：手动指定lambda范围测试
print("\n=== 手动测试不同lambda效果 ===")
lmdas = np.logspace(-3, 1, 20)  # 从0.001到10的对数序列
feature_weights = np.ones(loo_fits.shape[1])

best_mse = float('inf')
best_lmda = None
best_n_selected = None
best_beta = None

for lmda in lmdas:
    # 拟合单个lambda
    betas, intercepts = _fit_numba_lasso_path_accelerated(
        loo_fits, y, np.array([lmda]),
        alpha=0.0,  # 纯软约束，alpha=0时正负惩罚都是lmda*fw，即普通Lasso
        beta=1.0,
        family="gaussian",
        fit_intercept=True,
        feature_weights=feature_weights
    )

    # 转换为原始特征系数
    theta = betas[0]
    intercept = intercepts[0]
    beta_pred = theta * beta_coefs_fit

    # 计算MSE
    y_pred = X_test @ beta_pred + intercept
    mse = np.mean((y_pred - y_test) ** 2)
    n_selected = np.sum(np.abs(beta_pred) > 1e-8)

    print(f"Lambda={lmda:.6f} | MSE={mse:.4f} | Selected={n_selected} | Beta range=[{beta_pred.min():.4f}, {beta_pred.max():.4f}]")

    if mse < best_mse:
        best_mse = mse
        best_lmda = lmda
        best_n_selected = n_selected
        best_beta = beta_pred

print(f"\n最优结果: Lambda={best_lmda:.6f} | MSE={best_mse:.4f} | Selected={best_n_selected}")

# 和基准UniLasso对比
from unilasso.uni_lasso import cv_unilasso, extract_cv, predict
print("\n=== 基准UniLasso结果 ===")
cv_result_uni = cv_unilasso(
    X_train, y_train, family="gaussian", n_folds=3,
    verbose=False
)
fit_uni = extract_cv(cv_result_uni)
y_pred_uni = X_test @ fit_uni.coefs + fit_uni.intercept
mse_uni = np.mean((y_pred_uni - y_test) ** 2)
selected_uni = np.sum(np.abs(fit_uni.coefs) > 1e-8)
print(f"UniLasso MSE: {mse_uni:.4f} | Selected={selected_uni}")
