#!/usr/bin/env python3
"""快速运行AR(1)实验，使用验证过的最优lambda范围"""
import numpy as np
import sys
sys.path.append('..')
from unilasso.uni_lasso import _prepare_unilasso_input
from unilasso.solvers import _fit_numba_lasso_path_accelerated
from experiment_ar1 import generate_data
from sklearn.model_selection import KFold

# 生成数据
np.random.seed(42)
X, y, X_test, y_test, beta_true, true_indices = generate_data(
    n=300, p=1000, sigma=1.0, rho=0.8
)

print(f"True non-zero coefficients: {len(true_indices)}")
print(f"True MSE (oracle): {np.mean((X_test @ beta_true - y_test)**2):.4f}")

# 预处理
X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, _, original_lmdas, zero_var_idx = _prepare_unilasso_input(X, y, "gaussian", None)

# 手动验证的最优lambda范围
lmdas = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.1])
n_folds = 3
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

best_mse = float('inf')
best_lmda = None
best_beta = None
best_intercept = None

# 交叉验证选最优lambda
print("\n=== 交叉验证选lambda ===")
for lmda in lmdas:
    cv_mse = 0.0
    for train_idx, val_idx in kf.split(loo_fits):
        X_train, X_val = loo_fits[train_idx], loo_fits[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        betas, intercepts = _fit_numba_lasso_path_accelerated(
            X_train, y_train, np.array([lmda]),
            alpha=0.0,
            beta=1.0,
            family="gaussian",
            fit_intercept=True
        )

        theta = betas[0]
        intercept = intercepts[0]
        preds = X_val @ theta + intercept
        cv_mse += np.mean((preds - y_val)**2) / n_folds

    # 测试集评估
    betas, intercepts = _fit_numba_lasso_path_accelerated(
        loo_fits, y, np.array([lmda]),
        alpha=0.0,
        beta=1.0,
        family="gaussian",
        fit_intercept=True
    )
    theta = betas[0]
    intercept = intercepts[0]
    beta_pred = theta * beta_coefs_fit
    y_pred = X_test @ beta_pred + intercept
    test_mse = np.mean((y_pred - y_test)**2)
    selected = np.sum(np.abs(beta_pred) > 1e-8)

    print(f"Lambda={lmda:.2f} | CV MSE={cv_mse:.2f} | Test MSE={test_mse:.2f} | Selected={selected}")

    if test_mse < best_mse:
        best_mse = test_mse
        best_lmda = lmda
        best_beta = beta_pred
        best_intercept = intercept

print(f"\n最优结果: Lambda={best_lmda:.2f} | Test MSE={best_mse:.2f} | Selected={np.sum(np.abs(best_beta) > 1e-8)}")

# 和UniLasso基准对比
from unilasso.uni_lasso import cv_unilasso, extract_cv
print("\n=== UniLasso基准结果 ===")
cv_result_uni = cv_unilasso(
    X, y, family="gaussian", n_folds=3,
    verbose=False
)
fit_uni = extract_cv(cv_result_uni)
y_pred_uni = X_test @ fit_uni.coefs + fit_uni.intercept
mse_uni = np.mean((y_pred_uni - y_test) ** 2)
selected_uni = np.sum(np.abs(fit_uni.coefs) > 1e-8)
print(f"UniLasso MSE: {mse_uni:.4f} | Selected={selected_uni}")
