#!/usr/bin/env python3
"""调试交叉验证损失为什么这么大"""
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

# 预处理
X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, _, original_lmdas, zero_var_idx = _prepare_unilasso_input(X_train, y_train, "gaussian", None)
print(f"y的范围: [{y.min():.4f}, {y.max():.4f}], y均值: {y.mean():.4f}")

# 手动做2折交叉验证
n_samples = loo_fits.shape[0]
idx = np.arange(n_samples)
np.random.shuffle(idx)
train_idx = idx[:150]
val_idx = idx[150:]

X_train_loo, X_val_loo = loo_fits[train_idx], loo_fits[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# 拟合lambda=0.03
lmdas = np.array([0.03])
betas, intercepts = _fit_numba_lasso_path_accelerated(
    X_train_loo, y_train, lmdas,
    alpha=0.0,
    beta=1.0,
    family="gaussian",
    fit_intercept=True
)

theta = betas[0]
intercept = intercepts[0]

# 在验证集上预测
preds_val = X_val_loo @ theta + intercept
mse_val = np.mean((preds_val - y_val)**2)
print(f"验证集MSE（LOO矩阵直接预测）: {mse_val:.4f}")
print(f"预测值范围: [{preds_val.min():.4f}, {preds_val.max():.4f}]")
print(f"真实y范围: [{y_val.min():.4f}, {y_val.max():.4f}]")

# 转换为原始特征系数预测测试集
beta_pred = theta * beta_coefs_fit
preds_test = X_test @ beta_pred + intercept
mse_test = np.mean((preds_test - y_test)**2)
print(f"测试集MSE（原始特征预测）: {mse_test:.4f}")
