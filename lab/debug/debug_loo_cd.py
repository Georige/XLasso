#!/usr/bin/env python3
"""调试LOO矩阵下的坐标下降"""
import numpy as np
import sys
sys.path.append('..')
from unilasso.uni_lasso import _prepare_unilasso_input
from unilasso.solvers import _fit_numba_lasso_path_accelerated
from experiment_ar1 import generate_data

# 生成测试数据
np.random.seed(42)
X_train, y_train, _, _, _, _ = generate_data(n=300, p=1000, sigma=1.0, rho=0.8)

# 预处理得到loo_fits
X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, _, zero_var_idx = _prepare_unilasso_input(X_train, y_train, "gaussian", None)

print(f"loo_fits shape: {loo_fits.shape}")
print(f"loo_fits数值范围: [{loo_fits.min():.4f}, {loo_fits.max():.4f}]")

# 测试不同的lambda范围
print("\n测试不同lambda在LOO矩阵上的效果：")
lmdas = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])
betas, intercepts = _fit_numba_lasso_path_accelerated(
    loo_fits, y, lmdas,
    alpha=1.0,
    beta=1.0,
    family="gaussian",
    fit_intercept=True
)

for i, lmda in enumerate(lmdas):
    print(f"Lambda={lmda}: 非零系数={np.sum(np.abs(betas[i])>1e-8)}, 系数范围=[{betas[i].min():.4f}, {betas[i].max():.4f}]")
    y_pred = loo_fits @ betas[i] + intercepts[i]
    mse = np.mean((y_pred - y)**2)
    print(f"  MSE: {mse:.4f}")
