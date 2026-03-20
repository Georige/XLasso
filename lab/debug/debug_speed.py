#!/usr/bin/env python3
"""调试cv_uni速度问题"""
import numpy as np
import sys
import time
sys.path.append('..')
from unilasso.uni_lasso import _prepare_unilasso_input, _configure_lmda_path
from unilasso.solvers import _fit_numba_lasso_path_accelerated
from experiment_ar1 import generate_data

# 生成测试数据
np.random.seed(42)
X_train, y_train, _, _, _, _ = generate_data(n=300, p=1000, sigma=1.0, rho=0.8)
print(f"Data shape: X={X_train.shape}, y={y_train.shape}")

# 1. 测试_prepare_unilasso_input速度
print("\n1. 测试数据预处理速度：")
start = time.time()
X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, _, original_lmdas, zero_var_idx = _prepare_unilasso_input(X_train, y_train, "gaussian", None)
end = time.time()
print(f"   预处理耗时: {end - start:.2f}s")
print(f"   loo_fits shape: {loo_fits.shape}")

# 2. 测试lambda路径生成
print("\n2. 测试lambda路径生成：")
start = time.time()
lambda_path = _configure_lmda_path(
    X=loo_fits,
    y=y,
    family="gaussian",
    n_lmdas=100,
    lmda_min_ratio=1e-4
)
lambda_path *= 0.0001
end = time.time()
print(f"   lambda生成耗时: {end - start:.2f}s")
print(f"   lambda范围: [{lambda_path.min():.8f}, {lambda_path.max():.8f}]")

# 3. 测试坐标下降拟合速度
print("\n3. 测试坐标下降拟合速度：")
feature_weights = np.ones(loo_fits.shape[1])
start = time.time()
betas, intercepts = _fit_numba_lasso_path_accelerated(
    loo_fits, y, lambda_path,
    alpha=1.0,
    beta=1.0,
    family="gaussian",
    fit_intercept=True,
    feature_weights=feature_weights
)
end = time.time()
print(f"   拟合耗时: {end - start:.2f}s")
print(f"   非零系数平均数量: {np.mean(np.sum(np.abs(betas) > 1e-8, axis=1)):.1f}")
