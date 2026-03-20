#!/usr/bin/env python3
"""调试lambda尺度问题"""
import numpy as np
import sys
sys.path.append('..')
from unilasso.uni_lasso import _prepare_unilasso_input, _configure_lmda_path
from experiment_ar1 import generate_data

# 生成和实验一样的数据
np.random.seed(42)
X_train, y_train, _, _, _, _ = generate_data(n=300, p=1000, sigma=1.0, rho=0.8)

# 预处理得到loo_fits
X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, _, zero_var_idx = _prepare_unilasso_input(X_train, y_train, "gaussian", None)

print(f"loo_fits shape: {loo_fits.shape}")
print(f"y shape: {y.shape}")

# 生成lambda路径
lambda_path = _configure_lmda_path(
    X=loo_fits,
    y=y,
    family="gaussian",
    n_lmdas=100,
    lmda_min_ratio=1e-4
)

print(f"\n原始lambda路径范围: [{lambda_path.min():.8f}, {lambda_path.max():.8f}]")
print(f"乘以0.0001后范围: [{(lambda_path*0.0001).min():.8f}, {(lambda_path*0.0001).max():.8f}]")
print(f"乘以0.00001后范围: [{(lambda_path*0.00001).min():.8f}, {(lambda_path*0.00001).max():.8f}]")
print(f"乘以0.000001后范围: [{(lambda_path*0.000001).min():.8f}, {(lambda_path*0.000001).max():.8f}]")

# 看看坐标下降里的阈值大概是多少
feature_weights = np.ones(loo_fits.shape[1])
alpha = 1.0
beta = 1.0
lmda_max = lambda_path.max() * 0.0001
tau_pos_base = lmda_max * feature_weights[0]
tau_neg_base = lmda_max * (alpha / max(feature_weights[0], 1e-10) + beta * feature_weights[0])
print(f"\n当前最大lambda对应的阈值:")
print(f"  tau_pos: {tau_pos_base:.8f}")
print(f"  tau_neg: {tau_neg_base:.8f}")

# 看看单变量回归的系数大概是多少
print(f"\n单变量回归系数范围: [{beta_coefs_fit.min():.4f}, {beta_coefs_fit.max():.4f}]")
print(f"单变量回归系数绝对值均值: {np.mean(np.abs(beta_coefs_fit)):.4f}")
