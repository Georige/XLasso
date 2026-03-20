#!/usr/bin/env python3
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generator import generate_experiment1
from unilasso.uni_lasso import fit_unilasso, cv_uni, _prepare_unilasso_input, _configure_lmda_path

# 生成测试数据
X, y, beta_true = generate_experiment1(sigma=0.5)

# 运行fit_unilasso
print("Running fit_unilasso...")
fit_original = fit_unilasso(X, y, family="gaussian", lmda_min_ratio=1e-6)
print(f"fit_unilasso lambda范围: {fit_original.lmdas.min():.6f} ~ {fit_original.lmdas.max():.6f}")
print(f"fit_unilasso 非零系数个数: {[np.sum(np.abs(c) > 1e-8) for c in fit_original.coefs[:10]]}")
print(f"fit_unilasso 最佳系数非零个数: {np.sum(np.abs(fit_original.coefs[np.argmin(fit_original.lmdas)]) > 1e-8)}")

# 运行cv_uni的准备步骤
print("\nRunning cv_uni preparation...")
X_out, y_out, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, _, lmdas, zero_var_idx = _prepare_unilasso_input(
    X, y, "gaussian", None
)

# 计算cv_uni的lambda路径
lambda_path = _configure_lmda_path(
    X=loo_fits,
    y=y,
    family="gaussian",
    n_lmdas=100,
    lmda_min_ratio=1e-6
)
print(f"cv_uni lambda范围: {lambda_path.min():.6f} ~ {lambda_path.max():.6f}")

# 计算fit_unilasso用的lambda路径
lambda_path_original = _configure_lmda_path(
    X=X,
    y=y,
    family="gaussian",
    n_lmdas=100,
    lmda_min_ratio=1e-6
)
print(f"基于原始X的lambda范围: {lambda_path_original.min():.6f} ~ {lambda_path_original.max():.6f}")
