#!/usr/bin/env python3
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generator import generate_experiment1
from unilasso.uni_lasso import _prepare_unilasso_input, _configure_lmda_path

# 生成测试数据
X, y, beta_true = generate_experiment1(sigma=0.5)

# 准备输入
X_out, y_out, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, _, lmdas, zero_var_idx = _prepare_unilasso_input(
    X, y, "gaussian", None
)

# 计算lambda路径
lambda_path = _configure_lmda_path(
    X=loo_fits,
    y=y,
    family="gaussian",
    n_lmdas=100,
    lmda_min_ratio=1e-6
)

print(f"lambda范围: {lambda_path.min():.6f} ~ {lambda_path.max():.6f}")
print(f"最大lambda: {lambda_path[0]:.6f}")
print(f"最小lambda: {lambda_path[-1]:.6f}")

# 计算阈值
lr = 0.01
alpha = 1.0
beta = 1.0
fw = 1.0

tau_pos_base = lr * lambda_path[0] * fw
tau_neg_base = lr * lambda_path[0] * (alpha / fw + beta * fw)
print(f"\n最大lambda下的阈值:")
print(f"正阈值: {tau_pos_base:.6f}")
print(f"负阈值: {tau_neg_base:.6f}")

tau_pos_base_min = lr * lambda_path[-1] * fw
tau_neg_base_min = lr * lambda_path[-1] * (alpha / fw + beta * fw)
print(f"\n最小lambda下的阈值:")
print(f"正阈值: {tau_pos_base_min:.6f}")
print(f"负阈值: {tau_neg_base_min:.6f}")

# 查看loo_fits第一行的数值范围
print(f"\nloo_fits第一行系数范围: {loo_fits[0].min():.6f} ~ {loo_fits[0].max():.6f}")
print(f"loo_fits平均系数绝对值: {np.mean(np.abs(loo_fits)):.6f}")
