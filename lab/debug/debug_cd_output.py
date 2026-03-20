#!/usr/bin/env python3
"""调试坐标下降输出"""
import numpy as np
import sys
sys.path.append('..')
from unilasso.solvers import _fit_numba_lasso_path_accelerated
from experiment_ar1 import generate_data

# 生成测试数据
np.random.seed(42)
X_train, y_train, _, _, _, _ = generate_data(n=300, p=1000, sigma=1.0, rho=0.8)

# 直接用原始X和y测试坐标下降，不用LOO矩阵
print("直接用原始X测试坐标下降：")
lmdas = np.array([0.001, 0.01, 0.1, 0.5, 1.0])
betas, intercepts = _fit_numba_lasso_path_accelerated(
    X_train, y_train, lmdas,
    alpha=1.0,
    beta=1.0,
    family="gaussian",
    fit_intercept=True
)

print("\n结果：")
for i, lmda in enumerate(lmdas):
    print(f"Lambda={lmda}: 非零系数={np.sum(np.abs(betas[i])>1e-8)}, 系数范围=[{betas[i].min():.4f}, {betas[i].max():.4f}]")
    if np.any(np.isnan(betas[i])):
        print("  警告：包含NaN值！")
