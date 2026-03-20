#!/usr/bin/env python3
"""调试交叉验证损失曲线"""
import numpy as np
import sys
sys.path.append('..')
from unilasso.uni_lasso import cv_uni
from experiment_ar1 import generate_data

# 生成测试数据
np.random.seed(42)
X_train, y_train, X_test, y_test, beta_true, true_indices = generate_data(
    n=300, p=1000, sigma=1.0, rho=0.8
)

# 测试软约束XLasso
print("测试软约束XLasso：")
cv_result_soft = cv_uni(
    X_train, y_train, family="gaussian", n_folds=3,
    adaptive_weighting=False,
    enable_group_constraint=False,
    alpha=0.0,
    beta=1.0,
    backend="numba",
    verbose=False
)

print(f"lambda路径: {cv_result_soft.lmdas.round(6)}")
print(f"交叉验证损失: {cv_result_soft.avg_losses.round(2)}")
print(f"最优lambda索引: {cv_result_soft.best_idx}, 最优lambda: {cv_result_soft.lmdas[cv_result_soft.best_idx]:.6f}")
print(f"最小交叉验证损失: {cv_result_soft.avg_losses.min():.2f}")

# 打印每个lambda对应的测试集MSE
print("\n每个lambda对应的测试集MSE：")
for i, lmda in enumerate(cv_result_soft.lmdas):
    beta_pred = cv_result_soft.coefs[i]
    intercept = cv_result_soft.intercept[i]
    y_pred = X_test @ beta_pred + intercept
    mse = np.mean((y_pred - y_test) ** 2)
    selected = np.sum(np.abs(beta_pred) > 1e-8)
    print(f"Lambda={lmda:.6f} | CV Loss={cv_result_soft.avg_losses[i]:.2f} | Test MSE={mse:.2f} | Selected={selected}")
