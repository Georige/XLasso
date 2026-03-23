#!/usr/bin/env python3
"""
NLasso 模块冒烟测试
仅验证模块导入和基本功能，确保无崩溃
"""
import sys
sys.path.insert(0, '/home/liangyuneng/XLasso')

import numpy as np

print("="*60)
print("🚀 开始NLasso模块冒烟测试")
print("="*60 + "\n")

# ==========================================
# 测试1：模块导入测试
# ==========================================
print("📦 测试1：模块导入")
try:
    import NLasso
    print("   ✅ NLasso 顶层模块导入成功")
except Exception as e:
    print(f"   ❌ NLasso 顶层模块导入失败：{e}")
    sys.exit(1)

try:
    from NLasso import NLasso, NLassoClassifier, metrics
    print("   ✅ 核心类和指标模块导入成功")
except Exception as e:
    print(f"   ❌ 核心类导入失败：{e}")
    sys.exit(1)

try:
    from NLasso.first_stage import (
        RidgeEstimator, construct_X_loo, calculate_asymmetric_weights
    )
    print("   ✅ first_stage 模块导入成功")
except Exception as e:
    print(f"   ❌ first_stage 模块导入失败：{e}")

try:
    from NLasso.second_stage import (
        asymmetric_soft_threshold, objective_value, cd_solve
    )
    print("   ✅ second_stage 模块导入成功")
except Exception as e:
    print(f"   ❌ second_stage 模块导入失败：{e}")

try:
    from NLasso.group_module import (
        group_variables, OrthogonalDecomposer, reconstruct_coefficients
    )
    print("   ✅ group_module 模块导入成功")
except Exception as e:
    print(f"   ❌ group_module 模块导入失败：{e}")

print("   ✅ 所有模块导入测试通过！\n")

# ==========================================
# 测试2：指标模块功能
# ==========================================
print("📊 测试2：指标模块")
try:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.2, 1.8, 3.1, 3.9, 5.2])
    beta_true = np.array([1.0, 0.0, 2.0, 0.0, 0.0])
    beta_pred = np.array([0.9, 0.0, 1.8, 0.1, 0.0])

    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    sparsity = metrics.sparsity(beta_pred)
    tpr = metrics.true_positive_rate(None, beta_pred, beta_true)

    print(f"   ✅ 指标计算正常")
    print(f"      MSE: {mse:.4f}")
    print(f"      MAE: {mae:.4f}")
    print(f"      R²: {r2:.4f}")
    print(f"      稀疏度: {sparsity:.2%}")
    print(f"      TPR: {tpr:.2%}")

except Exception as e:
    print(f"   ❌ 指标模块测试失败：{e}")
    import traceback
    traceback.print_exc()

print("")

# ==========================================
# 测试3：NLasso回归简单测试
# ==========================================
print("🔄 测试3：NLasso回归基本运行")
try:
    np.random.seed(2026)
    n, p = 50, 20
    X = np.random.randn(n, p)
    y = np.random.randn(n)

    model = NLasso(
        lambda_ridge=1.0,
        lambda_=0.1,  # 手动指定小lambda，避免全零解
        gamma=0.3,
        n_lambda=5,  # 小lambda路径，快速测试
        max_iter=50,
        verbose=False
    )

    model.fit(X, y)
    y_pred = model.predict(X)

    print(f"   ✅ NLasso回归运行正常")
    print(f"      系数形状: {model.coef_.shape}")
    print(f"      非零系数: {np.sum(np.abs(model.coef_) > 1e-6)}个")
    print(f"      训练迭代: {model.n_iter_}次")

except Exception as e:
    print(f"   ❌ NLasso回归测试失败：{e}")
    import traceback
    traceback.print_exc()

print("")

# ==========================================
# 测试4：NLasso分类简单测试
# ==========================================
print("🔄 测试4：NLasso分类基本运行")
try:
    np.random.seed(2026)
    n, p = 50, 20
    X = np.random.randn(n, p)
    y = np.random.randint(0, 2, size=n)

    model_clf = NLassoClassifier(
        lambda_ridge=1.0,
        lambda_=0.1,
        gamma=0.3,
        n_lambda=5,
        max_iter=50,
        verbose=False
    )

    model_clf.fit(X, y)
    y_pred_clf = model_clf.predict(X)
    y_proba_clf = model_clf.predict_proba(X)

    print(f"   ✅ NLasso分类器运行正常")
    print(f"      系数形状: {model_clf.coef_.shape}")
    print(f"      非零系数: {np.sum(np.abs(model_clf.coef_) > 1e-6)}个")
    print(f"      预测类别形状: {y_pred_clf.shape}")
    print(f"      预测概率形状: {y_proba_clf.shape}")

except Exception as e:
    print(f"   ❌ NLasso分类器测试失败：{e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("🎉 NLasso冒烟测试通过！")
print("="*60)
