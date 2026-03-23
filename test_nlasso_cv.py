#!/usr/bin/env python3
"""
测试NLassoCV和NLassoClassifierCV
"""
import sys
sys.path.insert(0, '/home/liangyuneng/XLasso')

import numpy as np
from NLasso import NLassoCV, NLassoClassifierCV

print("="*60)
print("🚀 开始NLassoCV测试")
print("="*60 + "\n")

# ==========================================
# 测试1：NLassoCV回归
# ==========================================
print("📊 测试1：NLassoCV回归")
try:
    np.random.seed(2026)
    n, p = 50, 20
    X = np.random.randn(n, p)
    y = np.random.randn(n)

    # 使用简化的参数网格，快速测试
    small_param_grid = {
        'lambda_ridge': [5.0, 10.0],
        'gamma': [0.3, 0.5],
        'lambda_': [0.1]
    }

    model_cv = NLassoCV(
        param_grid=small_param_grid,
        cv=3,
        scoring='r2',
        n_jobs=1,  # 单进程避免可能的问题
        verbose=True,
        random_state=2026
    )

    model_cv.fit(X, y)

    print(f"\n✅ NLassoCV回归测试成功！")
    print(f"   最优参数: {model_cv.best_params_}")
    print(f"   最优R²: {model_cv.best_score_:.4f}")
    print(f"   系数形状: {model_cv.coef_.shape}")

except Exception as e:
    print(f"\n❌ NLassoCV回归测试失败：{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60 + "\n")

# ==========================================
# 测试2：NLassoClassifierCV分类
# ==========================================
print("📊 测试2：NLassoClassifierCV分类")
try:
    np.random.seed(2026)
    n, p = 50, 20
    X = np.random.randn(n, p)
    y = np.random.randint(0, 2, size=n)

    # 使用简化的参数网格
    small_param_grid = {
        'lambda_ridge': [5.0, 10.0],
        'gamma': [0.3, 0.5],
        'lambda_': [0.1]
    }

    model_clf_cv = NLassoClassifierCV(
        param_grid=small_param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=1,
        verbose=True,
        random_state=2026
    )

    model_clf_cv.fit(X, y)

    print(f"\n✅ NLassoClassifierCV分类测试成功！")
    print(f"   最优参数: {model_clf_cv.best_params_}")
    print(f"   最优准确率: {model_clf_cv.best_score_:.4f}")
    print(f"   系数形状: {model_clf_cv.coef_.shape}")

except Exception as e:
    print(f"\n❌ NLassoClassifierCV分类测试失败：{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("🎉 NLassoCV测试完成！")
print("="*60)
