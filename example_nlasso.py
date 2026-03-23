#!/usr/bin/env python3
"""
NLasso 使用示例
展示NLasso算法的基本用法和各种功能
"""
import sys
sys.path.insert(0, '/home/liangyuneng/XLasso')

import numpy as np

print("="*60)
print("📚 NLasso 使用示例")
print("="*60)

# ==========================================
# 示例1：简单回归任务
# ==========================================
print("\n" + "="*60)
print("示例1：简单回归任务")
print("="*60)

from NLasso import NLasso

# 生成模拟数据
np.random.seed(2026)
n, p = 200, 50
X = np.random.randn(n, p)

# 真实系数：前5个非零
beta_true = np.zeros(p)
beta_true[:5] = [1.5, -1.2, 2.0, -0.8, 1.0]

y = X @ beta_true + np.random.randn(n) * 0.5

print(f"\n数据维度: X={X.shape}, y={y.shape}")
print(f"真实非零系数: {np.sum(beta_true != 0)}个")

# 创建并拟合NLasso模型
model = NLasso(
    lambda_ridge=10.0,    # 强Ridge正则化强度
    lambda_=0.1,           # Lasso正则化强度
    gamma=0.3,             # 权重映射陡峭程度
    group_threshold=0.7,   # 高相关变量分组阈值
    max_iter=500,
    verbose=False
)

model.fit(X, y)

# 结果分析
print(f"\n拟合结果:")
print(f"  非零系数: {np.sum(np.abs(model.coef_) > 1e-3)}个")
print(f"  训练R²: {model.score(X, y):.4f}")
print(f"  迭代次数: {model.n_iter_}")

# 预测
y_pred = model.predict(X)
mse = np.mean((y - y_pred) ** 2)
print(f"  预测MSE: {mse:.4f}")


# ==========================================
# 示例2：分类任务
# ==========================================
print("\n" + "="*60)
print("示例2：二分类任务")
print("="*60)

from NLasso import NLassoClassifier

# 生成分类数据
np.random.seed(2026)
n, p = 200, 50
X = np.random.randn(n, p)

# 真实系数
beta_true = np.zeros(p)
beta_true[:5] = [1.0, -1.0, 0.8, -0.8, 0.5]

# 生成标签
logits = X @ beta_true + np.random.randn(n) * 0.3
y = (logits > 0).astype(int)

print(f"\n数据维度: X={X.shape}, y={y.shape}")
print(f"类别分布: y=0: {np.sum(y == 0)}, y=1: {np.sum(y == 1)}")

# 创建并拟合NLasso分类器
model_clf = NLassoClassifier(
    lambda_ridge=10.0,
    lambda_=0.1,
    gamma=0.3,
    max_iter=500,
    verbose=False
)

model_clf.fit(X, y)

# 结果分析
print(f"\n拟合结果:")
print(f"  非零系数: {np.sum(np.abs(model_clf.coef_) > 1e-3)}个")
print(f"  训练准确率: {model_clf.score(X, y):.4f}")

# 预测类别和概率
y_pred = model_clf.predict(X)
y_proba = model_clf.predict_proba(X)
print(f"  预测概率形状: {y_proba.shape}")


# ==========================================
# 示例3：使用交叉验证自动调参
# ==========================================
print("\n" + "="*60)
print("示例3：NLassoCV 自动调参")
print("="*60)

from NLasso import NLassoCV

# 生成数据
np.random.seed(2026)
n, p = 100, 20
X = np.random.randn(n, p)
y = np.random.randn(n)

# 定义参数网格
param_grid = {
    'lambda_ridge': [5.0, 10.0, 20.0],
    'gamma': [0.2, 0.3, 0.5],
    'lambda_': [0.05, 0.1, 0.2]
}

print(f"\n参数网格: {param_grid}")

# 创建NLassoCV
model_cv = NLassoCV(
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=1,
    verbose=False
)

print("\n开始交叉验证...")
model_cv.fit(X, y)

print(f"\n交叉验证结果:")
print(f"  最优参数: {model_cv.best_params_}")
print(f"  最优R²: {model_cv.best_score_:.4f}")
print(f"  非零系数: {np.sum(np.abs(model_cv.coef_) > 1e-3)}个")


# ==========================================
# 示例4：组感知截断功能
# ==========================================
print("\n" + "="*60)
print("示例4：组感知截断功能")
print("="*60)

# 生成有高相关组的数据
np.random.seed(2026)
n, p = 200, 30
X = np.random.randn(n, p)

# 创建高相关变量组
X[:, 1] = X[:, 0] * 0.9 + np.random.randn(n) * 0.1
X[:, 2] = X[:, 0] * 0.8 + np.random.randn(n) * 0.2
X[:, 4] = X[:, 3] * 0.85 + np.random.randn(n) * 0.15

# 真实系数
beta_true = np.zeros(p)
beta_true[0] = 1.0
beta_true[1] = 0.8
beta_true[3] = -1.2

y = X @ beta_true + np.random.randn(n) * 0.5

print(f"\n数据: X={X.shape}")
print(f"高相关组: [0,1,2], [3,4]")

# 测试不同组截断阈值
for threshold in [0.0, 0.3, 0.7]:
    model = NLasso(
        lambda_ridge=10.0,
        lambda_=0.1,
        gamma=0.3,
        group_threshold=0.7,
        group_truncation_threshold=threshold,
        max_iter=500,
        verbose=False
    )
    model.fit(X, y)
    non_zero = np.sum(np.abs(model.coef_) > 1e-3)
    print(f"  group_truncation_threshold={threshold:.1f}: 非零系数={non_zero}个")


# ==========================================
# 示例5：指标计算
# ==========================================
print("\n" + "="*60)
print("示例5：指标计算模块")
print("="*60)

from NLasso import metrics

# 模拟真实值和预测值
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.2, 1.8, 3.1, 3.9, 5.2])
beta_true = np.array([1.0, 0.0, 2.0, 0.0, 0.0])
beta_pred = np.array([0.9, 0.0, 1.8, 0.1, 0.0])

print("\n回归指标:")
print(f"  MSE: {metrics.mean_squared_error(y_true, y_pred):.4f}")
print(f"  MAE: {metrics.mean_absolute_error(y_true, y_pred):.4f}")
print(f"  R²:  {metrics.r2_score(y_true, y_pred):.4f}")

print("\n变量选择指标:")
print(f"  稀疏度: {metrics.sparsity(beta_pred):.2%}")
print(f"  TPR:    {metrics.true_positive_rate(None, beta_pred, beta_true):.2%}")
print(f"  FDR:    {metrics.false_discovery_rate(None, beta_pred, beta_true):.2%}")
print(f"  F1:     {metrics.f1_score(None, beta_pred, beta_true):.4f}")


print("\n" + "="*60)
print("✅ 所有示例运行完成！")
print("="*60)
