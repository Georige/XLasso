"""
第一阶段：LOO引导矩阵X_loo构造
X_loo[i,j] = 第j个特征在第i个样本上的Ridge留一贡献
"""
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from .ridge_estimator import build_ridge_estimator
from ..base import _DTYPE, _COPY_WHEN_POSSIBLE


def construct_X_loo(
    X: np.ndarray,
    y: np.ndarray,
    lambda_ridge: float = 10.0,
    task_type: str = 'regression',
    method: str = 'auto',
    n_jobs: int = -1,
    random_state: int = 2026,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    构造留一引导矩阵X_loo
    Args:
        X: 特征矩阵 (n_samples, n_features)
        y: 响应变量 (n_samples,)
        lambda_ridge: 强Ridge正则化强度
        task_type: 'regression' / 'classification'
        method: 'exact_sherman_morrison' - 基于Sherman-Morrison公式的高效精确实现（默认推荐）
               'exact_loo' - 逐样本拟合全模型Ridge的暴力精确实现（定义级实现）
               'approx_fast' - 极速近似版（基于组感知正交特性的快速近似）
               'auto' - 自动选择最优方法
        n_jobs: 并行线程数
        random_state: 随机种子
        verbose: 是否打印进度
    Returns:
        X_loo: 留一引导矩阵 (n_samples, n_features)
        beta_ridge: 全样本强Ridge系数 (n_features,)
    """
    n, p = X.shape
    X_loo = np.zeros((n, p), dtype=_DTYPE)

    # 第一步：拟合全样本强Ridge得到beta_ridge（用于权重计算和预测阶段）
    ridge_full = build_ridge_estimator(
        task_type=task_type,
        alpha=lambda_ridge,
        random_state=random_state
    )
    ridge_full.fit(X, y)
    beta_ridge = ridge_full.beta_ridge_

    if verbose:
        print(f"[LOO Constructor] 使用全模型留一方式构造X_loo, 特征数={p}")

    if task_type == 'regression':
        # 线性回归场景：支持两种精确实现
        if method == 'exact_sherman_morrison' or (method == 'auto' and n < 1000 and p < 5000):
            # 实现1：基于Sherman-Morrison公式的高效精确实现（默认推荐）
            # 时间复杂度O(p^3 + np^2)，比逐样本拟合快1-2个数量级
            if verbose:
                print(f"[LOO Constructor] 使用Sherman-Morrison公式构造X_loo，n={n}, p={p}")

            # 计算全样本预测值和残差
            y_pred_full = X @ beta_ridge
            e = y - y_pred_full  # (n,) 全局真实残差

            # 计算全局逆协方差矩阵S
            XTX = X.T @ X
            np.fill_diagonal(XTX, XTX.diagonal() + lambda_ridge + 1e-10)
            S = np.linalg.inv(XTX)  # (p, p) 逆协方差矩阵

            # 计算每个样本的全局杠杆率H_ii
            H = np.diag(X @ S @ X.T)  # (n,) 每个样本的杠杆率

            # 预计算S @ X.T 避免重复计算
            S_XT = S @ X.T  # (p, n)

            # 向量化计算所有样本的系数扰动
            delta_beta = S_XT * (e / (1 - H + 1e-10))  # (p, n) 每列是一个样本的Δβ

            # 最终X_loo计算：X[i,j] * (beta_ridge[j] - delta_beta[j,i])
            X_loo = X * (beta_ridge - delta_beta.T)  # 向量化实现，无循环

        elif method == 'exact_loo' or (method == 'auto' and n >= 1000 and n * p < 1e7):
            # 实现2：逐样本拟合全模型Ridge的暴力精确实现（定义级实现）
            # 时间复杂度O(n p^3)，适合n小的场景或验证使用
            if verbose:
                print(f"[LOO Constructor] 逐样本拟合全模型Ridge构造X_loo，共{n}个样本")

            for i in range(n):
                # 去掉第i个样本，构造训练集
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i, axis=0)

                # 拟合全模型Ridge（带lambda_ridge正则化）
                XTX_i = X_train.T @ X_train
                np.fill_diagonal(XTX_i, XTX_i.diagonal() + lambda_ridge + 1e-10)
                XTy_i = X_train.T @ y_train
                beta_ridge_i = np.linalg.solve(XTX_i, XTy_i)

                # 按定义计算X_loo[i,j]
                X_loo[i, :] = X[i, :] * beta_ridge_i

        elif method == 'approx_fast' or (method == 'auto' and n * p >= 1e7):
            # 实现3：极速近似版（基于组感知正交特性的快速近似）
            # 时间复杂度O(np)，适合超大规模数据集
            # 利用组感知预处理后X_trans高度正交的特性，忽略特征间相关性
            if verbose:
                print(f"[LOO Constructor] 使用极速近似版构造X_loo，n={n}, p={p}")

            # 计算全局残差
            y_pred_full = X @ beta_ridge
            e = y - y_pred_full  # (n,) 全局真实残差

            # 计算单变量杠杆率h_ij
            XTX_diag = np.sum(X ** 2, axis=0)
            h_ij = (X ** 2) / (XTX_diag + lambda_ridge + 1e-10)  # (n,p)

            # 计算全局多变量杠杆率H_ii = sum(h_ij, axis=1)
            H_ii = np.sum(h_ij, axis=1)  # (n,)

            # 极速近似X_loo计算
            denom = 1 - H_ii + 1e-10
            X_loo = X * beta_ridge - (e.reshape(-1, 1) * h_ij) / denom.reshape(-1, 1)


    else:
        # 分类等其他场景：逐特征计算（复用全模型beta_ridge）
        for j in range(p):
            # 固定单变量模型系数为全模型的beta_ridge[j]
            beta_j = beta_ridge[j]

            # 单变量模型预测值
            y_pred_j = beta_j * X[:, j]

            # 计算留一预测值（广义线性模型留一公式）
            # 注：分类场景后续补充完整实现，当前先保证线性回归正确
            X_loo[:, j] = y_pred_j  # 临时实现，后续完善

    return X_loo, beta_ridge
