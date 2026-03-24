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
        method: 'exact' - 精确留一（逐特征拟合）
               'approx' - 快速近似（基于单特征Ridge LOO公式）
               'auto' - 自动选择
        n_jobs: 并行线程数
        random_state: 随机种子
        verbose: 是否打印进度
    Returns:
        X_loo: 留一引导矩阵 (n_samples, n_features)
        beta_ridge: 全样本强Ridge系数 (n_features,)
    """
    n, p = X.shape
    X_loo = np.zeros((n, p), dtype=_DTYPE)

    # 第一步：拟合全样本强Ridge得到beta_ridge
    ridge_full = build_ridge_estimator(
        task_type=task_type,
        alpha=lambda_ridge,
        random_state=random_state
    )
    ridge_full.fit(X, y)
    beta_ridge = ridge_full.beta_ridge_

    # 正确逻辑：全模型Ridge系数固定，拆分单变量模型计算LOO
    if verbose:
        print(f"[LOO Constructor] 使用全模型拆分方式构造X_loo, 特征数={p}")

    if task_type == 'regression':
        # 线性回归场景：严格按照官方设计实现
        # 全模型Ridge已经得到beta_ridge，拆分为p个单变量固定系数模型
        # 每个单变量模型：y_j = beta_ridge[j] * X_j，系数固定不重新拟合
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # 对每个特征j，计算固定系数beta_ridge[j]的单变量模型留一预测值
        # 帽子对角元：h_jj = X_j^T (X_j X_j^T + λI)^{-1} X_j
        XTX_diag = np.sum(X_centered ** 2, axis=0)
        h_diag = XTX_diag / (XTX_diag + lambda_ridge + 1e-10)

        # 计算每个特征的单变量预测值（固定全模型beta_ridge[j]）
        y_pred_j = X_centered * beta_ridge + y_mean  # (n,p)

        # 留一公式：y_loo_j = y - (y - y_pred_j) / (1 - h_jj)
        residual = y.reshape(-1, 1) - y_pred_j  # (300,1) - (300,504) → (300,504)
        correction = 1 / (1 - np.clip(h_diag, 1e-10, 1 - 1e-10)).reshape(1, -1)  # (1,504)
        X_loo = y.reshape(-1, 1) - residual * correction  # (300,1) - (300,504)*(1,504) → (300,504)


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
