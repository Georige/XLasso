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

    if method == 'auto':
        # 自动选择策略：特征数<200用精确法，否则用近似法
        method = 'exact' if p < 200 else 'approx'

    if method == 'exact':
        # 精确方法：对每个特征j，拟合单变量强Ridge计算LOO贡献
        if verbose:
            print(f"[LOO Constructor] 使用精确方法构造X_loo, 特征数={p}")

        # 并行优化（如果n_jobs>1）
        if n_jobs != 1 and p > 10:
            from joblib import Parallel, delayed

            def process_feature(j):
                X_j = X[:, j:j+1]  # 单特征矩阵 (n,1)
                ridge_j = build_ridge_estimator(
                    task_type=task_type,
                    alpha=lambda_ridge,
                    random_state=random_state
                )
                ridge_j.fit(X_j, y)
                return ridge_j.predict_loo()

            results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(process_feature)(j) for j in range(p)
            )
            for j in range(p):
                X_loo[:, j] = results[j]

        else:
            # 串行版本
            for j in range(p):
                X_j = X[:, j:j+1]
                ridge_j = build_ridge_estimator(
                    task_type=task_type,
                    alpha=lambda_ridge,
                    random_state=random_state
                )
                ridge_j.fit(X_j, y)
                X_loo[:, j] = ridge_j.predict_loo()

    else:
        # 快速近似方法：利用全样本Ridge结果近似计算单特征LOO
        # 核心假设：强Ridge下不同特征的拟合近似独立，误差可接受
        if verbose:
            print(f"[LOO Constructor] 使用近似方法构造X_loo, 特征数={p}")

        # 中心化数据
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # 计算每个特征的单变量Ridge系数
        # 单变量Ridge闭式解：beta_j = (X_j^T y) / (X_j^T X_j + λ)
        XTX_diag = np.sum(X_centered ** 2, axis=0)
        XTy = safe_sparse_dot(X_centered.T, y_centered)
        beta_single = XTy / (XTX_diag + lambda_ridge + 1e-10)

        # 计算单变量Ridge的帽子对角元 h_jj = X_j^T (X_j X_j^T + λI)^{-1} X_j
        h_diag = XTX_diag / (XTX_diag + lambda_ridge + 1e-10)

        # 计算单变量Ridge残差和LOO预测值
        for j in range(p):
            # 对每个特征单独计算
            y_pred_j = X_centered[:, j] * beta_single[j] + y_mean
            residual_j = y - y_pred_j
            y_loo_j = (residual_j / (1 - np.clip(h_diag[j], 1e-10, 1 - 1e-10))) + y_pred_j
            X_loo[:, j] = y_loo_j

    return X_loo, beta_ridge
