"""
第二阶段：Numba加速坐标下降求解器
对应paper 3.5节
"""
import numpy as np
from numba import jit, prange
from typing import Optional
from .asymmetric_penalty import asymmetric_soft_threshold, objective_value
from ..base import _DTYPE


@jit(nopython=True, fastmath=True, parallel=False)  # 坐标下降是串行更新，不需要并行
def cd_solve(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lambda_: float,
    theta_init: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    verbose: bool = False
) -> tuple[np.ndarray, float, int]:
    """
    坐标下降求解非对称Lasso优化问题
    Args:
        X: 设计矩阵 (n_samples, n_features)
        y: 响应变量 (n_samples,)
        weights: 非对称权重矩阵 (n_features, 2) -> [w_plus, w_minus]
        lambda_: 正则化强度
        theta_init: 初始系数向量 (n_features,) 可选，用于warm start
        max_iter: 最大迭代次数
        tol: 收敛阈值（系数最大变化量小于tol则停止）
        verbose: 是否打印迭代信息
    Returns:
        theta: 最优系数向量 (n_features,)
        obj: 最优目标函数值
        n_iter: 实际迭代次数
    """
    n, p = X.shape
    n_float = float(n)

    # 初始化系数
    if theta_init is None:
        theta = np.zeros(p, dtype=_DTYPE)
    else:
        theta = theta_init.astype(_DTYPE, copy=True)

    # 预计算每个特征的平方和（分母）
    XTX_diag = np.sum(X ** 2, axis=0) / n_float + 1e-10  # 加小量防止除零

    # 初始化残差
    residuals = y - np.dot(X, theta)

    for iteration in range(max_iter):
        max_change = 0.0

        for j in range(p):
            # 跳过权重都为0的特征
            w_plus, w_minus = weights[j]
            if w_plus == 0 and w_minus == 0:
                continue

            # 保存旧系数
            theta_old = theta[j]

            # 计算z_j = (1/n) X_j^T (y - X_{-j} theta_{-j})
            # 等价于 (X_j^T residuals / n) + theta_old * XTX_diag[j]
            z_j = np.dot(X[:, j], residuals) / n_float + theta_old * XTX_diag[j]

            # 非对称软阈值
            theta_new = asymmetric_soft_threshold(z_j, w_plus, w_minus, lambda_)

            # 更新系数
            theta[j] = theta_new

            # 更新残差
            if theta_new != theta_old:
                delta = theta_old - theta_new
                residuals += delta * X[:, j]
                max_change = max(max_change, abs(delta))

        # 检查收敛
        if max_change < tol:
            if verbose:
                print(f"Converged in {iteration + 1} iterations, max change = {max_change:.6f}")
            break

    # 计算最终目标函数值
    obj = objective_value(y, X, theta, weights, lambda_, n_float)

    return theta, obj, iteration + 1


@jit(nopython=True, fastmath=True)
def cd_solve_path(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lambda_path: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-4,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    求解正则化路径，从大到小lambda，使用warm start加速
    Args:
        X: 设计矩阵 (n_samples, n_features)
        y: 响应变量 (n_samples,)
        weights: 非对称权重矩阵 (n_features, 2)
        lambda_path: 正则化强度路径，需按从大到小排序
        max_iter: 每个lambda的最大迭代次数
        tol: 收敛阈值
        verbose: 是否打印信息
    Returns:
        thetas: 所有lambda对应的系数矩阵 (n_lambda, n_features)
        objs: 所有lambda对应的目标函数值 (n_lambda,)
        n_iters: 所有lambda对应的迭代次数 (n_lambda,)
    """
    n_lambda = len(lambda_path)
    p = X.shape[1]

    thetas = np.zeros((n_lambda, p), dtype=_DTYPE)
    objs = np.zeros(n_lambda, dtype=_DTYPE)
    n_iters = np.zeros(n_lambda, dtype=np.int32)

    # 初始化解（全零）
    theta_prev = np.zeros(p, dtype=_DTYPE)

    for i in range(n_lambda):
        lambda_ = lambda_path[i]
        if verbose:
            print(f"Solving lambda {i+1}/{n_lambda}: {lambda_:.6f}")

        # warm start：用上一个lambda的解作为初始值
        theta, obj, n_iter = cd_solve(
            X, y, weights, lambda_,
            theta_init=theta_prev,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose
        )

        thetas[i] = theta
        objs[i] = obj
        n_iters[i] = n_iter
        theta_prev = theta.copy()

    return thetas, objs, n_iters
