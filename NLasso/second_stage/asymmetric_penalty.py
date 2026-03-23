"""
第二阶段：非对称惩罚项定义
对应paper 3.3节
"""
import numpy as np
from numba import jit
from ..base import _DTYPE


@jit(nopython=True, fastmath=True)
def asymmetric_soft_threshold(z: float, w_plus: float, w_minus: float, lambda_: float) -> float:
    """
    非对称软阈值算子（paper 3.5.1节公式）
    Args:
        z: 残差相关性 (1/n) X_j^T (y - X_{-j} θ_{-j})
        w_plus: 正侧惩罚权重
        w_minus: 负侧惩罚权重
        lambda_: 正则化强度
    Returns:
        theta_j: 阈值后的系数
    """
    lambda_plus = lambda_ * w_plus
    lambda_minus = lambda_ * w_minus

    if z > lambda_plus:
        return z - lambda_plus
    elif z < -lambda_minus:
        return z + lambda_minus
    else:
        return 0.0


@jit(nopython=True, fastmath=True)
def asymmetric_penalty_value(theta: np.ndarray, weights: np.ndarray, lambda_: float) -> float:
    """
    计算非对称惩罚项的值
    Args:
        theta: 系数向量 (p,)
        weights: 权重矩阵 (p, 2) -> [w_plus, w_minus]
        lambda_: 正则化强度
    Returns:
        penalty: 惩罚项值
    """
    p = len(theta)
    penalty = 0.0
    for j in range(p):
        w_plus, w_minus = weights[j]
        if theta[j] > 0:
            penalty += w_plus * theta[j]
        else:
            penalty += w_minus * (-theta[j])
    return lambda_ * penalty


@jit(nopython=True, fastmath=True)
def objective_value(
    y: np.ndarray,
    X: np.ndarray,
    theta: np.ndarray,
    weights: np.ndarray,
    lambda_: float,
    n: float
) -> float:
    """
    计算目标函数值：均方误差 + 非对称惩罚
    Args:
        y: 响应变量 (n,)
        X: 设计矩阵 (n, p)
        theta: 系数向量 (p,)
        weights: 权重矩阵 (p, 2)
        lambda_: 正则化强度
        n: 样本量
    Returns:
        obj: 目标函数值
    """
    residuals = y - np.dot(X, theta)
    mse = np.sum(residuals ** 2) / (2 * n)
    penalty = asymmetric_penalty_value(theta, weights, lambda_)
    return mse + penalty
