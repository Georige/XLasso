"""
第一阶段：自适应权重计算
基于强Ridge系数幅度构造非对称惩罚权重
"""
import numpy as np
from ..base import _DTYPE


def calculate_asymmetric_weights(
    beta_ridge: np.ndarray,
    gamma: float = 0.3,
    s: float = 1.0,
    eps: float = 1e-10
) -> np.ndarray:
    """
    计算非对称惩罚权重（对应paper 3.2-3.3节）
    Args:
        beta_ridge: 强Ridge回归系数 (p,)
        gamma: 指数映射陡峭程度（paper推荐默认0.3）
        s: 全局惩罚缩放因子
        eps: 防止除零的小常量
    Returns:
        weights: 权重矩阵 (p, 2) -> [w_plus, w_minus] per feature
    """
    p = len(beta_ridge)
    beta_abs = np.abs(beta_ridge)

    # 计算最大幅度M
    M = np.max(beta_abs)
    if M < eps:
        # 所有系数都接近0时，退化为对称Lasso（w_plus = w_minus = 0.5*s）
        w = np.full(p, 0.5, dtype=_DTYPE)
    else:
        # 指数映射：w_j = 0.5 * exp(-γ * |β_j| / M)
        scaled_magnitude = beta_abs / M
        w = 0.5 * np.exp(-gamma * scaled_magnitude)

    # 非对称权重：w_plus = s*w_j, w_minus = s*(1 - w_j)
    w_plus = s * w
    w_minus = s * (1 - w)

    # 合并为(p, 2)矩阵
    weights = np.column_stack([w_plus, w_minus]).astype(_DTYPE)

    return weights
