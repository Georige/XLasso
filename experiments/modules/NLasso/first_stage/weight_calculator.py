"""
第一阶段：自适应权重计算
基于强Ridge系数幅度构造非对称惩罚权重
"""
import numpy as np
from ..base import _DTYPE


def calculate_asymmetric_weights(
    beta_ridge: np.ndarray,
    gamma: float = 1.0,
    s: float = 1.0,
    eps: float = 1e-5
) -> np.ndarray:
    """
    计算非对称惩罚权重（对应paper 3.2-3.3节）

    权重公式（均值归一化幂律）:
    步骤1: w_j^raw = 1 / (|β_ridge_j| + ε)^γ
    步骤2: w_j^norm = w_j^raw / (1/p * Σ_k w_k^raw)

    解释：
    - β=0 (弱信号): w_raw 很大 → w_norm 大 → w_plus 大, w_minus 小
      → 正侧惩罚重，负侧惩罚轻 → 容易被压零（因为 w_minus 小意味着负侧惩罚轻，
        但非对称惩罚中 w_plus 越大则正侧越难保留，实际上弱信号正负侧惩罚都重）
    - β=max (强信号): w_raw 很小 → w_norm 小 → w_plus 小, w_minus 大
      → 正侧惩罚轻，负侧惩罚重 → 保留与Ridge同向的信号

    注意：这里采用与 fit_asymmetric_adaptive_lasso 一致的公式，
    直接使用 |β_ridge_j| + ε 而非相对幅度 |β_ridge_j|/M

    Args:
        beta_ridge: 强Ridge回归系数 (p,)
        gamma: 指数衰减参数（控制权重分化程度，γ越大分化越强）
        s: 全局惩罚缩放因子
        eps: 防止除零的小常量
    Returns:
        weights: 权重矩阵 (p, 2) -> [w_plus, w_minus] per feature
    """
    p = len(beta_ridge)
    beta_abs = np.abs(beta_ridge)

    # 步骤1: 计算原始权重（逆尺度幂律）
    # w_raw = 1 / (|β| + ε)^γ
    # β=0 (弱信号): w_raw = 1/ε^γ (很大)
    # β=large (强信号): w_raw = 1/|β|^γ (很小)
    w_raw = 1.0 / (beta_abs + eps) ** gamma

    # 步骤2: 均值归一化
    # 保证所有权重均值为1，便于 alpha 调参
    w_mean = np.mean(w_raw)
    if w_mean < eps:
        # 所有系数都接近0时，退化为对称Lasso
        w = np.full(p, 0.5, dtype=_DTYPE)
    else:
        w = w_raw / w_mean
        # 限制 w 在 [0, 1] 范围内
        w = np.clip(w, 0.0, 1.0)

    # 非对称权重：w_plus = s*w, w_minus = s*(1 - w)
    # w 越大 → w_plus 越大, w_minus 越小
    # 弱信号: w 大 → w_plus 大, w_minus 小 → 正侧惩罚重，负侧惩罚轻
    # 强信号: w 小 → w_plus 小, w_minus 大 → 正侧惩罚轻，负侧惩罚重
    w_plus = s * w
    w_minus = s * (1 - w)

    # 合并为(p, 2)矩阵
    weights = np.column_stack([w_plus, w_minus]).astype(_DTYPE)

    return weights
