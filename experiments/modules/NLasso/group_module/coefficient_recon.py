"""
组处理模块：系数还原与组感知截断
对应paper 3.4.3节
"""
import numpy as np
from ..base import _DTYPE


def reconstruct_coefficients(
    theta_transformed: np.ndarray,
    groups: list[list[int]],
    decomposers: list,
    p_original: int,
    group_truncation_threshold: float = 0.5,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    将变换空间的系数还原为原始特征空间系数（正交补投影修复版，唯一官方实现）
    理论设计：引入正交补投影算子修复L1正则化对几何约束的破坏
    Args:
        theta_transformed: 变换空间的系数向量 (p_transformed,)
        groups: 分组列表，每个元素是原始特征索引列表
        decomposers: 每个组对应的OrthogonalDecomposer实例
        p_original: 原始特征数量
        group_truncation_threshold: 组截断阈值，组内非零系数比例低于该值则整组清零
        epsilon: 非零系数判定阈值
    Returns:
        beta_original: 原始特征空间系数 (p_original,)
    """
    beta_original = np.zeros(p_original, dtype=_DTYPE)
    pos = 0  # 当前在theta_transformed中的位置

    for group_idx, (group, decomposer) in enumerate(zip(groups, decomposers)):
        k = len(group)  # 组大小
        if k == 1:
            # 单变量组：直接取对应系数
            theta_g = theta_transformed[pos:pos+1]
            beta_g = decomposer.inverse_transform(theta_g)
            pos += 1
        else:
            # 多变量组：取m + k个系数
            m = decomposer.m_  # 共同趋势分量数
            theta_g = theta_transformed[pos:pos + m + k]
            pos += m + k

            # 拆分共同趋势系数和细节系数
            theta_c = theta_g[:m]  # 共同趋势分量系数
            theta_d = theta_g[m:]  # 细节分量系数

            # 正交补投影修复L1正则化的几何约束破坏
            W_c = decomposer.W_c_
            k = len(group)
            # 计算正交补投影算子 P_perp = I - W_c @ W_c^T
            P_perp = np.eye(k) - np.dot(W_c, W_c.T)
            # 修复公式：beta_group = W_c @ theta_c + P_perp @ theta_d
            beta_std = np.dot(W_c, theta_c) + np.dot(P_perp, theta_d)

            # 逆标准化
            beta_g = beta_std / decomposer.std_

        # 组感知截断（可选，paper 3.4.3节）
        if group_truncation_threshold > 0 and k > 1:
            # 计算组内非零系数比例
            non_zero_ratio = np.sum(np.abs(beta_g) > epsilon) / k
            if non_zero_ratio < group_truncation_threshold:
                # 整组清零
                beta_g[:] = 0.0

        # 赋值到原始特征位置
        beta_original[group] = beta_g

    return beta_original
