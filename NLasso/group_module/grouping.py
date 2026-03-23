"""
组处理模块：相关性分组
基于层次聚类将高相关变量归入同一组
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.utils.extmath import safe_sparse_dot
from ..base import _DTYPE


def group_variables(
    X: np.ndarray,
    threshold: float = 0.7,
    min_group_size: int = 2,
    max_group_size: int = 10,
    verbose: bool = False
) -> tuple[list[list[int]], np.ndarray]:
    """
    基于相关性层次聚类分组（对应paper 3.4.3节）
    Args:
        X: 特征矩阵 (n_samples, n_features)
        threshold: 相关性阈值，组内任意两变量相关不低于该值
        min_group_size: 最小组大小
        max_group_size: 最大组大小
        verbose: 是否打印分组信息
    Returns:
        groups: 分组列表，每个元素是特征索引列表
        R: 相关系数矩阵 (p, p)
    """
    n, p = X.shape

    # 计算Pearson相关系数矩阵
    if hasattr(X, 'toarray'):
        # 稀疏矩阵处理
        X_centered = X - X.mean(axis=0)
        cov = safe_sparse_dot(X_centered.T, X_centered) / (n - 1)
        std = np.sqrt(np.diag(cov.toarray()))
        R = cov / (std[:, None] * std[None, :] + 1e-10)
        R = R.toarray()
    else:
        # 稠密矩阵处理
        X_centered = X - np.mean(X, axis=0)
        cov = np.dot(X_centered.T, X_centered) / (n - 1)
        std = np.sqrt(np.diag(cov))
        R = cov / (std[:, None] * std[None, :] + 1e-10)

    # 构造距离矩阵：d_ij = 1 - |R_ij|
    dist = 1 - np.abs(R)
    # 转为 condensed距离矩阵（上三角）
    dist_condensed = squareform(dist, checks=False)

    # 层次聚类：平均联结
    Z = linkage(dist_condensed, method='average', metric='precomputed')

    # 确定聚类截断阈值：距离小于1-threshold的聚为一类
    t = 1 - threshold
    clusters = fcluster(Z, t=t, criterion='distance')

    # 整理分组
    groups = []
    cluster_ids, counts = np.unique(clusters, return_counts=True)

    for cid, cnt in zip(cluster_ids, counts):
        if min_group_size <= cnt <= max_group_size:
            # 提取组内特征索引
            group_indices = np.where(clusters == cid)[0].tolist()
            # 验证组内最小相关性是否满足阈值
            group_R = R[np.ix_(group_indices, group_indices)]
            min_corr = np.min(np.abs(group_R[np.triu_indices_from(group_R, k=1)]))
            if min_corr >= threshold:
                groups.append(group_indices)

    # 未分组的变量单独成组？不，paper中是仅对高相关组进行分解，其余保持原样
    # 收集未分组的特征索引
    grouped_indices = set()
    for g in groups:
        grouped_indices.update(g)
    ungrouped = [i for i in range(p) if i not in grouped_indices]
    # 单独成组（大小为1，不进行分解）
    for i in ungrouped:
        groups.append([i])

    if verbose:
        print(f"[Grouping] 共得到{len(groups)}组："
              f"{len([g for g in groups if len(g)>=2])}个高相关组，"
              f"{len([g for g in groups if len(g)==1])}个独立特征")

    return groups, R
