"""
对标Lasso算法集合
包含三种常用的Lasso变种，作为XLasso的对比基准算法
"""
from .adaptive_lasso import AdaptiveLasso, AdaptiveLassoCV
from .fused_lasso import FusedLasso, FusedLassoCV
from .group_lasso import GroupLasso, GroupLassoCV
from .adaptive_sparse_group_lasso import AdaptiveSparseGroupLasso, AdaptiveSparseGroupLassoCV

__all__ = [
    'AdaptiveLasso',
    'AdaptiveLassoCV',
    'FusedLasso',
    'FusedLassoCV',
    'GroupLasso',
    'GroupLassoCV',
    'AdaptiveSparseGroupLasso',
    'AdaptiveSparseGroupLassoCV',
]

__version__ = '1.0.0'
