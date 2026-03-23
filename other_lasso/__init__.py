"""
对标Lasso算法集合
包含三种常用的Lasso变种，作为XLasso的对比基准算法
"""
import sys
from pathlib import Path

from .adaptive_lasso import AdaptiveLasso, AdaptiveLassoCV
from .fused_lasso import FusedLasso, FusedLassoCV
from .group_lasso import GroupLasso, GroupLassoCV
from .adaptive_sparse_group_lasso import AdaptiveSparseGroupLasso, AdaptiveSparseGroupLassoCV

# skglm 高性能Lasso算法 (Numba JIT优化)
# 使用方式:
#   import sys
#   sys.path.insert(0, 'path/to/other_lasso/skglm_benchmark')
#   from skglm import Lasso, GroupLasso
# 或者直接: from other_lasso.skglm_benchmark.skglm import Lasso
# 注意: skglm 需要 scikit-learn >= 1.3

__all__ = [
    # 本地实现
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
