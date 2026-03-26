"""
组处理模块：高相关变量正交分解与系数还原
对应paper 3.4节
"""
from .grouping import group_variables
from .orthogonal_decomp import OrthogonalDecomposer
from .coefficient_recon import reconstruct_coefficients

__all__ = [
    'group_variables',
    'OrthogonalDecomposer',
    'reconstruct_coefficients'
]
