"""
AdaptiveFlippedLasso: 数据翻转 + 归一化权重 + 非负 Lasso
一种简化的自适应稀疏回归方法，保留一阶段 Ridge 系数的方向信息
"""

__version__ = "1.0.0"

from .api import AdaptiveFlippedLasso, AdaptiveFlippedLassoClassifier, AdaptiveFlippedLassoCV, AdaptiveFlippedLassoClassifierEBIC, AdaptiveFlippedLassoCV_ENClassifier
from .base import AdaptiveFlippedLassoEBIC, AdaptiveFlippedLassoCV_EN, AdaptiveFlippedLassoEBIC_Simple

__all__ = [
    'AdaptiveFlippedLasso',
    'AdaptiveFlippedLassoClassifier',
    'AdaptiveFlippedLassoCV',
    'AdaptiveFlippedLassoClassifierEBIC',
    'AdaptiveFlippedLassoEBIC',
    'AdaptiveFlippedLassoCV_EN',
    'AdaptiveFlippedLassoCV_ENClassifier',
    'AdaptiveFlippedLassoEBIC_Simple',
]
