"""
NLasso: 一种基于UniLasso改进的稀疏回归方法
Copyright (c) 2026
"""

__version__ = "3.6.0"

from .api import NLasso, NLassoClassifier, NLassoCV, NLassoClassifierCV
from . import metrics

# AdaptiveFlippedLasso family
from .adaptive_flipped_lasso import (
    AdaptiveFlippedLasso,
    AdaptiveFlippedLassoClassifier,
    AdaptiveFlippedLassoCV,
)

__all__ = [
    # NLasso family
    'NLasso',
    'NLassoClassifier',
    'NLassoCV',
    'NLassoClassifierCV',
    # AdaptiveFlippedLasso family
    'AdaptiveFlippedLasso',
    'AdaptiveFlippedLassoClassifier',
    'AdaptiveFlippedLassoCV',
    # Metrics
    'metrics'
]
