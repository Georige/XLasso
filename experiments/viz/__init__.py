"""
Viz Module - Unified Visualization Layer
=======================================
Provides unified interface for all visualization functions.

Usage:
    from viz import plot_convergence, plot_metric_compare, plot_feature_importance
    import matplotlib.pyplot as plt
    plt.style.use('viz/themes/paper.mplstyle')
"""

# Import all visualization functions
from .curves import (
    plot_convergence,
    plot_lambda_sweep,
    plot_cv_fold_results,
    plot_lambda_path,
    plot_cv_loss_curve,
)

from .comparison import (
    plot_metric_compare,
    plot_rank_heatmap,
    plot_confusion_matrix,
    plot_radar_chart,
    plot_bar_chart,
)

from .features import (
    plot_feature_importance,
    plot_selected_features,
    plot_coefficient_path,
    plot_feature_correlation,
)

from .diagnostics import (
    plot_residuals,
    plot_qq,
    plot_loo_pred,
    plot_learning_curve,
    plot_residual_histogram,
)

import matplotlib.pyplot as plt

__all__ = [
    # Curves
    "plot_convergence",
    "plot_lambda_sweep",
    "plot_cv_fold_results",
    "plot_lambda_path",
    "plot_cv_loss_curve",
    # Comparison
    "plot_metric_compare",
    "plot_rank_heatmap",
    "plot_confusion_matrix",
    "plot_radar_chart",
    "plot_bar_chart",
    # Features
    "plot_feature_importance",
    "plot_selected_features",
    "plot_coefficient_path",
    "plot_feature_correlation",
    # Diagnostics
    "plot_residuals",
    "plot_qq",
    "plot_loo_pred",
    "plot_learning_curve",
    "plot_residual_histogram",
]