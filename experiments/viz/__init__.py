"""
Viz Module - Unified Visualization Layer
=======================================
Provides unified interface for all visualization functions.

Note:
    Paper style (Times New Roman font, publication-quality settings) is applied
    automatically when this module is imported. All plots use a unified color
    palette where each algorithm model has a fixed color across all figures.

Usage:
    from viz import plot_convergence, plot_metric_compare, plot_feature_importance
    from viz import MODEL_COLORS, get_model_display_name, get_model_color
"""

from ._shared import (
    MODEL_COLORS,
    get_model_display_name,
    get_model_color,
    get_all_model_colors,
    apply_paper_style,
    # Dual-metric (diverging) chart style
    DUAL_METRIC_LABEL_FONTIZE,
    DUAL_METRIC_YTICK_FONTSIZE,
    DUAL_METRIC_XTICK_FONTSIZE,
    DUAL_METRIC_VALUE_FONTSIZE,
    DUAL_METRIC_F1_COLOR,
    DUAL_METRIC_MSE_COLOR,
    DUAL_METRIC_MSE_VALUE_COLOR,
)

# Apply paper style globally when viz is imported
apply_paper_style()

# Import all visualization functions
from .curves import (
    plot_convergence,
    plot_lambda_sweep,
    plot_cv_fold_results,
    plot_lambda_path,
    plot_cv_loss_curve,
    plot_bafl_cv_error_path,
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
    plot_bafl_coefficient_path,
)

from .diagnostics import (
    plot_residuals,
    plot_qq,
    plot_loo_pred,
    plot_learning_curve,
    plot_residual_histogram,
)

from .ablation import (
    plot_ablation_heatmap,
    plot_ablation_gamma_marginal,
    plot_ablation_cap_marginal,
    plot_ablation_rank,
    plot_ablation_profile,
    plot_ablation_gamma_convergence,
    plot_all_ablation,
)

import matplotlib.pyplot as plt

__all__ = [
    # Curves
    "plot_convergence",
    "plot_lambda_sweep",
    "plot_cv_fold_results",
    "plot_lambda_path",
    "plot_cv_loss_curve",
    "plot_bafl_cv_error_path",
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
    "plot_bafl_coefficient_path",
    # Diagnostics
    "plot_residuals",
    "plot_qq",
    "plot_loo_pred",
    "plot_learning_curve",
    "plot_residual_histogram",
    # Ablation
    "plot_ablation_heatmap",
    "plot_ablation_gamma_marginal",
    "plot_ablation_cap_marginal",
    "plot_ablation_rank",
    "plot_ablation_profile",
    "plot_ablation_gamma_convergence",
    "plot_all_ablation",
    # Dual-metric style constants
    "DUAL_METRIC_LABEL_FONTIZE",
    "DUAL_METRIC_YTICK_FONTSIZE",
    "DUAL_METRIC_XTICK_FONTSIZE",
    "DUAL_METRIC_VALUE_FONTSIZE",
    "DUAL_METRIC_F1_COLOR",
    "DUAL_METRIC_MSE_COLOR",
    "DUAL_METRIC_MSE_VALUE_COLOR",
]