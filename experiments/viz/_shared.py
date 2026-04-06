"""
Shared Constants and Helpers for Visualization
=============================================
Shared across all viz submodules to ensure consistent colors, fonts, and naming.
Do not import from .curves, .features, .comparison, .diagnostics here.
"""

import os
import matplotlib.pyplot as plt

# =============================================================================
# Unified Model Color Palette
# Each algorithm model gets a FIXED color across all plots.
# BAFL (PFLRegressorCV) is the primary algorithm, highlighted in orange-red.
# =============================================================================
MODEL_COLORS = {
    'CG-Lasso': '#e76f51',      # Deep orange-red
    'AdaptiveLasso': '#4a7fb8',  # Deep blue
    'ElasticNet': '#d45d8a',    # Deep pink
    'Lasso': '#2db87a',         # Deep teal
    'RelaxedLasso': '#6ab028',  # Deep green
    'Unilasso': '#d4a800',      # Deep yellow
}


# =============================================================================
# Dual-Metric (Diverging) Chart Style
# F1 grows UP from x-axis, Relative MSE grows DOWN from x-axis
# F1: upper half (y >= 0), MSE: lower half (y < 0)
# =============================================================================
# Font sizes (统一字体大小)
DUAL_METRIC_LABEL_FONTIZE = 14   # F1 / Relative MSE 标签
DUAL_METRIC_YTICK_FONTSIZE = 12  # Y 轴刻度数字
DUAL_METRIC_XTICK_FONTSIZE = 14  # X 轴刻度数字
DUAL_METRIC_VALUE_FONTSIZE = 10  # 柱状图数值标签
DUAL_METRIC_AXIS_LABEL_FONTSIZE = 14  # 坐标轴标签
DUAL_METRIC_LEGEND_FONTSIZE = 10  # 图例

# Colors
DUAL_METRIC_F1_COLOR = 'black'      # F1 上半部分
DUAL_METRIC_MSE_COLOR = 'red'       # MSE 下半部分
DUAL_METRIC_MSE_VALUE_COLOR = 'dimgray'  # MSE 数值标签颜色


def get_model_display_name(model):
    """Shorten model name for display."""
    if 'BAFL' in model or 'PFLRegressorCV' in model:
        return 'CG-Lasso'
    elif 'AdaptiveLassoCV' in model:
        return 'AdaptiveLasso'
    elif 'ElasticNetCV' in model:
        return 'ElasticNet'
    elif 'RelaxedLassoCV' in model:
        return 'RelaxedLasso'
    elif 'LassoCV' in model:
        return 'Lasso'
    elif 'UnilassoCV' in model:
        return 'Unilasso'
    return model


def get_model_color(model_name):
    """Get fixed color for a model. Returns gray fallback if unknown."""
    short_name = get_model_display_name(model_name)
    return MODEL_COLORS.get(short_name, '#A0AEC0')


def get_all_model_colors(model_list):
    """Get list of fixed colors for a list of model names."""
    return [get_model_color(m) for m in model_list]


def apply_paper_style():
    """Apply paper style settings globally."""
    _viz_dir = os.path.dirname(os.path.abspath(__file__))
    plt.style.use(os.path.join(_viz_dir, 'themes', 'paper.mplstyle'))
