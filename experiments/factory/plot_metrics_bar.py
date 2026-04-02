#!/usr/bin/env python3
"""
Simulation Metrics Bar Chart Plotting Script
==============================================
Generate publication-quality bar charts from simulation raw.csv results.

Usage:
    python factory/plot_metrics_bar.py --exp 1 --metric f1
    python factory/plot_metrics_bar.py --exp 1 --metric f1 --group-by sigma
    python factory/plot_metrics_bar.py --exp 1 --metric all --output ./plots/
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add XLasso root to path
xlasso_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, xlasso_root)
sys.path.insert(0, os.path.join(xlasso_root, 'experiments'))

# Import unified colors and helpers from viz module
# This also applies plt.style.use('viz/themes/paper.mplstyle') globally
from viz import (
    MODEL_COLORS,
    get_model_display_name,
    get_model_color,
    get_all_model_colors,
    DUAL_METRIC_LABEL_FONTIZE,
    DUAL_METRIC_YTICK_FONTSIZE,
    DUAL_METRIC_XTICK_FONTSIZE,
    DUAL_METRIC_VALUE_FONTSIZE,
    DUAL_METRIC_F1_COLOR,
    DUAL_METRIC_MSE_COLOR,
    DUAL_METRIC_MSE_VALUE_COLOR,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot bar charts from simulation raw.csv results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--exp', type=int, default=1,
                        help='Experiment number (1-7, default: 1)')
    parser.add_argument('--metric', type=str, default='f1',
                        help='Metric to plot: f1, tpr, fdr, precision, recall, '
                             'sparsity, n_selected, mse, r2, all (default: f1)')
    parser.add_argument('--metric2', type=str, default=None,
                        help='Second metric to plot downward on twin axis (e.g., mse)')
    parser.add_argument('--group-by', type=str, default='model',
                        help='Group by variable for simple bar chart: model, sigma, snr')
    parser.add_argument('--x-axis', type=str, default=None,
                        choices=['sigma', 'snr', 'model'],
                        help='X-axis variable for grouped bar chart (use with --hue)')
    parser.add_argument('--hue', type=str, default=None,
                        choices=['model', 'sigma', 'snr'],
                        help='Grouping variable for bars within each x-axis tick (e.g., --x-axis snr --hue model)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: auto-generated)')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show plot interactively (default: False)')
    parser.add_argument('--style', type=str, default='publication',
                        choices=['publication', 'simple'],
                        help='Plot style (default: publication)')
    parser.add_argument('--err-bar', action='store_true', default=True,
                        help='Show error bars (default: True)')
    parser.add_argument('--no-err-bar', action='store_false', dest='err_bar',
                        help='Hide error bars')
    parser.add_argument('--orientation', type=str, default='vertical',
                        choices=['vertical', 'horizontal'],
                        help='Bar orientation (default: vertical)')
    return parser.parse_args()


def load_raw_data(exp_num):
    """Load raw.csv for specified experiment."""
    data_dir = os.path.join(
        xlasso_root, 'experiments', 'results', 'output_all', f'exp{exp_num}_all'
    )
    raw_path = os.path.join(data_dir, 'raw.csv')

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"raw.csv not found at {raw_path}")

    df = pd.read_csv(raw_path)
    return df


def aggregate_metrics(df, group_by, metric):
    """Aggregate metric by group, computing mean and std."""
    agg_funcs = {metric: ['mean', 'std', 'count']}
    grouped = df.groupby(group_by)[metric].agg(['mean', 'std'])
    grouped = grouped.dropna()
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
    return grouped.reset_index()


def get_metric_display_name(metric):
    """Get display name for metric."""
    names = {
        'f1': 'F1',
        'tpr': 'TPR',
        'fdr': 'FDR',
        'precision': 'Precision',
        'recall': 'Recall',
        'sparsity': 'Sparsity',
        'n_selected': '# Selected',
        'mse': 'MSE',
        'r2': 'R2',
        'accuracy': 'Accuracy',
    }
    return names.get(metric, metric)


def setup_publication_style():
    """Apply publication style (now handled globally by viz module import)."""
    # Style is already applied when `from viz import ...` is called
    # (viz/__init__.py runs plt.style.use('viz/themes/paper.mplstyle'))
    pass
    """Set up publication-quality plot style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.figsize': (10, 7),
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def plot_bar_chart_pub(
    labels,
    values,
    errors=None,
    metric_name=None,
    title=None,
    orientation='vertical',
    color=None,
    save_path=None,
    show=True,
):
    """Publication-quality bar chart with error bars."""
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color palette - use distinct colors for different groups
    if color is None:
        # Professional color sequence
        # Muted professional color palette
        base_colors = ['#4A5568', '#718096', '#A0AEC0', '#E53E3E', '#C05621',
                       '#2F855A', '#2B6CB0', '#805AD5', '#D69E2E', '#319795']
        color = base_colors[:len(labels)]

    x_pos = np.arange(len(labels))

    if orientation == 'vertical':
        bars = ax.bar(x_pos, values, yerr=errors if errors is not None else None,
                      color=color, capsize=4, error_kw={'linewidth': 1.5},
                      alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(metric_name if metric_name else 'Score', fontsize=14)
        ax.set_xlim(-0.6, len(labels) - 0.4)

        # Value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(values) * 0.02 if errors is None else max(values) * 0.05),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        bars = ax.barh(x_pos, values, xerr=errors if errors is not None else None,
                      color=color, capsize=4, error_kw={'linewidth': 1.5},
                      alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel(metric_name if metric_name else 'Score', fontsize=14)
        ax.set_ylim(-0.6, len(labels) - 0.4)

        # Value labels at end of bars
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(width + (max(values) * 0.01 if errors is None else max(values) * 0.03),
                    bar.get_y() + bar.get_height()/2.,
                    f'{val:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    ax.grid(True, alpha=0.3, axis='both' if orientation == 'vertical' else 'x',
            linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_grouped_bar_chart(
    df,
    metric,
    x_axis,
    hue,
    metric_name=None,
    title=None,
    save_path=None,
    show=True,
    metric2=None,
    metric2_name=None,
    show_legend=True,
):
    """Plot grouped bar chart: x_axis on X-axis, hue as grouped bars within each tick.

    When metric2 is provided (e.g. mse), both metrics are plotted on a shared x-axis
    with y=0 in the middle. metric bars grow UP from 0, metric2 bars grow DOWN from 0.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data.
    metric : str
        Metric for upward bars (e.g., 'f1').
    x_axis : str
        Variable on X-axis (e.g., 'snr').
    hue : str
        Grouping variable for bars (e.g., 'model').
    metric2 : str, optional
        Second metric plotted below the x-axis (e.g., 'mse').
    metric2_name : str, optional
        Y-axis label for metric2.
    """
    setup_publication_style()

    # Aggregate metrics
    grouped = df.groupby([x_axis, hue])[metric].agg(['mean', 'std', 'count']).reset_index()
    grouped = grouped.dropna()
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])

    if metric2:
        grouped2 = df.groupby([x_axis, hue])[metric2].agg(['mean', 'std', 'count']).reset_index()
        grouped2 = grouped2.dropna()
        grouped2['sem'] = grouped2['std'] / np.sqrt(grouped2['count'])

    # Get unique x-axis values
    if x_axis == 'snr':
        x_values = sorted(grouped[x_axis].unique(), reverse=True)
    else:
        x_values = sorted(grouped[x_axis].unique())
    # Hue values: CG-Lasso first for models
    if hue == 'model':
        unique_models = grouped[hue].unique()
        sorted_models = sorted(unique_models, key=lambda m: (0 if get_model_display_name(m) == 'CG-Lasso' else 1, get_model_display_name(m)))
        hue_values = list(sorted_models)
    else:
        hue_values = sorted(grouped[hue].unique())
    n_x = len(x_values)
    n_hue = len(hue_values)

    # Color
    if hue == 'model':
        color_map = {v: get_model_color(v) for v in hue_values}
    else:
        sequential_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_hue))
        color_map = {v: sequential_colors[i] for i, v in enumerate(hue_values)}

    bar_width = 0.8 / n_hue
    group_positions = np.arange(n_x)

    fig, ax = plt.subplots(figsize=(max(10, n_x * 1.5), 7))

    if metric2:
        # ============================================================
        # Dual-metric on SHARED left axis with EQUAL visual space:
        # F1 grows UP from y=0 to y=+1 (upper half, max visual = 1)
        # MSE grows DOWN from y=0 to y=-1 (lower half, max visual = 1)
        # ylim symmetric around 0, F1 and MSE each use exactly half
        # ============================================================
        # Compute relative MSE: for each SNR, divide by Lasso MSE as baseline
        if hue == 'model':
            grouped2_rel = grouped2.copy()
            for x_val in x_values:
                # Find Lasso MSE for this SNR
                lasso_mask = (grouped2[x_axis] == x_val) & (
                    grouped2[hue].str.contains('LassoCV', na=False) &
                    ~grouped2[hue].str.contains('Adaptive', na=False) &
                    ~grouped2[hue].str.contains('Relaxed', na=False) &
                    ~grouped2[hue].str.contains('Uni', na=False) &
                    ~grouped2[hue].str.contains('PFL', na=False) &
                    ~grouped2[hue].str.contains('Elastic', na=False)
                )
                lasso_rows = grouped2[lasso_mask]
                if len(lasso_rows) > 0:
                    lasso_mse = lasso_rows['mean'].values[0]
                    if lasso_mse > 0:
                        # Divide all MSE in this SNR group by Lasso MSE
                        mask = grouped2[x_axis] == x_val
                        grouped2_rel.loc[mask, 'mean'] = grouped2.loc[mask, 'mean'] / lasso_mse
                        grouped2_rel.loc[mask, 'sem'] = grouped2.loc[mask, 'sem'] / lasso_mse
        else:
            grouped2_rel = grouped2.copy()

        max_f1 = max(grouped['mean'].max(), 0.05)
        max_mse = max(grouped2_rel['mean'].max(), 0.05)

        # Symmetric shared range: ylim [-1, +f1_max] where f1_max = max_f1 (not 1)
        # This ensures MSE has its full half and F1 has its full half
        ax.set_ylim(-1.05, max_f1 + 0.05)
        ax.axhline(y=0, color='black', linewidth=1.0)

        # Store MSE as negative values (so bars grow DOWN from y=0)
        # But scale them so that max MSE reaches exactly -1 on y-axis
        mse_scale = 1.0 / max_mse  # so that max MSE -> -1.0

        # Normalize MSE sem too
        grouped2_norm = grouped2_rel.copy()
        grouped2_norm['mean'] = grouped2_rel['mean'] * mse_scale
        grouped2_norm['sem'] = grouped2_rel['sem'] * mse_scale

        # Single left axis with custom formatter:
        # Above y=0: show F1 values (positive)
        # Below y=0: show relative MSE values (stored negative, un-negated for display)
        from matplotlib.ticker import FuncFormatter
        def divergent_labels(y, pos):
            if y >= 0:
                # F1: show only within F1 data range
                return f'{y:.2f}' if y <= max_f1 * 1.08 else ''
            else:
                # Relative MSE: stored as negative, un-negate for display
                rel_mse_val = -y / mse_scale  # un-scale to get relative MSE
                return f'{rel_mse_val:.2f}' if -y <= 1.08 else ''
        ax.yaxis.set_major_formatter(FuncFormatter(divergent_labels))
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=12)

        # Color y-axis tick labels: red for y < 0 (MSE side), black for y >= 0 (F1 side)
        # Use tick positions (data coordinates) to determine color
        yticks = ax.get_yticks()  # data coordinates of ticks
        yticklabels = ax.get_yticklabels()
        for label, y_tick in zip(yticklabels, yticks):
            if y_tick < 0:
                label.set_color(DUAL_METRIC_MSE_COLOR)
            else:
                label.set_color(DUAL_METRIC_F1_COLOR)

        # F1 in upper half (y>0), metric2 in lower half (y<0), both left of y-axis
        ax.text(-0.05, 0.72, 'F1', fontsize=DUAL_METRIC_LABEL_FONTIZE, va='center', ha='center',
                rotation=90, transform=ax.transAxes, clip_on=False)
        lower_label = f'Relative {get_metric_display_name(metric2)} (vs. Lasso)'
        ax.text(-0.05, 0.28, lower_label, fontsize=DUAL_METRIC_LABEL_FONTIZE, va='center', ha='center',
                color=DUAL_METRIC_MSE_COLOR, rotation=90, transform=ax.transAxes, clip_on=False)

        # Draw red line for lower half of y-axis (y < 0) at x = -0.5
        ylim = ax.get_ylim()
        ax.plot([-0.5, -0.5], [ylim[0], 0], color='red', linewidth=1.5, zorder=10)

        for i, h in enumerate(hue_values):
            # ---- F1 bars: positive heights, grow UP from y=0 ----
            h_data = grouped[grouped[hue] == h]
            positions = [x_values.index(h_data[h_data[x_axis] == x][x_axis].values[0])
                         if x in h_data[x_axis].values else None
                         for x in x_values]
            means = []
            sems = []
            valid_positions = []
            for j, (pos, x_val) in enumerate(zip(positions, x_values)):
                h_row = h_data[h_data[x_axis] == x_val]
                if len(h_row) > 0:
                    means.append(h_row['mean'].values[0])
                    sems.append(h_row['sem'].values[0])
                    valid_positions.append(pos + (i - n_hue/2 + 0.5) * bar_width)
            ax.bar(valid_positions, means, width=bar_width * 0.9,
                   yerr=sems if len(sems) > 0 and not np.any(np.isnan(sems)) else None,
                   color=[color_map[h]] * len(means),
                   alpha=0.85, edgecolor='black', linewidth=0.5,
                   capsize=3)
            # F1 value labels ABOVE bars
            for bar, val in zip(ax.patches[-len(means):], means):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(means) * 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=DUAL_METRIC_VALUE_FONTSIZE, fontweight='bold')

            # ---- MSE bars: stored as NEGATIVE so they appear BELOW y=0 ----
            h2_data = grouped2_norm[grouped2_norm[hue] == h]
            positions2 = [x_values.index(h2_data[h2_data[x_axis] == x][x_axis].values[0])
                         if x in h2_data[x_axis].values else None
                         for x in x_values]
            means2 = []
            sems2 = []
            valid_positions2 = []
            for j, (pos, x_val) in enumerate(zip(positions2, x_values)):
                h2_row = h2_data[h2_data[x_axis] == x_val]
                if len(h2_row) > 0:
                    means2.append(h2_row['mean'].values[0])
                    sems2.append(h2_row['sem'].values[0])
                    valid_positions2.append(pos + (i - n_hue/2 + 0.5) * bar_width)
            ax.bar(valid_positions2, [-v for v in means2], width=bar_width * 0.9,
                   yerr=sems2 if len(sems2) > 0 and not np.any(np.isnan(sems2)) else None,
                   color=[color_map[h]] * len(means2),
                   alpha=0.55, edgecolor='black', linewidth=0.5,
                   capsize=3)
            # MSE value labels BELOW x-axis (below the bar top, which is negative)
            mse_bars = ax.patches[-len(means2):]
            # Get relative MSE values for label display
            h2_data_orig = grouped2_rel[grouped2_rel[hue] == h]
            orig_means2 = []
            for x_val in x_values:
                h2_row = h2_data_orig[h2_data_orig[x_axis] == x_val]
                if len(h2_row) > 0:
                    orig_means2.append(h2_row['mean'].values[0])
            for bar, val in zip(mse_bars, orig_means2):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() - max(means2) * 0.01,
                        f'{val:.2f}', ha='center', va='top', fontsize=DUAL_METRIC_VALUE_FONTSIZE,
                        color=DUAL_METRIC_MSE_VALUE_COLOR)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map[hue_values[0]], alpha=0.85,
                  edgecolor='black', label=get_hue_display_name(hue_values[0], hue))]
        for h in hue_values[1:]:
            legend_elements.append(
                Patch(facecolor=color_map[h], alpha=0.85, edgecolor='black',
                      label=get_hue_display_name(h, hue)))
        if show_legend:
            ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    else:
        # ============================================================
        # Single metric: standard grouped bar chart
        # ============================================================
        ax.set_ylabel(metric_name if metric_name else 'Score', fontsize=14)
        for i, h in enumerate(hue_values):
            h_data = grouped[grouped[hue] == h]
            positions = [x_values.index(h_data[h_data[x_axis] == x][x_axis].values[0])
                         if x in h_data[x_axis].values else None
                         for x in x_values]
            means = []
            sems = []
            valid_positions = []
            for j, (pos, x_val) in enumerate(zip(positions, x_values)):
                h_row = h_data[h_data[x_axis] == x_val]
                if len(h_row) > 0:
                    means.append(h_row['mean'].values[0])
                    sems.append(h_row['sem'].values[0])
                    valid_positions.append(pos + (i - n_hue/2 + 0.5) * bar_width)
            bars = ax.bar(valid_positions, means, width=bar_width * 0.9,
                          yerr=sems if len(sems) > 0 and not np.any(np.isnan(sems)) else None,
                          color=[color_map[h]] * len(means),
                          label=get_hue_display_name(h, hue),
                          capsize=3, alpha=0.85, edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(means) * 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
        if show_legend:
            ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # X-axis
    ax.set_xticks(group_positions)
    ax.set_xticklabels([get_x_tick_label(x, x_axis) for x in x_values], fontsize=DUAL_METRIC_XTICK_FONTSIZE)
    ax.set_xlabel(get_x_axis_label(x_axis), fontsize=14)
    ax.set_xlim(-0.5, n_x - 0.5)

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, grouped


def get_hue_display_name(value, hue):
    """Get display name for hue value."""
    if hue == 'model':
        return get_model_display_name(value)
    elif hue == 'sigma':
        return f'σ={value}'
    elif hue == 'snr':
        return f'SNR={value}'
    return str(value)


def get_x_tick_label(value, x_axis):
    """Get display label for a tick value on the x-axis."""
    if x_axis == 'sigma':
        return f'σ={value}'
    elif x_axis == 'snr':
        return str(value)  # "2.0", "1.0" etc.
    return str(value)


def get_x_axis_label(x_axis):
    """Get display name for the x-axis itself (used as axis label)."""
    if x_axis == 'sigma':
        return 'σ'
    elif x_axis == 'snr':
        return 'SNR'
    elif x_axis == 'model':
        return 'Model'
    return x_axis


def plot_multi_metric_bar(df, group_by, save_dir, show):
    """Plot all metrics as a grid of bar charts."""
    metrics = ['f1', 'tpr', 'fdr', 'precision', 'recall', 'sparsity', 'mse', 'r2']

    # Filter to only models that have all metrics
    models = df['model'].unique()
    n_models = len(models)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Fixed model colors
    colors = get_all_model_colors(models)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Aggregate by group_by and metric
        grouped = df.groupby(group_by)[metric].agg(['mean', 'std']).reset_index()
        grouped = grouped.dropna()

        if grouped.empty:
            ax.set_visible(False)
            continue

        x_pos = np.arange(len(grouped))
        bars = ax.bar(x_pos, grouped['mean'], yerr=grouped['std'],
                      color=colors[:len(grouped)], capsize=3, alpha=0.85,
                      edgecolor='black', linewidth=0.5)

        if group_by == 'model':
            labels = [get_model_display_name(m) for m in grouped['model']]
        else:
            labels = [str(v) for v in grouped[group_by]]

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(get_metric_display_name(metric), fontsize=11)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_title(get_metric_display_name(metric), fontsize=12, fontweight='bold')

        # Value labels
        for bar, val in zip(bars, grouped['mean']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + grouped['mean'].max() * 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle(f'Exp1: All Metrics by {group_by.title()}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(save_dir, f'all_metrics_by_{group_by}.pdf')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  All metrics plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()

    print("=" * 60)
    print(f"Simulation Metrics Bar Chart Plotting")
    print(f"Experiment: Exp{args.exp}")
    print(f"Metric: {args.metric}")
    if args.x_axis and args.hue:
        print(f"X-axis: {args.x_axis}, Hue: {args.hue}")
    else:
        print(f"Group by: {args.group_by}")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...")
    df = load_raw_data(args.exp)
    print(f"  Loaded {len(df)} rows from raw.csv")
    print(f"  Models: {df['model'].nunique()}")

    # Filter out ElasticNet (blacklist)
    elasticnet_mask = df['model'].str.contains('ElasticNet', na=False)
    n_removed = elasticnet_mask.sum()
    df = df[~elasticnet_mask]
    print(f"  Filtered out ElasticNet: {n_removed} rows removed")
    print(f"  Remaining rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Determine output directory
    if args.output is None:
        output_dir = os.path.join(xlasso_root, 'experiments', 'results', 'plots', 'bar_charts')
    else:
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect metric2: exp3 uses accuracy, others use mse (only if metric is f1 or mse)
    if args.metric2 is None:
        if args.metric in ['f1', 'mse']:
            if args.exp == 3 or 'accuracy' in df.columns:
                args.metric2 = 'accuracy'
            else:
                args.metric2 = 'mse'
        else:
            args.metric2 = None  # No auto-detect for other metrics

    # Plot
    print("\n[2/3] Generating plot...")

    if args.x_axis and args.hue:
        # Grouped bar chart: x_axis on X-axis, hue as grouped bars
        metric_display = get_metric_display_name(args.metric)
        metric2_display = get_metric_display_name(args.metric2) if args.metric2 else None
        title = None
        if args.metric2:
            output_path = args.output if args.output else os.path.join(
                output_dir, f'{args.metric}_{args.metric2}_x{args.x_axis}_hue{args.hue}_exp{args.exp}.pdf'
            )
        else:
            output_path = args.output if args.output else os.path.join(
                output_dir, f'{args.metric}_x{args.x_axis}_hue{args.hue}_exp{args.exp}.pdf'
            )
        fig, ax, grouped = plot_grouped_bar_chart(
            df=df,
            metric=args.metric,
            x_axis=args.x_axis,
            hue=args.hue,
            metric_name=metric_display,
            title=title,
            save_path=output_path,
            show=args.show,
            metric2=args.metric2,
            metric2_name=metric2_display,
            show_legend=True,
        )
        print(f"  Plot saved to: {output_path}")

        # Print summary
        print("\n" + "=" * 60)
        print(f"{args.metric.upper()} by {args.x_axis} & {args.hue}")
        print("=" * 60)
        print(grouped.to_string(index=False))

    elif args.metric == 'all':
        # Generate all metrics grid plot
        plot_multi_metric_bar(df, args.group_by, output_dir, args.show)
        output_path = os.path.join(output_dir, f'all_metrics_by_{args.group_by}.pdf')
    else:
        # Single metric bar chart
        # Aggregate by group_by
        grouped = df.groupby(args.group_by)[args.metric].agg(['mean', 'std', 'count']).reset_index()
        grouped = grouped.dropna()

        if grouped.empty:
            print(f"  Error: No data for metric '{args.metric}'")
            return

        if args.group_by == 'model':
            labels = [get_model_display_name(m) for m in grouped['model']]
            fixed_colors = get_all_model_colors(grouped['model'])
        elif args.group_by == 'sigma':
            labels = [f'σ={s}' for s in grouped['sigma']]
            fixed_colors = None
        elif args.group_by == 'snr':
            labels = [f'SNR={s}' for s in grouped['snr']]
            fixed_colors = None
        else:
            labels = [str(v) for v in grouped[args.group_by]]
            fixed_colors = None

        values = grouped['mean'].values
        errors = grouped['std'].values if args.err_bar else None

        metric_display = get_metric_display_name(args.metric)
        title = f'Exp{args.exp}: {metric_display} by {args.group_by.title()}'

        output_path = args.output if args.output else os.path.join(
            output_dir, f'{args.metric}_by_{args.group_by}_exp{args.exp}.pdf'
        )

        fig, ax = plot_bar_chart_pub(
            labels=labels,
            values=values,
            errors=errors,
            color=fixed_colors,
            metric_name=metric_display,
            title=title,
            orientation=args.orientation,
            save_path=output_path,
            show=args.show,
        )
        print(f"  Plot saved to: {output_path}")

        # Print summary statistics
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        print(f"\n{args.metric.upper()} by {args.group_by}:")
        for i, row in grouped.iterrows():
            if args.group_by == 'model':
                label = get_model_display_name(row['model'])
            elif args.group_by == 'sigma':
                label = f"σ={row['sigma']}"
            elif args.group_by == 'snr':
                label = f"SNR={row['snr']}"
            else:
                label = str(row[args.group_by])
            print(f"  {label}: {row['mean']:.4f} ± {row['std']:.4f}")

    print("\n[3/3] Done!")


if __name__ == '__main__':
    main()