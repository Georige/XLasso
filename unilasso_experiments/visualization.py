"""
Visualization utilities for experiment results.
Shared plotting functions used across all experiment types.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from matplotlib import rcParams


# Configure plotting style
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8
rcParams['figure.titlesize'] = 14


def plot_method_comparison_boxplot(
    raw_results: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    ascending: bool = False,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Create boxplot comparing metric distribution across methods.

    Parameters
    ----------
    raw_results : pd.DataFrame
        Raw results DataFrame from experiment (one row per repetition per method).
    metric : str
        Which metric column to plot.
    title : str, optional
        Plot title.
    ylabel : str, optional
        Y-axis label.
    ascending : bool
        Whether lower values are better (for ordering).
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Order methods by mean performance
    mean_order = raw_results.groupby('method')[metric].mean()
    if ascending:
        mean_order = mean_order.sort_values(ascending=True)
    else:
        mean_order = mean_order.sort_values(ascending=False)

    sns.boxplot(data=raw_results, x='method', y=metric, order=mean_order.index, ax=ax)

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(metric)
    ax.set_xlabel('Method')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def plot_performance_heatmap(
    results_df: pd.DataFrame,
    metric: str,
    x_labels: List[str],
    y_labels: List[str],
    title: Optional[str] = None,
    cmap: str = 'viridis',
    figsize: Tuple[float, float] = (8, 6)
) -> plt.Figure:
    """
    Create heatmap of performance across two dimensions.

    Parameters
    ----------
    results_df : pd.DataFrame
        Pivot table with metric values.
    metric : str
        Metric name.
    x_labels : list
        Labels for x-axis.
    y_labels : list
        Labels for y-axis.
    title : str, optional
        Plot title.
    cmap : str
        Colormap name.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(results_df, annot=True, fmt='.3f', cmap=cmap, ax=ax)

    if title:
        ax.set_title(title)
    ax.set_xlabel(x_labels)
    ax.set_ylabel(y_labels)
    plt.tight_layout()

    return fig


def plot_tpr_fpr_comparison(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = 'TPR-FPR Curve',
    figsize: Tuple[float, float] = (8, 6)
) -> plt.Figure:
    """
    Plot TPR-FPR curve similar to ROC for variable selection across methods.

    Parameters
    ----------
    results_dict : Dict[str, Tuple[np.ndarray, np.ndarray]]
        {method_name: (fpr_array, tpr_array)}
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for method_name, (fpr, tpr) in results_dict.items():
        # Sort by fpr
        order = np.argsort(fpr)
        fpr_sorted = fpr[order]
        tpr_sorted = tpr[order]
        ax.plot(fpr_sorted, tpr_sorted, '-o', markersize=4, label=method_name)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig


def plot_nonlinear_comparison(
    X: np.ndarray,
    y_true: np.ndarray,
    y_fitted: np.ndarray,
    feature_idx: int,
    true_func,
    title: str = 'Nonlinear Function Fit',
    figsize: Tuple[float, float] = (10, 4)
) -> plt.Figure:
    """
    Plot comparison of fitted nonlinear function vs true function.

    Parameters
    ----------
    X : np.ndarray
        Sorted x values for plotting.
    y_true : np.ndarray
        True function values.
    y_fitted : np.ndarray
        Fitted function values.
    feature_idx : int
        Feature index (for title).
    true_func : callable
        True function.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x_sort = np.argsort(X[:, feature_idx])
    x_vals = X[x_sort, feature_idx]

    ax.plot(x_vals, y_true[x_sort], 'r-', linewidth=2, label='True')
    ax.plot(x_vals, y_fitted[x_sort], 'b--', linewidth=2, label='Fitted')
    ax.scatter(X[:, feature_idx], y_true, alpha=0.5, s=10)

    ax.set_xlabel(f'x_{feature_idx}')
    ax.set_ylabel('f(x)')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig


def plot_summary_bar_chart(
    mean_results: pd.Series,
    std_results: pd.Series,
    title: str,
    ylabel: str,
    figsize: Tuple[float, float] = (10, 5),
    color_palette: Optional[List] = None
) -> plt.Figure:
    """
    Plot summary bar chart with error bars comparing methods.

    Parameters
    ----------
    mean_results : pd.Series
        Mean values indexed by method.
    std_results : pd.Series
        Standard deviation across repetitions.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    figsize : tuple
        Figure size.
    color_palette : list, optional
        Custom colors for bars.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(mean_results))
    width = 0.6

    if color_palette is None:
        colors = sns.color_palette('Set2', len(mean_results))
    else:
        colors = color_palette

    bars = ax.bar(x, mean_results.values, width,
                  yerr=std_results.values,
                  capsize=5,
                  color=colors)

    ax.set_xticks(x)
    ax.set_xticklabels(mean_results.index, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    return fig


def save_figure(fig: plt.Figure, filepath: str, save_pdf: bool = True):
    """
    Save figure to PNG and optionally PDF.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    filepath : str
        Base filepath (without extension).
    save_pdf : bool
        Whether to also save as PDF.
    """
    fig.savefig(f'{filepath}.png', bbox_inches='tight', dpi=300)
    if save_pdf:
        fig.savefig(f'{filepath}.pdf', bbox_inches='tight')
    plt.close(fig)


def results_to_latex_table(
    aggregated_results: pd.DataFrame,
    metric_order: List[str],
    method_order: List[str],
    caption: str,
    label: str
) -> str:
    """
    Convert aggregated results to LaTeX table format.

    Parameters
    ----------
    aggregated_results : pd.DataFrame
        Aggregated results from experiment.aggregate_results().
    metric_order : List[str]
        Which metrics to include, in order.
    method_order : List[str]
        Which methods to include, in order.
    caption : str
        Table caption.
    label : str
        LaTeX label.

    Returns
    -------
    latex : str
        LaTeX table code.
    """
    latex = []
    latex.append(r'\begin{table}[ht]')
    latex.append(r'\centering')
    latex.append(r'\caption{' + caption + r'}')
    latex.append(r'\label{' + label + r'}')
    latex.append(r'\begin{tabular}{l' + 'c' * len(metric_order) + r'}')
    latex.append(r'\hline')
    latex.append(r'\textbf{Method} & ' + ' & '.join([r'\textbf{' + m + r'}' for m in metric_order]) + r' \\')
    latex.append(r'\hline')

    for method in method_order:
        row = [method]
        for metric in metric_order:
            mean = aggregated_results.loc[method, (metric, 'mean')]
            std = aggregated_results.loc[method, (metric, 'std')]
            row.append(f'{mean:.3f} ± {std:.3f}')
        latex.append(' & '.join(row) + r' \\')

    latex.append(r'\hline')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')

    return '\n'.join(latex)
