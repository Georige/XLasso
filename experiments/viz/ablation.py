"""
Ablation Module - BAFL Parameter Ablation Visualizations
======================================================

Provides visualization functions for 2D parameter ablation studies
(gamma × cap grid search results).

Functions:
- plot_ablation_heatmap: Single metric heatmap with optional annotations
- plot_ablation_gamma_marginal: Marginal effect curve for gamma
- plot_ablation_cap_marginal: Marginal effect curve for cap
- plot_ablation_rank: Average rank heatmap
- plot_ablation_profile: Profile line plot for fixed gamma
- plot_ablation_gamma_convergence: Gamma convergence lines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Tuple


# Default colormaps for different metric types
F1_CMAP = 'RdYlGn'  # F1: higher is better (red=bad, green=good)
MSE_CMAP = 'YlOrRd'  # MSE: lower is better (yellow=good, red=bad)
FDR_CMAP = 'YlGnBu'  # FDR: lower is better
TPR_CMAP = 'YlGn'    # TPR: higher is better


def _preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe to handle nan/None in cap column.

    pandas reads 'None' from CSV as nan, this converts them back.
    """
    df = df.copy()
    df['cap'] = df['cap'].apply(lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    return df


def _prepare_pivot(df: pd.DataFrame, metric_col: str,
                  gammas: List[float], caps: List) -> Tuple[np.ndarray, np.ndarray, List, List]:
    """Prepare pivot table data for heatmap.

    Returns:
        (values, std_values, row_labels, col_labels)
    """
    # Work on a copy to avoid modifying original
    df_work = df.copy()

    # Handle nan in cap column (from pandas reading 'None' as nan)
    df_work['cap'] = df_work['cap'].apply(lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)

    # Get std column name
    std_col = f'{metric_col}_std'
    has_std = std_col in df_work.columns

    # Check if there are None/NaN caps in the original data
    has_null_caps = any((isinstance(c, float) and np.isnan(c)) for c in caps) or None in caps

    # Build cap order (None last)
    cap_order = [c for c in caps if c is not None and not (isinstance(c, float) and np.isnan(c))]
    if has_null_caps:
        cap_order.append(None)

    # Create values array
    n_rows = len(gammas)
    n_cols = len(cap_order)
    values = np.full((n_rows, n_cols), np.nan)
    std_values = np.full((n_rows, n_cols), np.nan)

    # Fill in values
    for _, row in df_work.iterrows():
        g = row['gamma']
        c = row['cap']
        # Skip if gamma not in our list
        if g not in gammas:
            continue
        # Handle None cap
        if pd.isna(c) or c is None:
            if None not in cap_order:
                continue
            c_idx = cap_order.index(None)
        else:
            if c not in cap_order:
                continue
            c_idx = cap_order.index(c)

        i = gammas.index(g)
        values[i, c_idx] = row[metric_col]
        if has_std:
            std_values[i, c_idx] = row.get(std_col, np.nan)

    row_labels = [f'{g:.1f}' for g in gammas]
    col_labels = [str(int(c)) if c is not None else 'None' for c in cap_order]

    return values, std_values, row_labels, col_labels


def plot_ablation_heatmap(
    df: pd.DataFrame,
    metric_col: str,
    gammas: List[float],
    caps: List,
    title: str,
    cmap: str = 'RdYlGn',
    annotate: bool = True,
    fmt: str = '.3f',
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (12, 8),
    highlight_best: bool = True,
    **kwargs
) -> plt.Figure:
    """Plot a single metric heatmap for gamma × cap ablation.

    Parameters
    ----------
    df : pd.DataFrame
        Summary dataframe with gamma, cap, and metric columns.
    metric_col : str
        Name of the metric column (e.g., 'f1_mean').
    gammas : List[float]
        Ordered list of gamma values.
    caps : List
        Ordered list of cap values (can include None).
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    annotate : bool
        Whether to annotate cells with values.
    fmt : str
        Format string for annotations.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the plot.
    figsize : tuple
        Figure size (width, height).
    highlight_best : bool
        Whether to highlight the best (max) cell with a star.

    Returns
    -------
    fig : plt.Figure
    """
    values, std_values, row_labels, col_labels = _prepare_pivot(df, metric_col, gammas, caps)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(values, cmap=cmap, aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    metric_label = metric_col.replace('_', ' ').title().replace('Fdr', 'FDR')
    cbar.set_label(metric_label.replace('Tpr', 'TRP').replace('Mse', 'MSE'), fontsize=14)

    # Ticks and labels
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.set_xlabel('c', fontsize=14)
    ax.set_ylabel('γ', fontsize=14)
    # Title removed

    # Find best cell - all metrics use green border
    # FDR: lower is better (argmin)
    # F1, TPR: higher is better (argmax)
    # MSE: lower is better (argmin)
    if highlight_best:
        if metric_col in ['fdr_mean', 'mse_mean']:
            best_idx = np.nanargmin(values)  # Lower is better
        else:
            best_idx = np.nanargmax(values)  # Higher is better
        best_border_color = 'green'
        best_i, best_j = np.unravel_index(best_idx, values.shape)
    else:
        best_i, best_j = None, None
        best_border_color = 'green'

    # Annotations
    if annotate:
        mean_val = np.nanmean(values)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                std = std_values[i, j] if not np.isnan(std_values[i, j]) else None

                if std is not None:
                    text = f'{val:{fmt}}\n±{std:.3f}'
                else:
                    text = f'{val:{fmt}}'

                # Black text for all, with * suffix for best cell
                if (i, j) == (best_i, best_j):
                    text = text + ' *'
                    color = 'black'
                    fontweight = 'bold'
                else:
                    color = 'black'
                    fontweight = 'normal'

                ax.text(j, i, text, ha='center', va='center', fontsize=9, color=color, fontweight=fontweight)

    # Draw box around best cell
    if highlight_best and best_i is not None:
        rect = plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                             fill=False, edgecolor=best_border_color, linewidth=3)
        ax.add_patch(rect)

    plt.tight_layout()

    if save_path:
        # Save as PDF only
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(save_path, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_ablation_gamma_marginal(
    df: pd.DataFrame,
    gammas: List[float],
    caps: List,
    metric_col: str = 'f1_mean',
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (12, 8)
) -> plt.Figure:
    """Plot marginal effect of gamma on the specified metric.

    Parameters
    ----------
    df : pd.DataFrame
        Summary dataframe.
    gammas : List[float]
        Ordered gamma values.
    caps : List
        Ordered cap values.
    metric_col : str
        Metric to plot.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display.
    figsize : tuple
        Figure size (default: 7x5).

    Returns
    -------
    fig : plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Preprocess df
    df = _preprocess_df(df)

    # Gamma marginal effect (mean over caps, excluding cap=1 which is outlier)
    gamma_data = df[df['cap'] != 1].groupby('gamma')[metric_col].agg(['mean', 'std'])
    gamma_data = gamma_data.reindex(gammas)

    ax.errorbar(gamma_data.index, gamma_data['mean'], yerr=gamma_data['std'],
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                color='#4575b4')
    ax.fill_between(gamma_data.index,
                    gamma_data['mean'] - gamma_data['std'],
                    gamma_data['mean'] + gamma_data['std'],
                    alpha=0.2, color='#4575b4')
    ax.set_xlabel('γ', fontsize=14)
    ax.set_ylabel('F1', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(gammas)

    plt.tight_layout()

    if save_path:
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(save_path, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_ablation_cap_marginal(
    df: pd.DataFrame,
    gammas: List[float],
    caps: List,
    metric_col: str = 'f1_mean',
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (12, 8)
) -> plt.Figure:
    """Plot marginal effect of cap on the specified metric.

    Parameters
    ----------
    df : pd.DataFrame
        Summary dataframe.
    gammas : List[float]
        Ordered gamma values.
    caps : List
        Ordered cap values.
    metric_col : str
        Metric to plot.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display.
    figsize : tuple
        Figure size (default: 7x5).

    Returns
    -------
    fig : plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Preprocess df
    df = _preprocess_df(df)

    # Cap marginal effect (mean over gammas)
    cap_order = [c for c in caps if c is not None]
    if None in caps:
        cap_order.append(None)

    cap_data = df.groupby('cap')[metric_col].agg(['mean', 'std'])
    cap_data = cap_data.reindex(cap_order)

    x_pos = range(len(cap_data))
    ax.errorbar(x_pos, cap_data['mean'], yerr=cap_data['std'],
                fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=8,
                color='#d73027')
    ax.fill_between(x_pos,
                    cap_data['mean'] - cap_data['std'],
                    cap_data['mean'] + cap_data['std'],
                    alpha=0.2, color='#d73027')
    ax.set_xlabel('c', fontsize=14)
    ax.set_ylabel('F1', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(c) if c is not None else 'None' for c in cap_data.index], fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(save_path, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_ablation_gamma_convergence(
    df: pd.DataFrame,
    gammas: List[float],
    caps: List,
    gamma_values: List[float] = None,
    metric_col: str = 'f1_mean',
    metric_se_col: str = 'f1_se',
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (12, 8)
) -> plt.Figure:
    """Plot gamma convergence lines: F1 vs cap for multiple gamma values.

    Shows how F1 changes with cap for different gamma values,
    allowing comparison of gamma settings.

    Parameters
    ----------
    df : pd.DataFrame
        Summary dataframe.
    gammas : List[float]
        Ordered gamma values.
    caps : List
        Ordered cap values.
    gamma_values : List[float]
        Which gamma values to plot. If None, uses all available gammas.
    metric_col : str
        Metric to plot (default: 'f1_mean').
    metric_se_col : str
        Standard error column (default: 'f1_se').
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : plt.Figure
    """
    if gamma_values is None:
        gamma_values = gammas  # Use all gamma values

    # Preprocess df
    df = _preprocess_df(df)

    # Build cap order (numeric first, then None)
    cap_order = []
    for c in caps:
        if c is None:
            continue
        if isinstance(c, float) and np.isnan(c):
            continue
        cap_order.append(c)
    cap_order = sorted(cap_order)
    if None in caps or any(isinstance(c, float) and np.isnan(c) for c in caps):
        cap_order.append(None)

    x_pos = range(len(cap_order))
    x_labels = ['None' if c is None else str(int(c)) for c in cap_order]

    fig, ax = plt.subplots(figsize=figsize)

    # Use deep, saturated colors matching scatter plot style
    n_gammas = len(gamma_values)
    # Deep, saturated colors similar to MODEL_COLORS
    deep_colors = ['#e76f51', '#4a7fb8', '#2db87a', '#6ab028', '#d4a800', '#d45d8a']
    colors = deep_colors[:n_gammas]

    linestyles = ['-'] * n_gammas  # solid lines
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P'][:n_gammas]

    for idx, gamma in enumerate(gamma_values):
        df_gamma = df[df['gamma'] == gamma]

        y_values = []
        y_errors = []

        for cap in cap_order:
            if cap is None:
                row = df_gamma[df_gamma['cap'].isna()]
            else:
                row = df_gamma[df_gamma['cap'] == cap]

            if len(row) > 0:
                y_values.append(row[metric_col].values[0])
                if metric_se_col in row.columns:
                    y_errors.append(row[metric_se_col].values[0])
                else:
                    y_errors.append(0)
            else:
                y_values.append(np.nan)
                y_errors.append(0)

        ax.errorbar(x_pos, y_values, yerr=y_errors,
                   fmt=f'{markers[idx % len(markers)]}-', color=colors[idx],
                   capsize=4, capthick=1.5, linewidth=2, markersize=7,
                   label=f'$\\gamma$ = {gamma}', alpha=1.0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlabel('c', fontsize=14)
    ax.set_ylabel('F1', fontsize=14)
    ax.legend(fontsize=10, frameon=True, framealpha=0.9, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        # Save as PDF only
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(save_path, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_ablation_profile(
    df: pd.DataFrame,
    gammas: List[float],
    caps: List,
    fixed_gamma: float = 1.0,
    metric_col: str = 'f1_mean',
    metric_se_col: str = 'f1_se',
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (12, 8)
) -> plt.Figure:
    """Plot profile line plot for a fixed gamma across cap values.

    Shows F1-score (or other metric) vs cap with error bars.
    X-axis: cap values with None at the far right.
    Used to show the effect of cap on performance for a fixed gamma.

    Parameters
    ----------
    df : pd.DataFrame
        Summary dataframe.
    gammas : List[float]
        Ordered gamma values.
    caps : List
        Ordered cap values.
    fixed_gamma : float
        The gamma value to fix for the profile (default: 1.0).
    metric_col : str
        Metric to plot (default: 'f1_mean').
    metric_se_col : str
        Standard error column (default: 'f1_se').
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : plt.Figure
    """
    # Preprocess df
    df = _preprocess_df(df)

    # Filter for fixed gamma
    df_gamma = df[df['gamma'] == fixed_gamma].copy()

    # Build cap order (None last)
    cap_order = []
    for c in caps:
        if c is None:
            continue
        if isinstance(c, float) and np.isnan(c):
            continue
        cap_order.append(c)
    # Add None at end if there are NaN values in original data
    if None in caps or any(isinstance(c, float) and np.isnan(c) for c in caps):
        cap_order.append(None)

    # Extract data for each cap
    x_labels = []
    y_values = []
    y_errors = []

    for cap in cap_order:
        if cap is None:
            row = df_gamma[df_gamma['cap'].isna()]
        else:
            row = df_gamma[df_gamma['cap'] == cap]

        if len(row) > 0:
            x_labels.append('None' if cap is None else str(int(cap)))
            y_values.append(row[metric_col].values[0])
            if metric_se_col in row.columns:
                y_errors.append(row[metric_se_col].values[0])
            else:
                y_errors.append(0)
        else:
            # Handle NaN cap values
            if cap is None or (isinstance(cap, float) and np.isnan(cap)):
                x_labels.append('None')
            else:
                x_labels.append(str(int(cap)))
            y_values.append(np.nan)
            y_errors.append(0)

    x_pos = range(len(x_labels))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot with error bars
    ax.errorbar(x_pos, y_values, yerr=y_errors,
                fmt='o-', capsize=6, capthick=2, linewidth=2.5, markersize=10,
                color='#d62728', ecolor='#666666',
                markerfacecolor='white', markeredgewidth=2)

    # Add value labels on points
    for i, (x, y, err) in enumerate(zip(x_pos, y_values, y_errors)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlabel('c', fontsize=14)
    ax.set_ylabel('F1', fontsize=14)
    ax.set_title('')
    ax.grid(True, alpha=0.3)

    # Find best point
    if not all(np.isnan(y_values)):
        best_idx = np.nanargmax(y_values)
        ax.scatter([best_idx], [y_values[best_idx]], s=200, zorder=5,
                  marker='*', color='gold', edgecolor='black', linewidth=1)

    plt.tight_layout()

    if save_path:
        # Save as PDF only
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(save_path, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_ablation_rank(
    df: pd.DataFrame,
    gammas: List[float],
    caps: List,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (12, 8)
) -> plt.Figure:
    """Plot median rank heatmap across F1, MSE, and FDR.

    Lower rank (greener) = better performance. Best cell highlighted with green border.

    Parameters
    ----------
    df : pd.DataFrame
        Summary dataframe.
    gammas : List[float]
        Gamma values.
    caps : List
        Cap values.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : plt.Figure
    """
    # Preprocess df
    df = _preprocess_df(df)
    df_rank = df.copy()

    # Rank each metric
    # F1: higher is better → rank ascending
    # MSE/FDR: lower is better → rank ascending
    df_rank['f1_rank'] = df_rank['f1_mean'].rank(ascending=False)
    df_rank['mse_rank'] = df_rank['mse_mean'].rank(ascending=True)
    df_rank['fdr_rank'] = df_rank['fdr_mean'].rank(ascending=True)

    # Median rank across F1, MSE, FDR
    df_rank['median_rank'] = df_rank[['f1_rank', 'mse_rank', 'fdr_rank']].median(axis=1)

    # Create pivot
    pivot = df_rank.pivot_table(index='gamma', columns='cap', values='median_rank')
    pivot = pivot.reindex(index=gammas)

    cap_order = [c for c in caps if c is not None]
    if None in caps:
        cap_order.append(None)
    pivot = pivot[[c for c in cap_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=figsize)

    # Inverted colormap: low rank = green = good
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) if c is not None else 'None' for c in pivot.columns], fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{g:.1f}' for g in pivot.index], fontsize=12)
    ax.set_xlabel('c', fontsize=14)
    ax.set_ylabel('γ', fontsize=14)
    ax.set_title('')

    # Find best cell (lowest median rank)
    best_idx = np.nanargmin(pivot.values)
    best_i, best_j = np.unravel_index(best_idx, pivot.values.shape)

    # Annotate with rank values, highlight best cell
    for i in range(pivot.values.shape[0]):
        for j in range(pivot.values.shape[1]):
            val = pivot.values[i, j]
            color = 'white' if val > 30 else 'black'
            fontweight = 'bold' if (i, j) == (best_i, best_j) else 'normal'
            text = f'{val:.1f}' + (' *' if (i, j) == (best_i, best_j) else '')
            ax.text(j, i, text, ha='center', va='center',
                   fontsize=10, color=color, fontweight=fontweight)

    # Draw green box around best cell
    rect = plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                         fill=False, edgecolor='green', linewidth=3)
    ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Median Rank', fontsize=14)

    plt.tight_layout()

    if save_path:
        # Save as PDF only
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(save_path, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_all_ablation(
    df: pd.DataFrame,
    gammas: List[float],
    caps: List,
    output_dir: str,
    prefix: str = 'ablation',
    dpi: int = 150
) -> dict:
    """Generate all ablation visualization plots.

    Parameters
    ----------
    df : pd.DataFrame
        Summary dataframe.
    gammas : List[float]
        Gamma values.
    caps : List
        Cap values.
    output_dir : str
        Directory to save plots.
    prefix : str
        Filename prefix.
    dpi : int
        Resolution for PNG output.

    Returns
    -------
    dict : Mapping of plot name to file paths.
    """
    import os

    paths = {}

    # 1. F1 heatmap
    path = os.path.join(output_dir, f'{prefix}_f1_heatmap.pdf')
    plot_ablation_heatmap(df, 'f1_mean', gammas, caps,
                          'F1 Score: gamma × cap', cmap='YlOrRd',
                          save_path=path, show=False)
    paths['f1_heatmap'] = path

    # 2. MSE heatmap
    path = os.path.join(output_dir, f'{prefix}_mse_heatmap.pdf')
    plot_ablation_heatmap(df, 'mse_mean', gammas, caps,
                          'MSE: gamma × cap', cmap='YlOrRd',
                          save_path=path, show=False, fmt='.2f')
    paths['mse_heatmap'] = path

    # 3. FDR heatmap
    path = os.path.join(output_dir, f'{prefix}_fdr_heatmap.pdf')
    plot_ablation_heatmap(df, 'fdr_mean', gammas, caps,
                          'FDR: gamma × cap', cmap='YlOrRd',
                          save_path=path, show=False)
    paths['fdr_heatmap'] = path

    # 4. TPR heatmap
    path = os.path.join(output_dir, f'{prefix}_tpr_heatmap.pdf')
    plot_ablation_heatmap(df, 'tpr_mean', gammas, caps,
                          'TPR (Recall): gamma × cap', cmap='YlOrRd',
                          save_path=path, show=False, fmt='.2f')
    paths['tpr_heatmap'] = path

    # 5. Gamma marginal effect
    path = os.path.join(output_dir, f'{prefix}_gamma_marginal.pdf')
    plot_ablation_gamma_marginal(df, gammas, caps,
                                  save_path=path, show=False)
    paths['gamma_marginal'] = path

    # 6. Cap marginal effect
    path = os.path.join(output_dir, f'{prefix}_cap_marginal.pdf')
    plot_ablation_cap_marginal(df, gammas, caps,
                                save_path=path, show=False)
    paths['cap_marginal'] = path

    # 7. Rank heatmap
    path = os.path.join(output_dir, f'{prefix}_rank_heatmap.pdf')
    plot_ablation_rank(df, gammas, caps,
                       save_path=path, show=False)
    paths['rank_heatmap'] = path

    return paths
