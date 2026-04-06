"""
Features Module - Feature Selection Visualizations
==================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any

from ._shared import MODEL_COLORS


def plot_feature_importance(
    importance: np.ndarray,
    feature_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    title: str = "Feature Importance",
    xlabel: str = "Importance",
    ylabel: str = "Feature",
    color: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot feature importance as horizontal bar chart."""
    if top_k is not None:
        top_indices = np.argsort(np.abs(importance))[-top_k:]
        importance = importance[top_indices]
        if feature_ids is not None:
            feature_ids = [feature_ids[i] for i in top_indices]
    else:
        top_indices = np.argsort(np.abs(importance))
    n_features = len(importance)
    y_pos = np.arange(n_features)
    fig, ax = plt.subplots(figsize=(10, max(6, n_features * 0.3)))
    ax.barh(y_pos, importance, color=color, **kwargs)
    if feature_ids is not None:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_ids, fontsize=8)
    else:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Feature {i}' for i in top_indices], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_selected_features(
    selected: np.ndarray,
    coefficients: np.ndarray,
    true_nonzero: Optional[np.ndarray] = None,
    feature_ids: Optional[List[str]] = None,
    title: str = "Selected Features",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot selected features with their coefficients."""
    n_selected = len(selected)
    fig, ax = plt.subplots(figsize=(12, max(6, n_selected * 0.4)))
    y_pos = np.arange(n_selected)
    colors = []
    for idx in selected:
        if true_nonzero is not None and idx in true_nonzero:
            colors.append('green')
        elif true_nonzero is not None:
            colors.append('red')
        else:
            colors.append('steelblue')
    ax.barh(y_pos, coefficients, color=colors, **kwargs)
    if feature_ids is not None:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_ids[i] for i in selected], fontsize=8)
    else:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Feature {i}' for i in selected], fontsize=8)
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(0, color='black', linewidth=0.5)
    if true_nonzero is not None:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='True Positive'),
            Patch(facecolor='red', label='False Positive'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_coefficient_path(
    coefs: np.ndarray,
    lmdas: np.ndarray,
    feature_indices: Optional[List[int]] = None,
    title: str = "Coefficient Path",
    xlabel: str = "-log(Lambda)",
    ylabel: str = "Coefficient",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot coefficient regularization path for selected features."""
    fig, ax = plt.subplots(figsize=(10, 6))
    neg_log_lmdas = -np.log(lmdas)
    if feature_indices is not None:
        for idx in feature_indices:
            ax.plot(neg_log_lmdas, coefs[:, idx], linewidth=2, label=f'Feature {idx}')
        ax.legend()
    else:
        nonzero_mask = np.any(coefs != 0, axis=0)
        for j in np.where(nonzero_mask)[0]:
            ax.plot(neg_log_lmdas, coefs[:, j], linewidth=1, alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_feature_correlation(
    corr_matrix: np.ndarray,
    feature_ids: Optional[List[str]] = None,
    threshold: float = 0.7,
    title: str = "Feature Correlation Matrix",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot feature correlation matrix with threshold highlighting."""
    n_features = corr_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))
    if feature_ids is not None:
        ax.set_xticklabels(feature_ids, rotation=90, fontsize=8)
        ax.set_yticklabels(feature_ids, fontsize=8)
    else:
        ax.set_xticklabels([f'{i}' for i in range(n_features)], fontsize=8)
        ax.set_yticklabels([f'{i}' for i in range(n_features)], fontsize=8)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, label='Correlation')
    for i in range(n_features):
        for j in range(n_features):
            if abs(corr_matrix[i, j]) >= threshold and i != j:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                           fill=False, edgecolor='black', linewidth=2))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bafl_coefficient_path(
    coefs: np.ndarray,
    alphas: np.ndarray,
    true_signal_indices: List[int],
    decoy_indices: List[int],
    optimal_alpha: Optional[float] = None,
    min_alpha: Optional[float] = None,
    title: Optional[str] = None,
    xlabel: str = r"$-\log_{10}(\alpha)$",
    ylabel: str = r"$\hat \beta_j$",
    true_signal_color: str = '#5C9BEB',  # Bright blue for coefficient curves
    decoy_color: str = MODEL_COLORS['ElasticNet'],  # Pink for decoys
    noise_color: str = "#7f7f7f",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot BAFL coefficient path with true signal vs noise distinction.

    Parameters
    ----------
    coefs : np.ndarray, shape (n_alphas, n_features)
        Coefficient matrix along regularization path.
    alphas : np.ndarray, shape (n_alphas,)
        Regularization parameter values (from large to small).
    true_signal_indices : List[int]
        Indices of true signal variables.
    decoy_indices : List[int]
        Indices of noise decoy variables (correlated with signals).
    optimal_alpha : float, optional
        Optimal alpha selected by 1-SE rule (BAFL color vertical line).
    min_alpha : float, optional
        Alpha with minimum CV error (red vertical line).
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    true_signal_color : str
        Color for true signal paths (default: bright blue #5C9BEB).
    decoy_color : str
        Color for noise decoy paths.
    noise_color : str
        Color for independent noise paths.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the plot.
    **kwargs
        Additional arguments passed to ax.plot().

    Returns
    -------
    fig, ax : tuple
        Figure and axis objects.
    """
    n_features = coefs.shape[1]
    noise_indices = [j for j in range(n_features)
                    if j not in true_signal_indices and j not in decoy_indices]

    fig, ax = plt.subplots(figsize=(10, 7))

    # X-axis: -log10(alpha)
    neg_log_alpha = -np.log10(alphas)

    # Epsilon to make zero-coefficient lines visible
    eps = 1e-5

    # Check if decoys have any non-zero coefficients
    has_nonzero_decoys = any(
        np.any(coefs[:, j] != 0) for j in decoy_indices
    ) if decoy_indices else False

    # Plot by category
    for j in range(n_features):
        coef_values = coefs[:, j].copy()
        if j in true_signal_indices:
            ax.plot(neg_log_alpha, coef_values,
                    color=true_signal_color, linewidth=2.0, alpha=0.9)
        elif j in decoy_indices:
            # Add small offset so zero-coef lines are visible
            coef_values = np.where(np.abs(coef_values) < eps, np.sign(coef_values + 1e-10) * eps, coef_values)
            ax.plot(neg_log_alpha, coef_values,
                    color=decoy_color, linewidth=2.0, alpha=0.9, linestyle='--',
                    marker='o', markersize=3, markevery=max(1, len(alphas)//10))
        else:
            ax.plot(neg_log_alpha, coef_values,
                    color=noise_color, linewidth=0.5, alpha=0.3)

    # Twin vertical lines: min MSE and 1-SE optimal
    if min_alpha is not None:
        min_neg_log = -np.log10(min_alpha)
        ax.axvline(x=min_neg_log, color='#d62728', linestyle='--',
                   linewidth=1.5, alpha=0.8, label=f'$\\alpha_{{min}}$')

    if optimal_alpha is not None:
        optimal_neg_log = -np.log10(optimal_alpha)
        ax.axvline(x=optimal_neg_log, color=MODEL_COLORS['CG-Lasso'], linestyle='--',
                   linewidth=2.5, alpha=0.9, label=f'$\\alpha_{{1SE}}$ (optimal)')

    # Zero horizontal line
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=true_signal_color, linewidth=2,
               label=f'True Signal (n={len(true_signal_indices)})'),
    ]
    if has_nonzero_decoys:
        legend_elements.append(
            Line2D([0], [0], color=decoy_color, linewidth=2, linestyle='--', marker='o', markersize=4,
                   label=f'Noise Decoy (n={len(decoy_indices)})')
        )
    legend_elements.append(
        Line2D([0], [0], color=noise_color, linewidth=0.5,
               label=f'Noise Signal (n={len(noise_indices)})'),
    )
    if min_alpha is not None:
        legend_elements.append(
            Line2D([0], [0], color='#d62728', linewidth=1.5, linestyle='--',
                   label=f'$\\alpha_{{min}}$ (min MSE)')
        )
    if optimal_alpha is not None:
        legend_elements.append(
            Line2D([0], [0], color=MODEL_COLORS['CG-Lasso'], linewidth=2.5, linestyle='--',
                   label=f'$\\alpha_{{1SE}}$ (optimal)')
        )
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim([neg_log_alpha[0], neg_log_alpha[-1]])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax