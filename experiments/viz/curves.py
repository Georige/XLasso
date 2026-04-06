"""
Curves Module - Convergence and Lambda Search Visualization
==========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any

from ._shared import MODEL_COLORS


def plot_convergence(
    losses: List[float],
    title: str = "Convergence Curve",
    xlabel: str = "Iteration",
    ylabel: str = "Loss",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot convergence curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    iterations = range(len(losses))
    ax.plot(iterations, losses, linewidth=2, **kwargs)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_lambda_sweep(
    lmdas: np.ndarray,
    losses: np.ndarray,
    best_lambda: Optional[float] = None,
    title: str = "Lambda Sweep",
    xlabel: str = "log(Lambda)",
    ylabel: str = "CV Loss",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot lambda regularization sweep results."""
    fig, ax = plt.subplots(figsize=(8, 6))
    log_lmdas = -np.log(lmdas)
    ax.plot(log_lmdas, losses, linewidth=2, marker='o', **kwargs)
    if best_lambda is not None:
        best_log_lmda = -np.log(best_lambda)
        ax.axvline(x=best_log_lmda, color='red', linestyle='--',
                   label=f'Best lambda={best_lambda:.4f}')
        ax.legend()
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cv_fold_results(
    fold_results: Dict[str, List[float]],
    metric: str = "f1",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot cross-validation fold results."""
    fig, ax = plt.subplots(figsize=(8, 6))
    folds = list(fold_results.keys())
    values = list(fold_results.values())
    x_pos = np.arange(len(folds))
    bars = ax.bar(x_pos, values, **kwargs)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(folds)
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(title or f"CV {metric.capitalize()} by Fold", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_lambda_path(
    coefs: np.ndarray,
    lmdas: np.ndarray,
    n_nonzero: Optional[np.ndarray] = None,
    title: str = "Coefficient Path",
    xlabel: str = "-log(Lambda)",
    ylabel: str = "Coefficients",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot Lasso coefficient regularization path."""
    fig, ax = plt.subplots(figsize=(10, 6))
    neg_log_lmdas = -np.log(lmdas)
    for j in range(coefs.shape[1]):
        ax.plot(neg_log_lmdas, coefs[:, j], linewidth=1.5, **kwargs)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    if n_nonzero is not None:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_positions = np.linspace(0, len(neg_log_lmdas) - 1, min(6, len(neg_log_lmdas)), dtype=int)
        ax2.set_xticks(neg_log_lmdas[tick_positions])
        ax2.set_xticklabels(n_nonzero[tick_positions])
        ax2.set_xlabel("Number of Nonzero Coefficients", fontsize=10)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cv_loss_curve(
    avg_losses: np.ndarray,
    lmdas: np.ndarray,
    fold_losses: Optional[np.ndarray] = None,
    best_idx: Optional[int] = None,
    title: str = "Cross-Validation Loss Curve",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot cross-validation loss curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    neg_log_lmdas = -np.log(lmdas)
    ax.plot(neg_log_lmdas, avg_losses, linewidth=2, marker='o',
            label='CV Loss', **kwargs)
    if fold_losses is not None:
        std_losses = np.std(fold_losses, axis=0)
        ax.fill_between(neg_log_lmdas,
                        avg_losses - std_losses,
                        avg_losses + std_losses,
                        alpha=0.2)
    if best_idx is not None:
        best_lmda = lmdas[best_idx]
        best_loss = avg_losses[best_idx]
        best_log_lmda = -np.log(best_lmda)
        ax.scatter([best_log_lmda], [best_loss], color='red', s=100,
                   zorder=5, label=f'Best lambda={best_lmda:.4f}')
        ax.axvline(x=best_log_lmda, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel(r"$-\log(\lambda)$", fontsize=12)
    ax.set_ylabel("CV Loss (MSE)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bafl_cv_error_path(
    alphas: np.ndarray,
    mean_error: np.ndarray,
    std_error: np.ndarray,
    nselected: np.ndarray,
    optimal_alpha: Optional[float] = None,
    min_alpha: Optional[float] = None,
    family: str = 'gaussian',
    xlabel: str = r"$-\log_{10}(\alpha)$",
    ylabel: str = None,
    error_color: str = '#66d2a6',  # Lasso cyan
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot BAFL CV error path with publication-quality styling.

    Features:
    - U-shaped CV error curve with ±1 SE error bars
    - Top axis showing Degrees of Freedom (non-zero coefficients)
    - Twin vertical lines: min MSE (red) and 1-SE optimal (BAFL color)

    Parameters
    ----------
    alphas : np.ndarray
        Regularization parameter values.
    mean_error : np.ndarray
        Mean CV error across folds for each alpha.
    std_error : np.ndarray
        Standard error (std / sqrt(K)) across folds.
    nselected : np.ndarray
        Mean number of non-zero coefficients across folds.
    optimal_alpha : float, optional
        Optimal alpha selected by 1-SE rule.
    min_alpha : float, optional
        Alpha with minimum CV error.
    family : str
        'gaussian' for MSE, 'binomial' for log-loss.
    xlabel : str
        X-axis label.
    ylabel : str, optional
        Y-axis label for error. Auto-generated if None.
    error_color : str
        Color for error curve.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display the plot.
    """
    if ylabel is None:
        ylabel = "Cross-Validation MSE" if family == 'gaussian' else "Cross-Validation Log-Loss"

    fig, ax1 = plt.subplots(figsize=(10, 7))

    # X-axis: -log10(alpha)
    neg_log_alpha = -np.log10(alphas)

    # Primary axis: CV error curve with error bars
    ax1.errorbar(neg_log_alpha, mean_error, yerr=std_error,
                 color=error_color, linewidth=2, fmt='o', markersize=4,
                 capsize=3, capthick=1, elinewidth=1,
                 alpha=0.8, label='CV Error ± 1 SE')

    # Twin vertical lines
    if min_alpha is not None:
        min_neg_log = -np.log10(min_alpha)
        ax1.axvline(x=min_neg_log, color='#d62728', linestyle='--',
                   linewidth=1.5, alpha=0.8, label=f'$\\alpha_{{min}}$')

    if optimal_alpha is not None:
        optimal_neg_log = -np.log10(optimal_alpha)
        ax1.axvline(x=optimal_neg_log, color=MODEL_COLORS['CG-Lasso'], linestyle='--',
                   linewidth=2.5, alpha=0.9, label=f'$\\alpha_{{1SE}}$ (optimal)')

    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle=':')

    # Top axis: Degrees of Freedom (non-zero coefficients)
    ax2 = ax1.twiny()
    # Plot with markers at data points
    ax2.plot(neg_log_alpha, nselected, color='#888888', linewidth=1.5,
             linestyle='-', alpha=0.6, marker='s', markersize=4,
             markevery=max(1, len(alphas)//15))
    ax2.set_xlabel("Degrees of Freedom", fontsize=12, color='#888888')
    ax2.tick_params(axis='x', labelcolor='#888888')

    # Set top axis ticks to match bottom axis exactly
    ax2.set_xlim(ax1.get_xlim())
    # Use same tick positions as bottom axis
    tick_pos = np.linspace(0, len(alphas)-1, min(6, len(alphas)), dtype=int)
    ax2.set_xticks(neg_log_alpha[tick_pos])
    ax2.set_xticklabels([f'{int(nselected[i])}' for i in tick_pos], fontsize=9)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=error_color, linewidth=2, marker='o', markersize=6,
               label='CV Error ± 1 SE'),
        Line2D([0], [0], color='#d62728', linewidth=1.5, linestyle='--',
               label=f'$\\alpha_{{min}}$ (min MSE)'),
        Line2D([0], [0], color=MODEL_COLORS['CG-Lasso'], linewidth=2.5, linestyle='--',
               label=f'$\\alpha_{{1SE}}$ (optimal)'),
        Line2D([0], [0], color='#888888', linewidth=1.5, marker='s', markersize=5,
               label='Degrees of Freedom'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.95,
              fontsize=10, frameon=True)

    ax1.set_xlim([neg_log_alpha[0], neg_log_alpha[-1]])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax1, ax2