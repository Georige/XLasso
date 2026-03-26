"""
Curves Module - Convergence and Lambda Search Visualization
==========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any


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