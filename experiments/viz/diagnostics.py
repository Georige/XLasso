"""
Diagnostics Module - Model Diagnostic Visualizations
===================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional, List, Dict, Any


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    xlabel: str = "Predicted Value",
    ylabel: str = "Residual",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot residuals vs predicted values."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, **kwargs)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    sorted_idx = np.argsort(y_pred)
    smoothed = np.convolve(residuals[sorted_idx], np.ones(5)/5, mode='valid')
    ax.plot(y_pred[sorted_idx][2:-2], smoothed, color='green', linewidth=2,
            label='Smoothed trend')
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


def plot_qq(
    residuals: np.ndarray,
    title: str = "Q-Q Plot",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot Q-Q plot to assess normality of residuals."""
    fig, ax = plt.subplots(figsize=(8, 8))
    sorted_residuals = np.sort(residuals)
    n = len(residuals)
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = stats.norm.ppf(quantiles)
    ax.scatter(theoretical_quantiles, sorted_residuals, alpha=0.5, **kwargs)
    slope, intercept = np.polyfit(theoretical_quantiles, sorted_residuals, 1)
    ax.plot(theoretical_quantiles, slope * theoretical_quantiles + intercept,
            color='red', linewidth=2, label='Reference line')
    ax.legend()
    ax.set_xlabel("Theoretical Quantiles", fontsize=12)
    ax.set_ylabel("Sample Quantiles", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:min(5000, len(residuals))])
    ax.text(0.05, 0.95, f'Shapiro-Wilk p={shapiro_p:.4f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_loo_pred(
    y_true: np.ndarray,
    y_loo_pred: np.ndarray,
    title: str = "Leave-One-Out Prediction",
    xlabel: str = "True Value",
    ylabel: str = "LOO Predicted Value",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot LOO predictions vs true values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_loo_pred, alpha=0.5, **kwargs)
    min_val = min(y_true.min(), y_loo_pred.min())
    max_val = max(y_true.max(), y_loo_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect prediction')
    ax.legend()
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ss_res = np.sum((y_true - y_loo_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    ax.text(0.05, 0.95, f'LOO R2={r2:.4f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    title: str = "Learning Curve",
    xlabel: str = "Training Set Size",
    ylabel: str = "Score",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot learning curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_scores, marker='o', linewidth=2,
            label='Training score', **kwargs)
    ax.plot(train_sizes, val_scores, marker='s', linewidth=2,
            label='Validation score', **kwargs)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_residual_histogram(
    residuals: np.ndarray,
    bins: int = 30,
    title: str = "Residual Distribution",
    xlabel: str = "Residual",
    ylabel: str = "Frequency",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot histogram of residuals."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=bins, density=True, alpha=0.7, **kwargs)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    ax.plot(x, p, 'r-', linewidth=2, label='Normal fit')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)