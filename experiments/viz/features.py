"""
Features Module - Feature Selection Visualizations
==================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any


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