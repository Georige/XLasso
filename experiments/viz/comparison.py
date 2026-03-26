"""
Comparison Module - Multi-Algorithm Comparison Visualizations
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any


def plot_metric_compare(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    title: str = "Metric Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot bar chart comparing metrics across algorithms/configs."""
    names = list(results.keys())
    n_metrics = len(metrics)
    n_groups = len(names)
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8 / n_metrics
    x_pos = np.arange(n_groups)
    for i, metric in enumerate(metrics):
        values = [results[name].get(metric, 0) for name in names]
        offset = (i - n_metrics/2 + 0.5) * bar_width
        bars = ax.bar(x_pos + offset, values, bar_width,
                     label=metric.capitalize(), **kwargs)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bar_chart(
    values: List[float],
    labels: List[str],
    title: str = "Comparison",
    ylabel: str = "Score",
    color: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot horizontal or vertical bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(values))
    bars = ax.bar(x_pos, values, color=color, **kwargs)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
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


def plot_radar_chart(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    title: str = "Radar Chart Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot radar chart for multi-metric comparison."""
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    for name, metrics_dict in results.items():
        values = [metrics_dict.get(m, 0) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=name, **kwargs)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_rank_heatmap(
    rankings: Dict[str, List[Dict]],
    title: str = "Ranking Heatmap",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot heatmap of rankings across different metrics."""
    metric_names = list(rankings.keys())
    n_metrics = len(metric_names)
    max_rank = max(len(rankings[m]) for m in metric_names)
    matrix = np.full((n_metrics, max_rank), np.nan)
    for i, metric in enumerate(metric_names):
        for j, entry in enumerate(rankings[metric]):
            for k, v in entry.items():
                if k == 'rank':
                    matrix[i, j] = v
                    break
    fig, ax = plt.subplots(figsize=(max_rank * 1.5, n_metrics * 1.5))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(np.arange(max_rank))
    ax.set_yticks(np.arange(n_metrics))
    ax.set_xticklabels([f'Rank {i+1}' for i in range(max_rank)])
    ax.set_yticklabels([m.replace('rank_best_by_', '') for m in metric_names])
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, label='Rank')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrix(
    selected: np.ndarray,
    true_nonzero: np.ndarray,
    n_features: int,
    title: str = "Feature Selection Confusion",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
):
    """Plot confusion matrix for feature selection."""
    selected_set = set(selected)
    true_set = set(true_nonzero)
    tp = len(selected_set & true_set)
    fp = len(selected_set - true_set)
    fn = len(true_set - selected_set)
    tn = n_features - tp - fp - fn
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14)
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, matrix[i, j],
                          ha="center", va="center", fontsize=20,
                          color="white" if matrix[i, j] > matrix.max()/2 else "black")
    plt.colorbar(im, ax=ax)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)