#!/usr/bin/env python3
"""
Simulation Metrics Scatter Plotting Script
==========================================
Generate publication-quality scatter plots from simulation raw.csv results.
Y-axis: algorithm metric, X-axis: SNR, all algorithms at same SNR as scatter points.

Usage:
    python factory/plot_metrics_scatter.py --exp 1 --metric f1
    python factory/plot_metrics_scatter.py --exp 1 --metric f1 --jitter 0.05
    python factory/plot_metrics_scatter.py --exp 1 --metric all --output ./plots/
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
from viz import (
    MODEL_COLORS,
    get_model_display_name,
    get_model_color,
    get_all_model_colors,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot scatter charts from simulation raw.csv results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--exp', type=int, default=1,
                        help='Experiment number (1-7, default: 1)')
    parser.add_argument('--metric', type=str, default='f1',
                        help='Metric to plot: f1, tpr, fdr, precision, recall, '
                             'sparsity, n_selected, mse, r2, all (default: f1)')
    parser.add_argument('--jitter', type=float, default=0.15,
                        help='Jitter amount for x-axis to separate scatter points (default: 0.15)')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Transparency of scatter points (default: 0.6)')
    parser.add_argument('--marker-size', type=float, default=80,
                        help='Scatter marker size (default: 80)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: auto-generated)')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show plot interactively (default: False)')
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
        'r2': 'R²',
        'accuracy': 'Accuracy',
    }
    return names.get(metric, metric)


def plot_scatter_by_snr(
    df,
    metric,
    jitter=0.15,
    alpha=0.6,
    marker_size=80,
    title=None,
    show_legend=True,
    save_path=None,
    show=True,
):
    """Plot scatter chart: y=metric, x=SNR, points colored by model."""
    # Filter out ElasticNet
    df = df[~df['model'].str.contains('ElasticNet', na=False)]

    # Get display names
    df['model_display'] = df['model'].apply(get_model_display_name)

    # Get unique models and SNR values (sorted descending)
    models = sorted(df['model_display'].unique(), key=lambda m: (0 if m == 'CG-Lasso' else 1, m))
    snrs = sorted(df['snr'].unique(), reverse=True)  # Descending order

    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(snrs) * 1.5), 7))

    # Map SNR to rank position for even spacing (0, 1, 2, 3, ...)
    snr_to_rank = {snr: i for i, snr in enumerate(snrs)}

    # Plot scatter for each model
    for model in models:
        model_data = df[df['model_display'] == model]
        color = get_model_color(model)

        # Get all unique SNR values for this model
        for snr in snrs:
            snr_data = model_data[model_data['snr'] == snr][metric].dropna()
            if len(snr_data) == 0:
                continue

            # Each point gets its own random x position
            x_positions = snr_to_rank[snr] + np.random.uniform(-jitter, jitter, size=len(snr_data))

            # Bin y values to detect clustering, and size points by cluster density
            y_values = snr_data.values
            y_unique = np.unique(y_values)
            counts = np.array([np.sum(y_values == y) for y in y_unique])

            # For very similar y values that are repeated, merge into one larger point
            # Use rounding to group similar values
            y_rounded = np.round(y_values, decimals=2)
            unique_y, counts = np.unique(y_rounded, return_counts=True)

            # Size proportional to count (min size for single points)
            sizes = np.clip(counts * 30, 30, 300)  # Scale count to size, clip max

            # Plot aggregated points with size proportional to count
            x_positions_agg = snr_to_rank[snr] + np.random.uniform(-jitter, jitter, size=len(unique_y))

            ax.scatter(
                x_positions_agg,
                unique_y,
                c=color,
                s=sizes,
                alpha=alpha,
                label=model,
                edgecolors='white',
                linewidth=0.5,
                zorder=3,
            )

    # X-axis: SNR values (evenly spaced in descending order)
    ax.set_xticks(range(len(snrs)))
    ax.set_xticklabels([str(s) for s in snrs], fontsize=12)
    ax.set_xlabel('SNR', fontsize=14)

    # Y-axis: metric
    ax.set_ylabel(get_metric_display_name(metric), fontsize=14)

    # Title
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    # Legend - only show unique labels if show_legend is True
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # Sort: CG-Lasso first
        sorted_items = sorted(by_label.items(), key=lambda x: (0 if x[0] == 'CG-Lasso' else 1, x[0]))
        # Position legend based on metric to avoid covering scatter points
        legend_loc_map = {
            'fdr': 'lower right',
            'tpr': 'lower left',
            'mse': 'upper left',
        }
        legend_loc = legend_loc_map.get(metric, 'upper right')
        ax.legend([h for _, h in sorted_items], [l for l, _ in sorted_items],
                  loc=legend_loc, framealpha=0.95, fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(snrs) - 0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_all_metrics_scatter(df, save_dir, jitter=0.15, alpha=0.6, marker_size=80):
    """Plot all metrics as a grid of scatter charts."""
    metrics = ['f1', 'tpr', 'fdr', 'precision', 'recall', 'sparsity', 'mse', 'r2']

    # Filter out ElasticNet
    df = df[~df['model'].str.contains('ElasticNet', na=False)]
    df['model_display'] = df['model'].apply(get_model_display_name)

    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    models = sorted(df['model_display'].unique(), key=lambda m: (0 if m == 'CG-Lasso' else 1, m))
    snrs = sorted(df['snr'].unique(), reverse=True)  # Descending order
    snr_to_rank = {snr: i for i, snr in enumerate(snrs)}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for model in models:
            model_data = df[df['model_display'] == model]

            for snr in snrs:
                snr_data = model_data[model_data['snr'] == snr][metric].dropna()
                if len(snr_data) == 0:
                    continue

                # Each point gets its own random x position
                x_positions = snr_to_rank[snr] + np.random.uniform(-jitter, jitter, size=len(snr_data))
                color = get_model_color(model)

                ax.scatter(
                    x_positions,
                    snr_data.values,
                    c=color,
                    s=marker_size * 0.5,
                    alpha=alpha,
                    edgecolors='white',
                    linewidth=0.3,
                    zorder=3,
                )

        ax.set_xticks(range(len(snrs)))
        ax.set_xticklabels([str(s) for s in snrs], fontsize=9)
        ax.set_ylabel(get_metric_display_name(metric), fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
        ax.set_axisbelow(True)

    # Add legend to the figure
    handles, labels = fig.axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    sorted_items = sorted(by_label.items(), key=lambda x: (0 if x[0] == 'CG-Lasso' else 1, x[0]))
    fig.legend([h for _, h in sorted_items], [l for l, _ in sorted_items],
               loc='upper right', bbox_to_anchor=(0.99, 0.99), ncol=len(sorted_items), framealpha=0.95, fontsize=10)

    plt.suptitle(f'Exp1: All Metrics by SNR', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(save_dir, f'all_metrics_scatter.pdf')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  All metrics scatter saved to: {output_path}")

    plt.close(fig)


def main():
    args = parse_args()

    print("=" * 60)
    print(f"Simulation Metrics Scatter Plotting")
    print(f"Experiment: Exp{args.exp}")
    print(f"Metric: {args.metric}")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...")
    df = load_raw_data(args.exp)
    print(f"  Loaded {len(df)} rows from raw.csv")
    print(f"  Models: {df['model'].nunique()}")

    # Filter out ElasticNet
    elasticnet_mask = df['model'].str.contains('ElasticNet', na=False)
    n_removed = elasticnet_mask.sum()
    df = df[~elasticnet_mask]
    print(f"  Filtered out ElasticNet: {n_removed} rows removed")
    print(f"  Remaining rows: {len(df)}")

    # Determine output directory
    if args.output is None:
        output_dir = os.path.join(xlasso_root, 'experiments', 'results', 'plots', 'scatter_plots')
    else:
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
    os.makedirs(output_dir, exist_ok=True)

    # Plot
    print("\n[2/3] Generating plot...")

    if args.metric == 'all':
        plot_all_metrics_scatter(
            df, output_dir,
            jitter=args.jitter, alpha=args.alpha, marker_size=args.marker_size
        )
        output_path = os.path.join(output_dir, 'all_metrics_scatter.pdf')
    else:
        metric_display = get_metric_display_name(args.metric)
        # No title for all experiments
        title = None
        # Only show legend for exp1
        show_legend = (args.exp == 1)
        # Exp3 is classification, use accuracy instead of mse
        if args.metric == 'mse' and args.exp == 3:
            print("  Note: Exp3 is classification, skipping MSE (no MSE column)")
            print("\n[3/3] Done!")
            return

        output_path = args.output if args.output else os.path.join(
            output_dir, f'exp{args.exp}_{args.metric}_scatter_by_snr.pdf'
        )

        fig, ax = plot_scatter_by_snr(
            df=df,
            metric=args.metric,
            jitter=args.jitter,
            alpha=args.alpha,
            marker_size=args.marker_size,
            title=title,
            show_legend=show_legend,
            save_path=output_path,
            show=args.show,
        )
        print(f"  Plot saved to: {output_path}")

    print("\n[3/3] Done!")


if __name__ == '__main__':
    main()
