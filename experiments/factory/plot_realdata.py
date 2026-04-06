#!/usr/bin/env python3
"""
Real Data Visualization Script
==============================
Generate bar charts and feature selection frequency plots from real data results.

Usage:
    python factory/plot_realdata.py
    python factory/plot_realdata.py --input /path/to/realdata_dir
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add XLasso root to path
xlasso_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, xlasso_root)
sys.path.insert(0, os.path.join(xlasso_root, 'experiments'))

from viz import (
    MODEL_COLORS,
    get_model_display_name,
    get_model_color,
    get_all_model_colors,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot real data experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Input directory containing summary.json and selection_frequency.csv')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for plots (default: same as input)')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show plots interactively')
    return parser.parse_args()


def get_short_name(algo_name):
    """Map algorithm name to display name."""
    if 'pfl' in algo_name.lower() or 'bafl' in algo_name.lower():
        return 'CG-Lasso'
    elif 'adaptive' in algo_name.lower():
        return 'AdaptiveLasso'
    elif 'elastic' in algo_name.lower():
        return 'ElasticNet'
    elif 'relaxed' in algo_name.lower():
        return 'RelaxedLasso'
    elif 'unilasso' in algo_name.lower():
        return 'Unilasso'
    elif 'lasso' in algo_name.lower():
        return 'Lasso'
    return algo_name


def plot_metrics_bar(summary_data, save_path=None, show=True):
    """Plot bar charts for MSE and Model Size across algorithms."""
    import json

    algorithms = list(summary_data.keys())
    display_names = [get_short_name(a) for a in algorithms]

    mse_means = [summary_data[a]['test_mse']['mean'] for a in algorithms]
    mse_stds = [summary_data[a]['test_mse']['std'] for a in algorithms]
    size_means = [summary_data[a]['model_size']['mean'] for a in algorithms]
    size_stds = [summary_data[a]['model_size']['std'] for a in algorithms]

    # Sort by model size (CG-Lasso first)
    sorted_data = sorted(zip(display_names, mse_means, mse_stds, size_means, size_stds, algorithms),
                        key=lambda x: (0 if x[0] == 'CG-Lasso' else 1, x[0]))
    display_names = [x[0] for x in sorted_data]
    mse_means = [x[1] for x in sorted_data]
    mse_stds = [x[2] for x in sorted_data]
    size_means = [x[3] for x in sorted_data]
    size_stds = [x[4] for x in sorted_data]

    # Get colors
    colors = [get_model_color(name) for name in display_names]

    # MSE plot
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(len(display_names))
    bars1 = ax1.bar(x_pos, mse_means, yerr=mse_stds,
                     color=colors, capsize=5, alpha=0.85,
                     edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(display_names, rotation=45, ha='right', fontsize=11)
    ax1.set_ylabel('MSE', fontsize=14)
    ax1.set_title('')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, mse_means):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    mse_path = save_path.replace('.pdf', '_mse.pdf')
    fig1.savefig(mse_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {mse_path}")
    plt.close(fig1)

    # Model Size plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    bars2 = ax2.bar(x_pos, size_means, yerr=size_stds,
                    color=colors, capsize=5, alpha=0.85,
                    edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(display_names, rotation=45, ha='right', fontsize=11)
    ax2.set_ylabel('# Selected Features', fontsize=14)
    ax2.set_title('')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, size_means):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    size_path = save_path.replace('.pdf', '_size.pdf')
    fig2.savefig(size_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {size_path}")
    plt.close(fig2)

    return None


def plot_selection_frequency(freq_df, save_path=None, show=True, top_n=30):
    """Plot feature selection frequency across algorithms."""
    algo_cols = [c for c in freq_df.columns if c.endswith('_freq')]
    algorithms = [c.replace('_freq', '') for c in algo_cols]
    display_names = [get_short_name(a) for a in algorithms]

    # Get top features by CG-Lasso (pfl_regressor_cv) frequency
    cglasso_col = None
    for col in algo_cols:
        if 'pfl' in col.lower() or 'bafl' in col.lower():
            cglasso_col = col
            break
    if cglasso_col is None:
        # Fallback to first algorithm if CG-Lasso not found
        cglasso_col = algo_cols[0]
    cglasso_freq = freq_df[cglasso_col].values
    top_features = np.argsort(cglasso_freq)[::-1][:top_n]

    n_algos = len(algorithms)
    bar_width = 0.8 / n_algos
    x_pos = np.arange(len(top_features))

    # Get deep colors
    deep_colors = ['#e76f51', '#4a7fb8', '#2db87a', '#6ab028', '#d4a800', '#d45d8a'][:n_algos]

    fig, ax = plt.subplots(figsize=(16, 8))

    for i, (algo, display_name, col) in enumerate(zip(algorithms, display_names, algo_cols)):
        freqs = freq_df[col].values[top_features]
        positions = x_pos + (i - n_algos/2 + 0.5) * bar_width
        ax.bar(positions, freqs, width=bar_width * 0.9,
               color=deep_colors[i % len(deep_colors)], alpha=1.0,
               label=display_name, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'F{int(f)}' for f in freq_df['feature'].values[top_features]],
                       rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Feature Index', fontsize=14)
    ax.set_ylabel('Model Size', fontsize=14)
    ax.set_title('')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_consensus_heatmap(input_dir, save_path=None, show=True):
    """Plot feature selection frequency heatmaps for all algorithms.

    Each cell = one feature, color = selection frequency.
    Features arranged in a square-ish grid, colored by frequency.
    """
    import json
    import ast
    from collections import Counter

    # Find all algorithm files
    raw_files = [f for f in os.listdir(input_dir) if f.startswith('raw_') and f.endswith('.csv')]
    algo_names = [f.replace('raw_', '').replace('.csv', '') for f in raw_files]

    print(f"  Found algorithms: {algo_names}")

    # Determine total number of features from first file
    first_path = os.path.join(input_dir, raw_files[0])
    df_sample = pd.read_csv(first_path, nrows=5)
    max_feat = 0
    for _, row in df_sample.iterrows():
        feats = ast.literal_eval(row['selected_features'])
        if feats:
            max_feat = max(max_feat, max(feats))
    df_full = pd.read_csv(first_path)
    for _, row in df_full.iterrows():
        feats = ast.literal_eval(row['selected_features'])
        if feats:
            max_feat = max(max_feat, max(feats))
    n_features = max_feat + 1
    print(f"  Total features: {n_features}")

    # Calculate grid dimensions (square-ish)
    grid_size = int(np.ceil(np.sqrt(n_features)))
    print(f"  Grid size: {grid_size} x {grid_size}")

    # Generate one heatmap per algorithm
    for algo_file, algo_name in zip(raw_files, algo_names):
        raw_path = os.path.join(input_dir, algo_file)
        df = pd.read_csv(raw_path)
        n_repeats = len(df)

        # Count selection frequency for each feature
        freq_counter = Counter()
        for _, row in df.iterrows():
            feats = ast.literal_eval(row['selected_features'])
            for feat in feats:
                freq_counter[feat] += 1

        # Create frequency matrix for visualization
        # Arrange features in a grid, fill empty cells with NaN
        freq_matrix = np.full((grid_size, grid_size), np.nan)
        for feat_idx in range(n_features):
            row = feat_idx // grid_size
            col = feat_idx % grid_size
            freq_matrix[row, col] = freq_counter.get(feat_idx, 0) / n_repeats

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 12))
        # Use masked array to handle NaN as white/transparent
        masked_matrix = np.ma.masked_invalid(freq_matrix)
        im = ax.imshow(masked_matrix, cmap='Blues', vmin=0, vmax=1, aspect='equal')
        ax.set_facecolor('white')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('')
        ax.set_xlabel(f'{n_features} Features',
                     fontsize=11)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Selection Frequency', fontsize=12)

        plt.tight_layout()

        # Save individual figure
        algo_save_path = save_path.replace('.pdf', f'_{algo_name}.pdf')
        fig.savefig(algo_save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {algo_save_path}")
        plt.close(fig)

    return None


def plot_model_size_scatter(input_dir, save_path=None, show=True):
    """Plot scatter + bar chart for Model Size across algorithms.

    Scatter points from raw data with jitter, overlaid with semi-transparent bars.
    """
    import ast
    import json

    # Find all algorithm files
    raw_files = [f for f in os.listdir(input_dir) if f.startswith('raw_') and f.endswith('.csv')]
    algo_names = [f.replace('raw_', '').replace('.csv', '') for f in raw_files]

    # Sort: CG-Lasso first
    sorted_files = sorted(zip(algo_names, raw_files),
                        key=lambda x: (0 if 'pfl' in x[0].lower() else 1, x[0]))
    algo_names = [x[0] for x in sorted_files]
    raw_files = [x[1] for x in sorted_files]

    # Get colors
    display_names = [get_short_name(a) for a in algo_names]
    colors = [get_model_color(name) for name in display_names]

    # Collect model sizes for each algorithm
    algo_sizes = {}
    for algo_name in algo_names:
        raw_path = os.path.join(input_dir, f'raw_{algo_name}.csv')
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
            algo_sizes[algo_name] = df['model_size'].values

    # Load summary for bar heights
    summary_path = os.path.join(input_dir, 'summary.json')
    summary_data = {}
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)

    fig, ax = plt.subplots(figsize=(max(10, len(algo_names) * 1.5), 7))

    x_pos = np.arange(len(algo_names))
    jitter = 0.15
    bar_width = 0.6

    for i, (algo_name, color) in enumerate(zip(algo_names, colors)):
        if algo_name not in algo_sizes:
            continue

        sizes_data = algo_sizes[algo_name]
        if len(sizes_data) == 0:
            continue

        # Aggregate by unique values with counts
        unique_vals, counts = np.unique(sizes_data, return_counts=True)
        sizes = np.clip(counts * 5, 30, 200)

        # Jitter x positions for scatter
        x_positions = i + np.random.uniform(-jitter, jitter, size=len(unique_vals))

        # Scatter points
        ax.scatter(
            x_positions,
            unique_vals,
            c=color,
            s=sizes,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5,
            zorder=3,
            label=display_names[i]
        )

        # Bar from summary mean with error bar
        if algo_name in summary_data:
            mean_val = summary_data[algo_name]['model_size']['mean']
            std_val = summary_data[algo_name]['model_size']['std']
            ax.bar(i, mean_val, width=bar_width, color=color, alpha=0.3,
                   edgecolor=color, linewidth=1, zorder=2,
                   yerr=std_val, capsize=4, error_kw={'linewidth': 1.5})

    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('# Selected Features', fontsize=14)
    ax.set_title('')
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(algo_names) - 0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_mse_scatter(input_dir, save_path=None, show=True):
    """Plot scatter + bar chart for MSE across algorithms.

    Scatter points from raw data with jitter, overlaid with semi-transparent bars.
    """
    import ast
    import json

    # Find all algorithm files
    raw_files = [f for f in os.listdir(input_dir) if f.startswith('raw_') and f.endswith('.csv')]
    algo_names = [f.replace('raw_', '').replace('.csv', '') for f in raw_files]

    # Sort: CG-Lasso first
    sorted_files = sorted(zip(algo_names, raw_files),
                        key=lambda x: (0 if 'pfl' in x[0].lower() else 1, x[0]))
    algo_names = [x[0] for x in sorted_files]
    raw_files = [x[1] for x in sorted_files]

    # Get colors
    display_names = [get_short_name(a) for a in algo_names]
    colors = [get_model_color(name) for name in display_names]

    # Collect test MSE for each algorithm
    algo_mse = {}
    for algo_name in algo_names:
        raw_path = os.path.join(input_dir, f'raw_{algo_name}.csv')
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
            algo_mse[algo_name] = df['test_mse'].values

    # Load summary for bar heights
    summary_path = os.path.join(input_dir, 'summary.json')
    summary_data = {}
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)

    fig, ax = plt.subplots(figsize=(max(10, len(algo_names) * 1.5), 7))

    x_pos = np.arange(len(algo_names))
    jitter = 0.15
    bar_width = 0.6

    for i, (algo_name, color) in enumerate(zip(algo_names, colors)):
        if algo_name not in algo_mse:
            continue

        mse_data = algo_mse[algo_name]
        if len(mse_data) == 0:
            continue

        # Aggregate by unique values with counts (rounded to 3 decimals)
        mse_rounded = np.round(mse_data, decimals=3)
        unique_vals, counts = np.unique(mse_rounded, return_counts=True)
        sizes = np.clip(counts * 5, 30, 200)

        # Jitter x positions for scatter
        x_positions = i + np.random.uniform(-jitter, jitter, size=len(unique_vals))

        # Scatter points
        ax.scatter(
            x_positions,
            unique_vals,
            c=color,
            s=sizes,
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5,
            zorder=3,
            label=display_names[i]
        )

        # Bar from summary mean with error bar
        if algo_name in summary_data:
            mean_val = summary_data[algo_name]['test_mse']['mean']
            std_val = summary_data[algo_name]['test_mse']['std']
            ax.bar(i, mean_val, width=bar_width, color=color, alpha=0.3,
                   edgecolor=color, linewidth=1, zorder=2,
                   yerr=std_val, capsize=4, error_kw={'linewidth': 1.5})

    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('MSE', fontsize=14)
    ax.set_title('')
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(algo_names) - 0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def main():
    args = parse_args()

    # Default input directory
    if args.input is None:
        input_dir = os.path.join(xlasso_root, 'experiments', 'results', 'output_all', 'realdata')
    else:
        input_dir = args.input

    if args.output is None:
        output_dir = input_dir
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Load summary data
    summary_path = os.path.join(input_dir, 'summary.json')
    if not os.path.exists(summary_path):
        print(f"Error: summary.json not found at {summary_path}")
        return 1

    import json
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)

    print(f"\nAlgorithms found: {list(summary_data.keys())}")

    # Plot 1: MSE scatter + bar chart
    print("\n[1/2] Generating MSE scatter + bar chart...")
    mse_path = os.path.join(output_dir, 'realdata_metrics_mse.pdf')
    plot_mse_scatter(input_dir, save_path=mse_path, show=args.show)

    # Plot 1b: Model Size scatter + bar chart
    print("\n[1b] Generating model size scatter + bar chart...")
    size_path = os.path.join(output_dir, 'realdata_metrics_size_scatter.pdf')
    plot_model_size_scatter(input_dir, save_path=size_path, show=args.show)

    # Plot 2: Feature selection frequency
    print("\n[2/2] Generating feature selection frequency plot...")
    freq_path = os.path.join(input_dir, 'selection_frequency.csv')
    if os.path.exists(freq_path):
        freq_df = pd.read_csv(freq_path)
        freq_plot_path = os.path.join(output_dir, 'realdata_selection_frequency.pdf')
        plot_selection_frequency(freq_df, save_path=freq_plot_path, show=args.show)
        print(f"  Saved to: {freq_plot_path}")
    else:
        print(f"  Warning: selection_frequency.csv not found at {freq_path}")

    # Plot 3: Consensus heatmaps for all algorithms
    print("\n[3/3] Generating consensus heatmaps...")
    consensus_path = os.path.join(output_dir, 'realdata_consensus_heatmap.pdf')
    plot_consensus_heatmap(input_dir, save_path=consensus_path, show=args.show)
    print(f"  Saved to: {output_dir}/realdata_consensus_heatmap_<algo>.pdf")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
