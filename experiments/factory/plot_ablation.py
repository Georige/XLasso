#!/usr/bin/env python
"""
Plot BAFL Ablation Results
==========================

Generate visualization plots for BAFL parameter ablation experiment results.

Usage:
    python factory/plot_ablation.py --input /path/to/ablation_result_dir

Output:
    - *f1_heatmap.pdf
    - *mse_heatmap.pdf
    - *fdr_heatmap.pdf
    - *tpr_heatmap.pdf
    - *gamma_marginal.pdf
    - *cap_marginal.pdf
    - *rank_heatmap.pdf
    - *profile_gamma10.pdf
    - *gamma_convergence.pdf
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import viz to apply paper style
import experiments.viz
from experiments.viz import ablation


def parse_args():
    parser = argparse.ArgumentParser(description="Plot BAFL ablation results")
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to ablation result directory containing summary.csv"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory for plots (default: results/plots/ablation/<experiment_name>)"
    )
    parser.add_argument(
        "--prefix", "-p", default="ablation",
        help="Output filename prefix (default: ablation)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't display plots, only save"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        sys.exit(1)

    summary_path = input_path / "summary.csv"
    if not summary_path.exists():
        print(f"Error: summary.csv not found in {input_path}")
        sys.exit(1)

    # Determine output directory
    # Default: results/plot/<experiment_name>/
    if args.output is None:
        # Use the experiment name as subfolder under results/plot/
        experiment_name = input_path.name
        output_path = input_path.parent / "plots" / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {summary_path}")
    df = pd.read_csv(summary_path)

    # Get unique gamma and cap values (in sorted order)
    gammas = sorted(df['gamma'].unique())

    # Caps need special handling - convert nan back to None
    df['cap'] = df['cap'].apply(lambda x: None if pd.isna(x) else x)
    caps_raw = df['cap'].unique()
    caps = sorted(caps_raw, key=lambda x: (x is None, x if x is not None else float('inf')))

    print(f"Gamma values: {gammas}")
    print(f"Cap values: {caps}")
    print(f"Total configurations: {len(df)}")

    # Generate all plots
    print(f"\nGenerating visualizations...")
    print(f"Output directory: {output_path}")

    show = not args.no_show

    paths = ablation.plot_all_ablation(
        df, gammas, caps,
        output_dir=str(output_path),
        prefix=args.prefix
    )

    # Add profile plot for fixed gamma=1.0
    path = str(output_path / f'{args.prefix}_profile_gamma10.pdf')
    ablation.plot_ablation_profile(
        df, gammas, caps,
        fixed_gamma=1.0,
        metric_col='f1_mean',
        metric_se_col='f1_se',
        save_path=path,
        show=False
    )
    paths['profile_gamma10'] = path

    # Add gamma convergence plot (selected gamma values)
    path = str(output_path / f'{args.prefix}_gamma_convergence.pdf')
    ablation.plot_ablation_gamma_convergence(
        df, gammas, caps,
        gamma_values=[1.0, 1.5, 2.0, 3.0],
        metric_col='f1_mean',
        metric_se_col='f1_se',
        save_path=path,
        show=False
    )
    paths['gamma_convergence'] = path

    print("\nGenerated plots:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print(f"\nAll plots saved to: {output_path}")


if __name__ == "__main__":
    main()
