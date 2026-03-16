#!/usr/bin/env python3
"""
Command-line script to run nonlinear experiments.
Usage:
    python scripts/run_nonlinear_experiment.py --n-repeats 10 --output experiments/results
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.nonlinear_experiment import run_all_nonlinear_experiments, compare_nonlinear_models


def main():
    parser = argparse.ArgumentParser(
        description='Run nonlinear simulation experiments'
    )
    parser.add_argument(
        '--n-repeats', type=int, default=None,
        help='Override number of repetitions (use smaller for testing)'
    )
    parser.add_argument(
        '--output', type=str, default='experiments/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip generating plots'
    )

    args = parser.parse_args()

    print(f"Running all nonlinear experiments...")
    print(f"Output directory: {args.output}")
    print(f"n_repeats override: {args.n_repeats}")

    os.makedirs(args.output, exist_ok=True)

    all_results = run_all_nonlinear_experiments(
        output_dir=args.output,
        n_repeats_override=args.n_repeats,
        save_plots=not args.no_plots
    )

    # Generate summary comparison
    compare_nonlinear_models(results_dir=args.output, output_dir=args.output)

    print("\nDone! Summary of results:")
    for scenario, df in all_results.items():
        print(f"\n{scenario}:")
        # Show F1 for full configurations
        full_rows = [idx for idx in df.index if 'full' in idx]
        if full_rows:
            print(df.loc[full_rows, [('f1', 'mean'), ('mse', 'mean')]].round(3))

    return 0


if __name__ == '__main__':
    sys.exit(main())
