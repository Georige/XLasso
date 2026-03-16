#!/usr/bin/env python3
"""
Command-line script to run linear Gaussian experiments.
Usage:
    python scripts/run_linear_experiment.py --n-repeats 10 --output experiments/results
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.linear_experiment import run_all_linear_scenarios


def main():
    parser = argparse.ArgumentParser(
        description='Run all linear Gaussian simulation experiments'
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

    print(f"Running all linear experiments...")
    print(f"Output directory: {args.output}")
    print(f"n_repeats override: {args.n_repeats}")

    os.makedirs(args.output, exist_ok=True)

    all_results = run_all_linear_scenarios(
        output_dir=args.output,
        n_repeats_override=args.n_repeats,
        save_plots=not args.no_plots
    )

    print("\nDone! Summary of results:")
    for scenario, df in all_results.items():
        print(f"\n{scenario}:")
        print(df[['f1', 'tpr', 'fpr', 'mse']].round(3))

    return 0


if __name__ == '__main__':
    sys.exit(main())
