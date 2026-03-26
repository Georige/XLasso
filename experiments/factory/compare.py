#!/usr/bin/env python
"""
Multi-Algorithm / Multi-Config / Multi-Runs Comparison Hub
==========================================================
Compares algorithms, configurations, or stability across multiple runs.

Usage:
    # Compare algorithms with same config
    python factory/compare.py --algo unilasso nlasso postlasso

    # Compare specific configurations
    python factory/compare.py --config configs/stage2/*_best.yaml

    # Compare stability across multiple runs
    python factory/compare.py --algo unilasso --runs 1 5 10

    # Full comparison
    python factory/compare.py \
        --algo unilasso nlasso postlasso \
        --config configs/stage2/*_best.yaml \
        --runs 1 5 10 \
        --metrics f1 r2 sparsity precision recall
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.factory.run import run_experiment, load_config, generate_experiment_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-algorithm/config comparison experiments"
    )
    parser.add_argument(
        "--algo",
        nargs="+",
        help="Algorithms to compare (xlasso, nlasso, postlasso)",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        help="Config files to compare",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        help="Number of repeats for stability check",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["f1", "r2", "sparsity", "precision", "recall"],
        help="Metrics to compare",
    )
    parser.add_argument(
        "--output-dir",
        default="results/pilot/compare",
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=300,
        help="Number of samples for synthetic data",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=500,
        help="Number of features",
    )
    parser.add_argument(
        "--n-nonzero",
        type=int,
        default=20,
        help="Number of non-zero coefficients",
    )
    return parser.parse_args()


def run_algorithm_comparison(algorithms, metrics, n_folds, n_samples, n_features, n_nonzero):
    """Compare multiple algorithms with same data settings."""
    results = []

    for algo in algorithms:
        print(f"\n[compare] Running algorithm: {algo}")

        config = {
            "experiment": f"compare_{algo}",
            "algo": algo,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_nonzero": n_nonzero,
            "sigma": 1.0,
            "correlation_type": "pairwise",
            "rho": 0.5,
            "n_repeats": 1,
            "cv_folds": n_folds,
            "output_dir": "results/pilot/compare",
            "lambda_1": 0.01,
            "lambda_2": 0.01,
            "group_threshold": 0.7,
        }

        try:
            # Create temp config file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config, f)
                temp_config = f.name

            result = run_experiment(config_path=temp_config)

            # Load summary
            exp_dir = Path(result["exp_dir"])
            summary_path = exp_dir / "summary.csv"

            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)
                for metric in metrics:
                    # Find column matching metric_mean
                    col_name = f"{metric}_mean"
                    if col_name in summary_df.columns:
                        results.append({
                            "algo": algo,
                            "metric": metric,
                            "value": float(summary_df.iloc[0][col_name]),
                        })

            os.unlink(temp_config)

        except Exception as e:
            print(f"[compare] ERROR with {algo}: {e}")
            continue

    return pd.DataFrame(results)


def run_config_comparison(configs, metrics, n_folds):
    """Compare multiple config files."""
    results = []

    for config_path in configs:
        config_name = Path(config_path).stem
        print(f"\n[compare] Running config: {config_name}")

        try:
            result = run_experiment(config_path=config_path)

            # Load summary
            exp_dir = Path(result["exp_dir"])
            summary_path = exp_dir / "summary.csv"

            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)
                for metric in metrics:
                    col_name = f"{metric}_mean"
                    if col_name in summary_df.columns:
                        results.append({
                            "config": config_name,
                            "metric": metric,
                            "value": float(summary_df.iloc[0][col_name]),
                        })

        except Exception as e:
            print(f"[compare] ERROR with {config_path}: {e}")
            continue

    return pd.DataFrame(results)


def run_stability_comparison(algo, runs_list, n_folds, n_samples, n_features, n_nonzero):
    """Compare stability across multiple run counts."""
    results = []

    for n_runs in runs_list:
        print(f"\n[compare] Running stability test: {n_runs} repeats")

        config = {
            "experiment": f"stability_{algo}_r{n_runs}",
            "algo": algo,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_nonzero": n_nonzero,
            "sigma": 1.0,
            "correlation_type": "pairwise",
            "rho": 0.5,
            "n_repeats": n_runs,
            "cv_folds": n_folds,
            "output_dir": "results/pilot/compare",
            "lambda_1": 0.01,
            "lambda_2": 0.01,
            "group_threshold": 0.7,
        }

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config, f)
                temp_config = f.name

            result = run_experiment(config_path=temp_config)

            # Load summary
            exp_dir = Path(result["exp_dir"])
            summary_path = exp_dir / "summary.csv"

            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)
                row = {"n_runs": n_runs}
                for metric in metrics:
                    col_name = f"{metric}_mean"
                    if col_name in summary_df.columns:
                        row[metric] = float(summary_df.iloc[0][col_name])
                results.append(row)

            os.unlink(temp_config)

        except Exception as e:
            print(f"[compare] ERROR with {n_runs} runs: {e}")
            continue

    return pd.DataFrame(results)


def compute_rankings(results_df, metric_col="value"):
    """Compute rankings for each metric."""
    rankings = {}

    for metric in results_df["metric"].unique():
        metric_df = results_df[results_df["metric"] == metric].copy()
        metric_df = metric_df.sort_values(metric_col, ascending=False)
        metric_df["rank"] = range(1, len(metric_df) + 1)

        rankings[f"by_{metric}"] = {
            f"rank_{row['rank']}": {
                k: v for k, v in row.items() if k not in ["metric", "rank"]
            }
            for _, row in metric_df.iterrows()
        }

    return rankings


def save_comparison_outputs(results_df, rankings, output_dir, compare_name):
    """Save all comparison outputs."""
    output_path = Path(output_dir) / f"{compare_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary_path = output_path / "summary.csv"
    results_df.to_csv(summary_path, index=False)

    # Save rankings
    for rank_name, rank_data in rankings.items():
        rank_path = output_path / f"rank_{rank_name}.yaml"
        with open(rank_path, "w") as f:
            yaml.dump(rank_data, f, default_flow_style=False)

    # Save config
    config_path = output_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            {
                "compare_name": compare_name,
                "output_dir": str(output_path),
                "timestamp": datetime.now().isoformat(),
            },
            f,
        )

    # Generate README
    readme_path = output_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(f"# Comparison: {compare_name}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Summary\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n## Rankings\n\n")
        for rank_name, rank_data in rankings.items():
            f.write(f"\n### {rank_name}\n\n")
            for rank_key, entry in rank_data.items():
                f.write(f"- {rank_key}: {entry}\n")

    print(f"[compare] Results saved to {output_path}")
    return output_path


def main():
    args = parse_args()

    if not any([args.algo, args.config, args.runs]):
        print("[compare] ERROR: Must specify --algo, --config, or --runs")
        return 1

    all_results = []
    metrics = args.metrics

    # Algorithm comparison
    if args.algo:
        print(f"\n[compare] === Algorithm Comparison ===")
        print(f"[compare] Algorithms: {args.algo}")

        algo_results = run_algorithm_comparison(
            algorithms=args.algo,
            metrics=metrics,
            n_folds=args.n_folds,
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_nonzero=args.n_nonzero,
        )

        if not algo_results.empty:
            algo_results["type"] = "algorithm"
            all_results.append(algo_results)

    # Config comparison
    if args.config:
        print(f"\n[compare] === Config Comparison ===")
        print(f"[compare] Configs: {args.config}")

        config_results = run_config_comparison(
            configs=args.config,
            metrics=metrics,
            n_folds=args.n_folds,
        )

        if not config_results.empty:
            config_results["type"] = "config"
            all_results.append(config_results)

    # Stability comparison
    if args.runs and args.algo:
        print(f"\n[compare] === Stability Comparison ===")
        print(f"[compare] Runs: {args.runs}")

        stability_results = run_stability_comparison(
            algo=args.algo[0],
            runs_list=args.runs,
            n_folds=args.n_folds,
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_nonzero=args.n_nonzero,
        )

        if not stability_results.empty:
            stability_results["type"] = "stability"
            all_results.append(stability_results)

    # Combine and save results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Compute rankings
        metric_col = "value"
        rankings = compute_rankings(combined_df, metric_col)

        # Save outputs
        compare_name = "_".join(
            filter(
                None,
                [
                    "_".join(args.algo) if args.algo else None,
                    f"configs_{len(args.config)}" if args.config else None,
                    f"runs_{'_'.join(map(str, args.runs))}" if args.runs else None,
                ],
            )
        ) or "comparison"

        output_path = save_comparison_outputs(
            combined_df, rankings, args.output_dir, compare_name
        )

        print(f"\n[compare] === Summary ===")
        print(combined_df.to_string(index=False))
        print(f"\n[compare] Output: {output_path}")

    else:
        print("[compare] WARNING: No results collected!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
