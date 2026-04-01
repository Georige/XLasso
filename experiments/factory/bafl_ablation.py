#!/usr/bin/env python
"""
BAFL Parameter Ablation Experiment
==================================
2D parameter grid search over gamma and weight_cap (cap) for BAFL (PFLRegressorCV).

Usage:
    python factory/bafl_ablation.py --config configs/bafl_ablation/example.yaml

Output:
    - raw.csv: complete iteration record (all repeats, all parameter combinations)
    - summary.csv: aggregated statistics by parameter combination
    - report.md: ablation analysis report with heatmaps and best config
"""

import argparse
import inspect
import sys
import traceback
from datetime import datetime
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.factory.run import load_config, generate_experiment_dir, save_config


def parse_args():
    parser = argparse.ArgumentParser(description="BAFL parameter ablation")
    parser.add_argument(
        "--config", "-c", required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--output-dir", "-o", default=None, help="Override output directory"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config without running"
    )
    return parser.parse_args()


def grid_search(config):
    """
    2D Grid search over gamma and weight_cap parameters for BAFL.

    For each (gamma, cap) combination:
        - Run n_repeats independent experiments
        - Each repeat uses K-fold CV internally
        - Collect all metrics for statistical analysis
    """
    print("[bafl_ablation] Starting BAFL 2D Parameter Ablation")
    print(f"[bafl_ablation] Config: {config['experiment']}")

    # Extract parameter grids
    gamma_values = config.get("gamma_list", [0.3, 0.5, 1.0, 1.5, 2.0, 3.0])
    cap_values = config.get("cap_list", [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, None])

    # Handle None in cap_list (means no cap)
    cap_values = [c if c is not None else float('inf') for c in cap_values]
    cap_labels = [f"{c:.0f}" if c != float('inf') else "None" for c in cap_values]

    print(f"[bafl_ablation] gamma values: {gamma_values}")
    print(f"[bafl_ablation] cap values: {cap_labels}")
    print(f"[bafl_ablation] Total combinations: {len(gamma_values) * len(cap_values)}")

    # Generate experiment directory
    config["output_dir"] = config.get(
        "output_dir",
        "/home/lili/lyn/clear/NLasso/XLasso/experiments/results/bafl_ablation"
    )
    exp_dir = generate_experiment_dir(config)
    save_config(config, exp_dir)

    n_repeats = config.get("n_repeats", 5)
    n_folds = config.get("cv_folds", 5)
    n_jobs = config.get("n_jobs", -1)

    # Build all tasks
    tasks = []
    for gamma in gamma_values:
        for cap, cap_label in zip(cap_values, cap_labels):
            for repeat in range(n_repeats):
                tasks.append({
                    "gamma": gamma,
                    "cap": cap,
                    "cap_label": cap_label,
                    "repeat": repeat,
                    "config": config,
                    "exp_dir": exp_dir,
                    "n_folds": n_folds,
                })

    total_tasks = len(tasks)
    print(f"[bafl_ablation] Total tasks: {total_tasks} (gamma={len(gamma_values)} × cap={len(cap_values)} × repeat={n_repeats})")

    # Parallel execution
    print(f"[bafl_ablation] Parallel jobs: {n_jobs}")
    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_single_task)(task) for task in tasks
    )

    # Collect results
    all_results = []
    error_count = 0
    for i, result in enumerate(results_list):
        if result is None:
            error_count += 1
            continue
        if isinstance(result, dict) and "error" in result:
            print(f"[bafl_ablation] ERROR at gamma={result['gamma']}, cap={result['cap_label']}, repeat={result['repeat']}: {result['error']}")
            error_count += 1
            continue
        all_results.extend(result)

    if error_count > 0:
        print(f"[bafl_ablation] {error_count} tasks failed")

    if not all_results:
        raise RuntimeError("No successful trials in BAFL ablation")

    # Save raw results
    results_df = pd.DataFrame(all_results)
    raw_path = exp_dir / "raw.csv"
    results_df.to_csv(raw_path, index=False)

    # Generate summary statistics
    summary = generate_summary(results_df, exp_dir)

    # Generate report
    report_path = generate_report(config, exp_dir, results_df, summary, gamma_values, cap_labels)

    print(f"\n[bafl_ablation] Completed!")
    print(f"[bafl_ablation] Results saved to {exp_dir}")
    print(f"[bafl_ablation] Report: {report_path}")

    return {
        "status": "completed",
        "exp_dir": str(exp_dir),
        "report": str(report_path),
        "n_trials": len(all_results),
    }


def _run_single_task(task):
    """Run a single BAFL trial for one (gamma, cap, repeat) combination."""
    from sklearn.model_selection import KFold
    from experiments.modules import DataGenerator, MetricCalculator

    gamma = task["gamma"]
    cap = task["cap"]
    cap_label = task["cap_label"]
    repeat = task["repeat"]
    config = task["config"]
    exp_dir = task["exp_dir"]
    n_folds = task["n_folds"]

    seed = 42 + repeat
    weight_cap = None if cap == float('inf') else cap

    try:
        # Data generation
        data_gen = DataGenerator(random_state=seed)
        X, y, beta_true = data_gen.generate(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            n_nonzero=config["n_nonzero"],
            sigma=config.get("sigma", 1.0),
            correlation_type=config.get("correlation_type", "pairwise"),
            rho=config.get("rho", 0.5),
            block_size=config.get("block_size", 10),
            n_blocks=config.get("n_blocks", 50),
        )

        # CV splits
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kfold.split(X))

        # Build BAFL (PFLRegressorCV) parameters
        from experiments.modules import PFLRegressorCV
        algo_params = {
            "cv": n_folds,
            "lambda_ridge_list": tuple(config.get("lambda_ridge_list", [0.1, 1.0, 10.0, 100.0])),
            "gamma": gamma,
            "weight_cap": weight_cap,
            "alpha_min_ratio": config.get("alpha_min_ratio", 1e-4),
            "n_alpha": config.get("n_alpha", 100),
            "max_iter": config.get("max_iter", 2000),
            "tol": config.get("tol", 0.0001),
            "standardize": config.get("standardize", False),
            "fit_intercept": config.get("fit_intercept", True),
            "random_state": seed,
            "verbose": config.get("verbose", False),
            "n_jobs": 1,  # Already parallelized at task level
        }

        algo = PFLRegressorCV(**algo_params)
        algo.fit(X, y, cv_splits=splits)

        # Evaluate on last fold's validation set (or aggregate across folds)
        # For ablation, we use the held-out validation approach
        fold_metrics_list = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_val, y_val = X[val_idx], y[val_idx]
            y_pred = algo.predict(X_val)

            metrics_calc = MetricCalculator()
            fold_metrics = metrics_calc.calculate(
                y_true=y_val,
                y_pred=y_pred,
                beta_true=beta_true,
                beta_est=algo.coef_,
            )

            # Add metadata
            fold_metrics["gamma"] = gamma
            fold_metrics["cap"] = cap_label
            fold_metrics["repeat"] = repeat
            fold_metrics["fold"] = fold_idx
            fold_metrics["train_time"] = 0  # Already aggregated

            # Add best parameters from CV
            if hasattr(algo, 'best_alpha_'):
                fold_metrics["best_alpha"] = algo.best_alpha_
            if hasattr(algo, 'best_lambda_ridge_'):
                fold_metrics["best_lambda_ridge"] = algo.best_lambda_ridge_

            # Sign accuracy
            if hasattr(algo, 'signs_') and algo.signs_ is not None:
                true_signals_idx = np.where(np.abs(beta_true) > 1e-6)[0]
                if len(true_signals_idx) > 0:
                    signs_true = np.sign(beta_true[true_signals_idx])
                    signs_est = algo.signs_[true_signals_idx]
                    fold_metrics["sign_accuracy"] = np.mean(signs_true == signs_est)

            fold_metrics_list.append(fold_metrics)

        return fold_metrics_list

    except Exception as e:
        return {
            "error": str(e),
            "gamma": gamma,
            "cap": cap_label,
            "repeat": repeat,
        }


def generate_summary(results_df, exp_dir):
    """Generate summary statistics by (gamma, cap) combination, sorted by gamma and cap ascending."""
    # Aggregate across repeats and folds
    summary = results_df.groupby(['gamma', 'cap']).agg({
        'f1': ['mean', 'std'],
        'mse': ['mean', 'std'],
        'tpr': ['mean', 'std'],
        'fdr': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'sparsity': ['mean', 'std'],
        'n_selected': ['mean', 'std'],
        'sign_accuracy': ['mean', 'std'],
    }).round(4)

    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()

    # Sort by gamma ascending, then by cap ascending (None last)
    def cap_sort_key(x):
        if x == 'None':
            return float('inf')
        return float(x)
    summary['_cap_sort'] = summary['cap'].apply(cap_sort_key)
    summary = summary.sort_values(['gamma', '_cap_sort']).drop('_cap_sort', axis=1)

    summary_path = exp_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    return summary


def generate_report(config, exp_dir, results_df, summary, gamma_values, cap_labels):
    """Generate comprehensive ablation analysis report."""
    report_path = exp_dir / "report.md"

    # Sort gamma_values and cap_labels for consistent ordering
    gamma_values_sorted = sorted(gamma_values)
    cap_labels_sorted = sorted(cap_labels, key=lambda x: float('inf') if x == 'None' else float(x))

    # Prepare sorted pivot tables
    def make_sorted_pivot(summary, value_name):
        """Create pivot table with sorted axes (gamma ascending, cap ascending with None last)."""
        pivot = summary.pivot(index='gamma', columns='cap', values=value_name)
        # Sort rows by gamma
        pivot = pivot.sort_index()
        # Sort columns by cap (None last)
        def col_sort_key(c):
            if c == 'None':
                return float('inf')
            return float(c)
        pivot = pivot.reindex(sorted(pivot.columns, key=col_sort_key))
        pivot = pivot.round(4)
        return pivot

    with open(report_path, "w") as f:
        f.write(f"# BAFL Parameter Ablation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Experiment**: {config['experiment']}\n")
        f.write(f"**Data**: n={config['n_samples']}, p={config['n_features']}, n_nonzero={config['n_nonzero']}\n")
        f.write(f"**SNR**: {1.0 / config.get('sigma', 1.0):.2f} (sigma={config.get('sigma', 1.0)})\n")
        f.write(f"**Repeats**: {config.get('n_repeats', 5)} per configuration\n")
        f.write(f"**CV Folds**: {config.get('cv_folds', 5)}\n\n")

        f.write("## 1. Parameter Grid\n\n")
        f.write(f"- **gamma**: {gamma_values_sorted}\n")
        f.write(f"- **cap**: {cap_labels_sorted}\n\n")

        f.write("## 2. Best Configurations by Metric\n\n")

        # Best by F1
        best_f1_idx = summary['f1_mean'].idxmax()
        best_f1_row = summary.loc[best_f1_idx]
        f.write("### 2.1 Best by F1 (higher is better)\n\n")
        f.write(f"| gamma | cap | F1 Mean | F1 Std | MSE Mean | TPR Mean | FDR Mean |\n")
        f.write("|-------|-----|---------|--------|----------|----------|----------|\n")
        f.write(f"| {best_f1_row['gamma']} | {best_f1_row['cap']} | {best_f1_row['f1_mean']:.4f} | {best_f1_row['f1_std']:.4f} | {best_f1_row['mse_mean']:.4f} | {best_f1_row['tpr_mean']:.4f} | {best_f1_row['fdr_mean']:.4f} |\n\n")

        # Best by MSE
        best_mse_idx = summary['mse_mean'].idxmin()
        best_mse_row = summary.loc[best_mse_idx]
        f.write("### 2.2 Best by MSE (lower is better)\n\n")
        f.write(f"| gamma | cap | MSE Mean | MSE Std | F1 Mean | TPR Mean | FDR Mean |\n")
        f.write("|-------|-----|----------|---------|---------|----------|----------|\n")
        f.write(f"| {best_mse_row['gamma']} | {best_mse_row['cap']} | {best_mse_row['mse_mean']:.4f} | {best_mse_row['mse_std']:.4f} | {best_mse_row['f1_mean']:.4f} | {best_mse_row['tpr_mean']:.4f} | {best_mse_row['fdr_mean']:.4f} |\n\n")

        # Best by FDR
        best_fdr_idx = summary['fdr_mean'].idxmin()
        best_fdr_row = summary.loc[best_fdr_idx]
        f.write("### 2.3 Best by FDR (lower is better)\n\n")
        f.write(f"| gamma | cap | FDR Mean | FDR Std | F1 Mean | MSE Mean | TPR Mean |\n")
        f.write("|-------|-----|----------|---------|---------|----------|----------|\n")
        f.write(f"| {best_fdr_row['gamma']} | {best_fdr_row['cap']} | {best_fdr_row['fdr_mean']:.4f} | {best_fdr_row['fdr_std']:.4f} | {best_fdr_row['f1_mean']:.4f} | {best_fdr_row['mse_mean']:.4f} | {best_fdr_row['tpr_mean']:.4f} |\n\n")

        # Best by Sign Accuracy
        if 'sign_accuracy_mean' in summary.columns:
            best_sign_idx = summary['sign_accuracy_mean'].idxmax()
            best_sign_row = summary.loc[best_sign_idx]
            f.write("### 2.4 Best by Sign Accuracy (higher is better)\n\n")
            f.write(f"| gamma | cap | Sign Acc Mean | Sign Acc Std | F1 Mean | MSE Mean |\n")
            f.write("|-------|-----|---------------|--------------|---------|----------|\n")
            f.write(f"| {best_sign_row['gamma']} | {best_sign_row['cap']} | {best_sign_row['sign_accuracy_mean']:.4f} | {best_sign_row['sign_accuracy_std']:.4f} | {best_sign_row['f1_mean']:.4f} | {best_sign_row['mse_mean']:.4f} |\n\n")

        # 3D Heatmap data: F1 by (gamma, cap)
        f.write("## 3. F1 Heatmap Data (gamma × cap)\n\n")
        f1_pivot = make_sorted_pivot(summary, 'f1_mean')
        f.write("```\n")
        f.write(f1_pivot.to_string())
        f.write("\n```\n\n")

        # MSE Heatmap data
        f.write("## 4. MSE Heatmap Data (gamma × cap)\n\n")
        mse_pivot = make_sorted_pivot(summary, 'mse_mean')
        f.write("```\n")
        f.write(mse_pivot.to_string())
        f.write("\n```\n\n")

        # TPR Heatmap data
        f.write("## 5. TPR Heatmap Data (gamma × cap)\n\n")
        tpr_pivot = make_sorted_pivot(summary, 'tpr_mean')
        f.write("```\n")
        f.write(tpr_pivot.to_string())
        f.write("\n```\n\n")

        # FDR Heatmap data
        f.write("## 6. FDR Heatmap Data (gamma × cap)\n\n")
        fdr_pivot = make_sorted_pivot(summary, 'fdr_mean')
        f.write("```\n")
        f.write(fdr_pivot.to_string())
        f.write("\n```\n\n")

        # Sign Accuracy Heatmap
        if 'sign_accuracy_mean' in summary.columns:
            f.write("## 7. Sign Accuracy Heatmap Data (gamma × cap)\n\n")
            sign_pivot = make_sorted_pivot(summary, 'sign_accuracy_mean')
            f.write("```\n")
            f.write(sign_pivot.to_string())
            f.write("\n```\n\n")

        # Full summary table
        f.write("## 8. Complete Summary Table\n\n")
        f.write("| gamma | cap | F1 | MSE | TPR | FDR | Precision | Recall | R2 | Sparsity | Sign Acc |\n")
        f.write("|-------|-----|-----|-----|-----|-----|----------|--------|-----|----------|----------|\n")
        for _, row in summary.iterrows():
            f.write(f"| {row['gamma']} | {row['cap']} | {row['f1_mean']:.4f}±{row['f1_std']:.4f} | {row['mse_mean']:.4f}±{row['mse_std']:.4f} | {row['tpr_mean']:.4f}±{row['tpr_std']:.4f} | {row['fdr_mean']:.4f}±{row['fdr_std']:.4f} | {row['precision_mean']:.4f}±{row['precision_std']:.4f} | {row['recall_mean']:.4f}±{row['recall_std']:.4f} | {row['r2_mean']:.4f}±{row['r2_std']:.4f} | {row['sparsity_mean']:.4f}±{row['sparsity_std']:.4f} | {row['sign_accuracy_mean']:.4f}±{row['sign_accuracy_std']:.4f} |\n")
        f.write("\n")

        # Key findings
        f.write("## 9. Key Findings\n\n")

        # Analyze gamma effect (sorted by gamma ascending)
        gamma_effect = summary.groupby('gamma')['f1_mean'].mean().sort_index()
        f.write("### 9.1 Gamma Effect (averaged over cap, sorted by gamma ascending)\n\n")
        for gamma_val, f1_val in gamma_effect.items():
            f.write(f"- gamma={gamma_val}: F1={f1_val:.4f}\n")
        f.write("\n")

        # Analyze cap effect (sorted by cap ascending, None last)
        cap_effect = summary.groupby('cap')['f1_mean'].mean()
        cap_effect = cap_effect.reset_index()
        cap_effect['_cap_sort'] = cap_effect['cap'].apply(lambda x: float('inf') if x == 'None' else float(x))
        cap_effect = cap_effect.sort_values('_cap_sort').drop('_cap_sort', axis=1).set_index('cap')['f1_mean']
        f.write("### 9.2 Cap Effect (averaged over gamma, sorted by cap ascending)\n\n")
        for cap_val, f1_val in cap_effect.items():
            f.write(f"- cap={cap_val}: F1={f1_val:.4f}\n")
        f.write("\n")

        # Best overall config
        best_overall_idx = summary['f1_mean'].idxmax()
        best_overall = summary.loc[best_overall_idx]
        f.write("### 9.3 Best Overall Configuration\n\n")
        f.write(f"- **gamma**: {best_overall['gamma']}\n")
        f.write(f"- **cap**: {best_overall['cap']}\n")
        f.write(f"- **F1**: {best_overall['f1_mean']:.4f} ± {best_overall['f1_std']:.4f}\n")
        f.write(f"- **MSE**: {best_overall['mse_mean']:.4f} ± {best_overall['mse_std']:.4f}\n")
        f.write(f"- **TPR**: {best_overall['tpr_mean']:.4f} ± {best_overall['tpr_std']:.4f}\n")
        f.write(f"- **FDR**: {best_overall['fdr_mean']:.4f} ± {best_overall['fdr_std']:.4f}\n")
        f.write("\n")

        f.write("---\n")
        f.write(f"*Report generated: {datetime.now().isoformat()}*\n")

    return report_path


def run_ablation(config_path, output_dir=None, dry_run=False):
    """Run BAFL ablation experiment."""
    config = load_config(config_path)
    if output_dir:
        config["output_dir"] = output_dir
    config["config_path"] = config_path

    if dry_run:
        print("[bafl_ablation] Dry-run mode - validation only")
        print(f"[bafl_ablation] Config: {config['experiment']}")
        print(f"[bafl_ablation] gamma values: {config.get('gamma_list', [])}")
        print(f"[bafl_ablation] cap values: {config.get('cap_list', [])}")
        print(f"[bafl_ablation] n_repeats: {config.get('n_repeats', 5)}")
        return {"status": "dry_run", "config": config}

    return grid_search(config)


def main():
    args = parse_args()

    try:
        result = run_ablation(
            config_path=args.config,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        print(f"\n[bafl_ablation] completed!")
        return 0

    except Exception as e:
        print(f"[bafl_ablation] FATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
