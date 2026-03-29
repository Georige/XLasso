#!/usr/bin/env python
"""
Two-Stage Hyperparameter Tuning Driver
======================================
Stage1: Grid search to identify optimal structural region
Stage2: Fine-grained search with CV within the optimal region
SNR:    Signal-to-noise ratio sensitivity analysis
Benchmark: Compare AdaptiveFlippedLasso vs other models' CV versions

Usage:
    # Stage 1: Coarse grid search
    python factory/sweep.py stage1 --config configs/stage1/example.yaml

    # Stage 2: Fine CV search within Stage1's optimal region
    python factory/sweep.py stage2 --config configs/stage2/example_cv.yaml

    # SNR study: Analyze performance across different noise levels
    python factory/sweep.py snr --config configs/snr/example_snr.yaml

    # Benchmark: Compare AdaptiveFlippedLasso (optimal params) vs other models' CV versions
    python factory/sweep.py benchmark --config configs/benchmark/example_benchmark.yaml
"""

import argparse
import inspect
import json
import sys
import traceback
from datetime import datetime
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.factory.run import run_experiment, load_config, generate_experiment_dir, save_config


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage hyperparameter tuning")
    parser.add_argument(
        "stage",
        choices=["stage1", "stage2", "snr", "benchmark"],
        help="Stage to run: stage1 (grid search), stage2 (fine CV tuning), snr (SNR sensitivity), or benchmark (model comparison)",
    )
    parser.add_argument(
        "--config", "-c", required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running",
    )
    return parser.parse_args()


def grid_search(config):
    """
    Stage 1: Grid search over hyperparameter space.
    Identifies the optimal structural region.
    """
    print("[sweep:stage1] Starting Stage 1: Grid Search")
    print(f"[sweep:stage1] Config: {config['experiment']}")

    search_space = config.get("search_space", {})
    if not search_space:
        raise ValueError("Stage1 requires search_space in config")

    # Extract parameter grids
    param_names = list(search_space.keys())
    param_values = [search_space[p] if isinstance(search_space[p], list) else [search_space[p]] for p in param_names]

    # Generate all grid combinations
    grid_combinations = list(product(*param_values))
    total_combinations = len(grid_combinations)

    print(f"[sweep:stage1] Grid size: {total_combinations}")
    print(f"[sweep:stage1] Parameters: {param_names}")

    # Generate experiment directory
    config["output_dir"] = config.get("output_dir", "/home/lili/lyn/clear/NLasso/XLasso/experiments/results/stage1")
    exp_dir = generate_experiment_dir(config)
    save_config(config, exp_dir)

    all_results = []
    best_metrics = None
    best_idx = None

    for idx, combo in enumerate(grid_combinations):
        param_dict = dict(zip(param_names, combo))
        print(f"\n[sweep:stage1] [{idx+1}/{total_combinations}] Testing: {param_dict}")

        # Create temporary config with this combination
        trial_config = {**config, **param_dict}

        try:
            # Use internal run since we need to pass config directly
            trial_config["experiment"] = f"{config['experiment']}_grid_{idx}"
            result = run_trial(trial_config, exp_dir)

            if result and result.get("n_folds_completed", 0) > 0:
                # Check if we have individual repeat results
                individual = result.get("individual_metrics", [])
                n_repeats_completed = result.get("n_repeats_completed", 1)

                if len(individual) > 1:
                    # Multiple repeats: save each repeat as separate row for stability analysis
                    for repeat_idx, repeat_metrics in enumerate(individual):
                        result_entry = {"idx": idx, "repeat": repeat_idx}
                        for k, v in param_dict.items():
                            result_entry[f"params_{k}"] = v
                        result_entry.update(repeat_metrics)
                        all_results.append(result_entry)
                else:
                    # Single repeat: use standard path
                    result_entry = {"idx": idx}
                    for k, v in param_dict.items():
                        result_entry[f"params_{k}"] = v
                    result_entry.update(result["metrics"])
                    all_results.append(result_entry)

                # Track best (use mean metrics)
                current_f1 = result["metrics"].get("f1", 0)
                if best_metrics is None or current_f1 > best_metrics.get("f1", 0):
                    best_metrics = result["metrics"]
                    best_idx = idx
                    print(f"[sweep:stage1] *** New best F1: {current_f1:.4f}")

        except Exception as e:
            print(f"[sweep:stage1] ERROR in trial {idx}: {e}")
            traceback.print_exc()
            continue

    if not all_results:
        raise RuntimeError("No successful trials in Stage 1")

    # Save raw results
    results_df = pd.DataFrame(all_results)
    raw_path = exp_dir / "raw.csv"
    results_df.to_csv(raw_path, index=False)

    # Analyze optimal region
    optimal_region = analyze_optimal_region(results_df, param_names)

    # Generate best config
    best_config_path = generate_best_config_stage1(
        config, exp_dir, param_names, results_df, optimal_region
    )

    print(f"\n[sweep:stage1] Completed!")
    print(f"[sweep:stage1] Best config: {best_config_path}")
    print(f"[sweep:stage1] Optimal region: {optimal_region}")

    return {
        "status": "completed",
        "exp_dir": str(exp_dir),
        "best_config": str(best_config_path),
        "optimal_region": optimal_region,
        "n_trials": len(all_results),
    }


def analyze_optimal_region(results_df, param_names):
    """
    Analyze grid search results to identify optimal structural region.
    Identifies the hypercube (interval for each parameter) containing
    top-performing configurations.
    """
    # Sort by F1 score (or primary metric)
    sorted_df = results_df.sort_values("f1", ascending=False)

    # Take top 20% of configurations
    n_top = max(1, len(sorted_df) // 5)
    top_configs = sorted_df.head(n_top)

    optimal_region = {}

    for param in param_names:
        param_vals = top_configs[f"params_{param}"]
        if len(param_vals) > 0:
            optimal_region[param] = [float(param_vals.min()), float(param_vals.max())]

    # Generate insights
    insights = generate_insights(results_df, sorted_df.head(n_top), param_names)

    return {
        "region": optimal_region,
        "insights": insights,
        "top_n_configs": n_top,
    }


def generate_insights(results_df, top_configs, param_names):
    """Generate human-readable insights from grid search."""
    insights = []

    # F1 correlation with each parameter
    for param in param_names:
        col = f"params_{param}"
        if col in results_df.columns:
            corr = results_df[col].corr(results_df["f1"])
            if abs(corr) > 0.3:
                direction = "positively" if corr > 0 else "negatively"
                insights.append(f"{param} is {direction} correlated with F1 (r={corr:.3f})")

    # Best parameter range
    for param in param_names:
        col = f"params_{param}"
        if col in top_configs.columns:
            best_range = f"[{top_configs[col].min():.6f}, {top_configs[col].max():.6f}]"
            insights.append(f"Best {param} range: {best_range}")

    return insights


def generate_best_config_stage1(config, exp_dir, param_names, results_df, optimal_region):
    """Generate _best.yaml config with optimal structural region."""
    best_row = results_df.loc[results_df["f1"].idxmax()]

    best_config = {
        "experiment": config["experiment"],
        "generated_by": "sweep.py stage1",
        "generated_at": datetime.now().isoformat(),
        "base_config": str(config.get("config_path", "")),
        "search_space": config.get("search_space", {}),
        "best_structural_region": optimal_region["region"],
        "structural_insight": optimal_region.get("insights", []),
        "best_grid_point": {p: best_row[f"params_{p}"] for p in param_names},
        "best_metrics": {
            "f1": float(best_row.get("f1", 0)),
            "mse": float(best_row.get("mse", float("inf"))),
            "tpr": float(best_row.get("tpr", 0)),
            "fdr": float(best_row.get("fdr", float("inf"))),
        },
    }

    best_path = Path("configs/stage1") / f"{config['experiment']}_best.yaml"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    with open(best_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)

    return best_path


def run_trial(config, parent_dir):
    """Run a single trial with given config."""
    from experiments.factory.run import (
        run_single_fold,
    )
    from experiments.modules import (
        ALGO_REGISTRY,
        MetricCalculator,
        CrossValidator,
        DataGenerator,
    )

    n_folds = config.get("cv_folds", 5)
    n_repeats = config.get("n_repeats", 1)

    metrics_list = []

    for repeat in range(n_repeats):
        # Use random_state_base if provided, otherwise default to 42
        random_state_base = config.get("random_state_base", 42)
        random_state = random_state_base + repeat

        # Generate data
        data_gen = DataGenerator(random_state=random_state)
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

        # Get algorithm
        algo_name = config["algo"].lower()
        algo_class = ALGO_REGISTRY.get(algo_name)
        if algo_class is None:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        # Single split (no CV): use train_test_split equivalent
        if n_folds == 1:
            from sklearn.model_selection import train_test_split
            train_idx, test_idx = train_test_split(
                np.arange(len(y)),
                test_size=0.2,
                random_state=random_state,
                shuffle=True
            )
            fold_metrics, coefs = run_single_fold(
                algo_class, config, X, y, beta_true, repeat, train_idx, test_idx
            )
            metrics_list.append(fold_metrics)
        else:
            # CV
            cv = CrossValidator(n_folds=n_folds, shuffle=True, random_state=42 + repeat)

            repeat_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
                fold_metrics, coefs = run_single_fold(
                    algo_class, config, X, y, beta_true, fold_idx, train_idx, test_idx
                )
                repeat_metrics.append(fold_metrics)

            # Average across folds
            metrics_df = pd.DataFrame(repeat_metrics)
            avg_metrics = metrics_df.mean().to_dict()
            metrics_list.append(avg_metrics)

    # Average across repeats
    final_metrics = pd.DataFrame(metrics_list).mean().to_dict()

    return {
        "n_folds_completed": n_folds,
        "metrics": final_metrics,
        "individual_metrics": metrics_list,
        "n_repeats_completed": n_repeats,
    }


def cv_fine_tuning(config):
    """
    Stage 2: Fine-grained CV tuning within Stage1's optimal region.
    """
    print("[sweep:stage2] Starting Stage 2: Fine CV Tuning")
    print(f"[sweep:stage2] Config: {config['experiment']}")

    # Load Stage1 best config if reference provided
    base_config_path = config.get("base_config")
    if base_config_path and Path(base_config_path).exists():
        with open(base_config_path, "r") as f:
            base_config = yaml.safe_load(f)
        print(f"[sweep:stage2] Loaded base config: {base_config_path}")

        # Get optimal region from Stage1
        structural_region = base_config.get("best_structural_region", {})
        if not structural_region:
            print("[sweep:stage2] WARNING: No best_structural_region in base config")
            structural_region = config.get("best_structural_region", {})
    else:
        structural_region = config.get("best_structural_region", {})

    if not structural_region:
        raise ValueError("Stage2 requires best_structural_region from Stage1")

    # Generate fine search space from region
    fine_space = config.get("fine_search_space", {})
    if not fine_space:
        print("[sweep:stage2] Generating fine search space from region...")
        fine_space = generate_fine_space(structural_region)

    print(f"[sweep:stage2] Fine search space: {fine_space}")

    # Generate experiment directory
    config["output_dir"] = config.get("output_dir", "/home/lili/lyn/clear/NLasso/XLasso/experiments/results/stage2")
    exp_dir = generate_experiment_dir(config)
    save_config(config, exp_dir)

    # Grid search with fine space
    param_names = list(fine_space.keys())
    param_values = [fine_space[p] if isinstance(fine_space[p], list) else [fine_space[p]] for p in param_names]
    grid_combinations = list(product(*param_values))
    total_combinations = len(grid_combinations)

    print(f"[sweep:stage2] Fine grid size: {total_combinations}")

    all_results = []
    rankings = {}

    for idx, combo in enumerate(grid_combinations):
        param_dict = dict(zip(param_names, combo))
        print(f"\n[sweep:stage2] [{idx+1}/{total_combinations}] Testing: {param_dict}")

        trial_config = {**config, **param_dict}
        trial_config["experiment"] = f"{config['experiment']}_cv_{idx}"

        try:
            result = run_trial(trial_config, exp_dir)

            if result and result.get("n_folds_completed", 0) > 0:
                # Check if we have individual repeat results
                individual = result.get("individual_metrics", [])

                if len(individual) > 1:
                    # Multiple repeats: save each repeat as separate row
                    for repeat_idx, repeat_metrics in enumerate(individual):
                        result_entry = {"idx": idx, "repeat": repeat_idx, "params": param_dict}
                        result_entry.update(repeat_metrics)
                        all_results.append(result_entry)
                else:
                    # Single repeat: use standard path
                    metrics = result["metrics"]
                    all_results.append({
                        "idx": idx,
                        "params": param_dict,
                        **metrics,
                    })

                # Track best
                current_f1 = metrics.get("f1", 0)
                if "f1" not in rankings or current_f1 > rankings["f1"].get("f1", 0):
                    rankings["f1"] = {
                        "rank_1": {
                            "params": param_dict,
                            "f1": current_f1,
                            "metrics": metrics,
                        }
                    }

        except Exception as e:
            print(f"[sweep:stage2] ERROR in trial {idx}: {e}")
            traceback.print_exc()
            continue

    if not all_results:
        raise RuntimeError("No successful trials in Stage 2")

    # Save results
    results_df = pd.DataFrame(all_results)
    raw_path = exp_dir / "raw.csv"
    results_df.to_csv(raw_path, index=False)

    # Generate CV best config
    best_config_path = generate_best_config_stage2(
        config, exp_dir, all_results, rankings
    )

    print(f"\n[sweep:stage2] Completed!")
    print(f"[sweep:stage2] Best config: {best_config_path}")

    return {
        "status": "completed",
        "exp_dir": str(exp_dir),
        "best_config": str(best_config_path),
        "n_trials": len(all_results),
    }


def generate_fine_space(structural_region, n_points=5):
    """Generate fine-grained search space from optimal region."""
    fine_space = {}

    for param, (low, high) in structural_region.items():
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            # Generate evenly spaced points
            fine_space[param] = np.linspace(low, high, n_points).tolist()
        else:
            fine_space[param] = [low, high]

    return fine_space


def generate_best_config_stage2(config, exp_dir, all_results, rankings):
    """Generate _cv_best.yaml with optimal params, metrics, and rankings."""
    results_df = pd.DataFrame(all_results)
    best_row = results_df.loc[results_df["f1"].idxmax()]

    best_config = {
        "experiment": config["experiment"],
        "generated_by": "sweep.py stage2",
        "generated_at": datetime.now().isoformat(),
        "best_config_ref": config.get("base_config", ""),
        "optimal_params": {p: float(best_row[f"params_{p}"]) for p in all_results[0]["params"].keys()},
        "best_metrics": {
            "f1": float(best_row.get("f1", 0)),
            "mse": float(best_row.get("mse", float("inf"))),
            "r2": float(best_row.get("r2", 0)),
            "tpr": float(best_row.get("tpr", 0)),
            "fdr": float(best_row.get("fdr", float("inf"))),
            "precision": float(best_row.get("precision", 0)),
            "recall": float(best_row.get("recall", 0)),
            "sparsity": float(best_row.get("sparsity", 0)),
        },
        "rankings": {
            "by_f1": rankings.get("f1", {}),
        },
    }

    best_path = Path("configs/stage2") / f"{config['experiment']}_cv_best.yaml"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    with open(best_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)

    return best_path


def run_stage1(config_path, output_dir=None, dry_run=False):
    """Run Stage 1."""
    config = load_config(config_path)
    if output_dir:
        config["output_dir"] = output_dir
    config["config_path"] = config_path

    return grid_search(config)


def run_stage2(config_path, output_dir=None, dry_run=False):
    """Run Stage 2."""
    config = load_config(config_path)
    if output_dir:
        config["output_dir"] = output_dir

    return cv_fine_tuning(config)


def snr_study(config):
    """
    SNR Sensitivity Analysis: Study performance across different noise levels.
    Uses fixed best parameters from stage1 and varies sigma to analyze SNR effects.
    """
    print("[sweep:snr] Starting SNR Sensitivity Analysis")
    print(f"[sweep:snr] Config: {config['experiment']}")

    search_space = config.get("search_space", {})
    if "sigma" not in search_space:
        raise ValueError("SNR study requires 'sigma' in search_space")

    # Extract sigma values
    sigma_values = search_space["sigma"] if isinstance(search_space["sigma"], list) else [search_space["sigma"]]
    print(f"[sweep:snr] SNR values: {sigma_values}")

    # Fixed parameters (should be in config)
    fixed_params = {
        "lambda_ridge": config.get("lambda_ridge", 10.0),
        "lambda_": config.get("lambda_", 1.0),
        "gamma": config.get("gamma", 0.5),
    }
    print(f"[sweep:snr] Fixed params: {fixed_params}")

    # Generate experiment directory
    config["output_dir"] = config.get("output_dir", "/home/lili/lyn/clear/NLasso/XLasso/experiments/results/snr")
    exp_dir = generate_experiment_dir(config)
    save_config(config, exp_dir)

    all_results = []
    n_repeats = config.get("n_repeats", 5)

    for sigma in sigma_values:
        print(f"\n[sweep:snr] Testing sigma={sigma} (SNR={1.0/sigma:.2f})...")

        for repeat in range(n_repeats):
            trial_config = {
                **config,
                **fixed_params,
                "sigma": sigma,
                "n_repeats": 1,  # run_trial handles repeats internally
                "random_state_base": repeat,  # Each call gets different data
            }
            trial_config["experiment"] = f"{config['experiment']}_sigma{sigma}_r{repeat}"

            try:
                result = run_trial(trial_config, exp_dir)

                if result and result.get("n_folds_completed", 0) > 0:
                    individual = result.get("individual_metrics", [])
                    for repeat_idx, repeat_metrics in enumerate(individual):
                        result_entry = {
                            "sigma": sigma,
                            "snr": 1.0 / sigma,
                            "repeat": repeat,
                            **fixed_params,
                            **repeat_metrics,
                        }
                        all_results.append(result_entry)

            except Exception as e:
                print(f"[sweep:snr] ERROR at sigma={sigma}, repeat={repeat}: {e}")
                traceback.print_exc()
                continue

    if not all_results:
        raise RuntimeError("No successful trials in SNR study")

    # Save raw results
    results_df = pd.DataFrame(all_results)
    raw_path = exp_dir / "raw.csv"
    results_df.to_csv(raw_path, index=False)

    # Generate summary statistics
    summary = generate_snr_summary(results_df, exp_dir)

    # Generate sensitivity analysis
    sensitivity = analyze_snr_sensitivity(results_df)

    # Generate best SNR config
    best_config_path = generate_best_config_snr(config, exp_dir, results_df, summary, sensitivity)

    print(f"\n[sweep:snr] Completed!")
    print(f"[sweep:snr] Results saved to {exp_dir}")
    print(f"[sweep:snr] Best config: {best_config_path}")
    print(f"\n[sweep:snr] Summary:")
    print(summary.to_string())

    return {
        "status": "completed",
        "exp_dir": str(exp_dir),
        "best_config": str(best_config_path),
        "n_trials": len(all_results),
    }


def generate_snr_summary(results_df, exp_dir):
    """Generate summary statistics for SNR study."""
    summary = results_df.groupby('sigma').agg({
        'f1': ['mean', 'std'],
        'mse': ['mean', 'std'],
        'tpr': ['mean', 'std'],
        'fdr': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'r2': ['mean', 'std'],
    }).round(4)

    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary['snr'] = (1.0 / summary['sigma']).round(2)

    summary_path = exp_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    return summary


def analyze_snr_sensitivity(results_df):
    """Analyze sensitivity of metrics to sigma changes."""
    sensitivity = {}

    metrics = ['f1', 'mse', 'tpr', 'fdr', 'precision', 'recall', 'r2']
    for metric in metrics:
        if metric in results_df.columns:
            corr = results_df['sigma'].corr(results_df[metric])
            sensitivity[f"{metric}_corr"] = round(corr, 4)

    return sensitivity


def generate_best_config_snr(config, exp_dir, results_df, summary, sensitivity):
    """Generate best config for SNR study."""
    # Find best sigma (highest mean F1)
    best_sigma = summary.loc[summary['f1_mean'].idxmax(), 'sigma']

    best_config = {
        "experiment": config["experiment"],
        "generated_by": "sweep.py snr",
        "generated_at": datetime.now().isoformat(),
        "base_config": str(config.get("config_path", "")),
        "search_space": config.get("search_space", {}),
        "fixed_params": {
            "lambda_ridge": config.get("lambda_ridge", 10.0),
            "lambda_": config.get("lambda_", 1.0),
            "gamma": config.get("gamma", 0.5),
        },
        "best_snr_region": {
            "sigma": [summary['sigma'].min(), summary['sigma'].max()],
            "snr": [1.0/summary['sigma'].max(), 1.0/summary['sigma'].min()],
        },
        "sensitivity_analysis": sensitivity,
        "summary": summary.to_dict('records'),
    }

    best_path = Path("configs/snr") / f"{config['experiment']}_best.yaml"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    with open(best_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)

    return best_path


def run_snr(config_path, output_dir=None, dry_run=False):
    """Run SNR study."""
    config = load_config(config_path)
    if output_dir:
        config["output_dir"] = output_dir
    config["config_path"] = config_path

    return snr_study(config)


def benchmark_study(config):
    """
    Benchmark: Compare AdaptiveFlippedLasso (optimal params) vs other models' CV versions.
    Uses exp2 data configuration (AR(1), rho=0.8) across different SNR values.

    Fair comparison: All models use the SAME data and SAME CV splits for each (sigma, repeat).
    - Data generation: outside model loop, same seed = same data for all models
    - CV splits: pre-generated and shared across all models
    - Seeds: 42 + repeat for reproducibility
    """
    from sklearn.model_selection import KFold
    from experiments.modules import DataGenerator

    print("[sweep:benchmark] Starting Fair Benchmark Comparison")
    print(f"[sweep:benchmark] Config: {config['experiment']}")

    search_space = config.get("search_space", {})
    if "sigma" not in search_space:
        raise ValueError("Benchmark requires 'sigma' in search_space")

    sigma_values = search_space["sigma"] if isinstance(search_space["sigma"], list) else [search_space["sigma"]]
    print(f"[sweep:benchmark] SNR values: {sigma_values}")

    # AdaptiveFlippedLasso optimal params from stage1
    afl_params = {
        "lambda_ridge": config.get("lambda_ridge", 1.0),
        "lambda_": config.get("lambda_", 1.0),
        "gamma": config.get("gamma", 0.5),
    }
    print(f"[sweep:benchmark] AdaptiveFlippedLasso params: {afl_params}")

    # Models to compare (algorithm name -> display name, use_cv flag)
    models_to_compare = config.get("compare_models", [
        {"algo": "adaptive_flipped_lasso", "display": "AdaptiveFlippedLasso", "params": afl_params, "cv": False},
        {"algo": "lasso_cv", "display": "LassoCV", "params": {}, "cv": True},
        {"algo": "nlasso_cv", "display": "NLassoCV", "params": {}, "cv": True},
        {"algo": "adaptive_lasso_cv", "display": "AdaptiveLassoCV", "params": {}, "cv": True},
        {"algo": "group_lasso_cv", "display": "GroupLassoCV", "params": {}, "cv": True},
        {"algo": "unilasso_cv", "display": "UniLassoCV", "params": {}, "cv": True},
    ])
    print(f"[sweep:benchmark] Comparing {len(models_to_compare)} models")

    # Generate experiment directory
    config["output_dir"] = config.get("output_dir", "/home/lili/lyn/clear/NLasso/XLasso/experiments/results/benchmark")
    exp_dir = generate_experiment_dir(config)
    save_config(config, exp_dir)

    all_results = []
    n_repeats = config.get("n_repeats", 5)
    n_folds = config.get("cv_folds", 5)

    for sigma in sigma_values:
        print(f"\n[sweep:benchmark] Testing sigma={sigma} (SNR={1.0/sigma:.2f})...")

        for repeat in range(n_repeats):
            seed = 42 + repeat  # 保证可重复性，种子递增

            # ============================================================
            # 数据生成：每个 (sigma, repeat) 只生成一次
            # ============================================================
            data_gen = DataGenerator(random_state=seed)
            X, y, beta_true = data_gen.generate(
                n_samples=config["n_samples"],
                n_features=config["n_features"],
                n_nonzero=config["n_nonzero"],
                sigma=sigma,  # 传入 sigma，不同 SNR 生成不同数据
                correlation_type=config.get("correlation_type", "pairwise"),
                rho=config.get("rho", 0.5),
                block_size=config.get("block_size", 10),
                n_blocks=config.get("n_blocks", 50),
            )

            # ============================================================
            # CV 分割：每个 (sigma, repeat) 只生成一次，转为列表可复用
            # ============================================================
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits = list(kfold.split(X))  # 物化为列表，避免一次性生成器问题

            for model_info in models_to_compare:
                algo_name = model_info["algo"]
                display_name = model_info["display"]
                use_cv = model_info.get("cv", False)
                model_params = model_info.get("params", {})

                trial_config = {
                    **config,
                    **model_params,
                    "algo": algo_name,
                    "sigma": sigma,
                    "n_repeats": 1,
                    "random_state_base": seed,
                    "cv_folds": n_folds if use_cv else 1,
                }
                trial_config["experiment"] = f"{config['experiment']}_sigma{sigma}_{algo_name}_r{repeat}"

                try:
                    # 传入预生成的数据和分割
                    result = run_benchmark_trial_with_splits(
                        trial_config, exp_dir, X=X, y=y, beta_true=beta_true, splits=splits
                    )

                    if result and result.get("n_folds_completed", 0) > 0:
                        individual = result.get("individual_metrics", [])
                        for repeat_idx, repeat_metrics in enumerate(individual):
                            result_entry = {
                                "sigma": sigma,
                                "snr": round(1.0 / sigma, 2),
                                "repeat": repeat,
                                "model": display_name,
                                "algo": algo_name,
                                "use_cv": use_cv,
                                **repeat_metrics,
                            }
                            all_results.append(result_entry)

                except Exception as e:
                    print(f"  [sweep:benchmark] ERROR at sigma={sigma}, model={display_name}, repeat={repeat}: {e}")
                    traceback.print_exc()
                    continue

    if not all_results:
        raise RuntimeError("No successful trials in benchmark")

    # Save raw results
    results_df = pd.DataFrame(all_results)
    raw_path = exp_dir / "raw.csv"
    results_df.to_csv(raw_path, index=False)

    # Generate summary statistics by model and sigma
    summary = generate_benchmark_summary(results_df, exp_dir)

    # Generate rankings
    rankings = generate_benchmark_rankings(results_df, exp_dir)

    # Generate report
    report_path = generate_benchmark_report(config, exp_dir, results_df, summary, rankings)

    print(f"\n[sweep:benchmark] Completed!")
    print(f"[sweep:benchmark] Results saved to {exp_dir}")
    print(f"[sweep:benchmark] Report: {report_path}")

    return {
        "status": "completed",
        "exp_dir": str(exp_dir),
        "report": str(report_path),
        "n_trials": len(all_results),
    }


def run_benchmark_trial_with_splits(config, parent_dir, X, y, beta_true, splits):
    """
    Run a single benchmark trial with the new CV strategy:

    1. Split data: 80% train, 20% validation (hold-out)
    2. On 80% train: generate K-fold CV splits for hyperparameter selection
    3. Algorithm uses K-fold CV to select hyperparameters, trains on full 80% train
    4. Evaluate on 20% validation (hold-out)

    All algorithms use the SAME train/val split and SAME K-fold CV splits for fair comparison.
    """
    from experiments.factory.run import ALGO_REGISTRY
    from experiments.modules import MetricCalculator
    from sklearn.model_selection import KFold, train_test_split
    import time

    n_folds = config.get("cv_folds", 10)  # K-fold for internal CV
    random_state = config.get("random_state_base", 42)

    # Use correct params for each algo
    algo_name = config["algo"].lower()
    algo_params = get_benchmark_algo_params(algo_name, config)

    # Get algorithm
    algo_class = ALGO_REGISTRY.get(algo_name)
    if algo_class is None:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    metrics_list = []

    # ============================================================
    # Step 1: 80/20 split - 80% train, 20% validation (hold-out)
    # ============================================================
    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        random_state=random_state,
        shuffle=True
    )
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # ============================================================
    # Step 2: Generate K-fold CV splits on 80% train
    # ============================================================
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_splits = list(kfold.split(X_train))  # List of (train_idx, val_idx) within X_train

    # ============================================================
    # Step 3: Algorithm uses K-fold CV for hyperparameter selection
    # ============================================================
    # Check if algorithm supports cv_splits parameter
    algo = algo_class(**algo_params)
    supports_cv_splits = 'cv_splits' in inspect.signature(algo.fit).parameters

    start_time = time.time()
    if supports_cv_splits:
        algo.fit(X_train, y_train, cv_splits=cv_splits)
    else:
        # Fallback: algorithm creates its own CV splits
        algo.fit(X_train, y_train)
    train_time = time.time() - start_time

    # ============================================================
    # Step 4: Evaluate on 20% validation (hold-out)
    # ============================================================
    y_pred = algo.predict(X_val)

    metrics_calc = MetricCalculator()
    fold_metrics = metrics_calc.calculate(
        y_true=y_val,
        y_pred=y_pred,
        beta_true=beta_true,
        beta_est=algo.coef_,
    )
    fold_metrics["fold"] = 0
    fold_metrics["train_time"] = train_time
    if hasattr(algo, 'best_gamma_'):
        fold_metrics["best_gamma"] = algo.best_gamma_
    if hasattr(algo, 'best_alpha_'):
        fold_metrics["best_alpha"] = algo.best_alpha_
    if hasattr(algo, 'best_lambda_ridge_'):
        fold_metrics["best_lambda_ridge"] = algo.best_lambda_ridge_
    if hasattr(algo, 'cv_score_'):
        fold_metrics["cv_score"] = algo.cv_score_
    metrics_list.append(fold_metrics)

    return {
        "n_folds_completed": len(metrics_list),
        "individual_metrics": metrics_list,
    }


def get_benchmark_algo_params(algo_name, config):
    """Get algorithm-specific parameters for benchmark - handles CV variants correctly."""
    from experiments.modules import (
        NLasso,
        NLassoClassifier,
        NLassoCV,
        NLassoClassifierCV,
        AdaptiveFlippedLasso,
        AdaptiveFlippedLassoClassifier,
        AdaptiveFlippedLassoCV,
        AdaptiveLasso,
        AdaptiveLassoCV,
        FusedLasso,
        FusedLassoCV,
        GroupLasso,
        GroupLassoCV,
        AdaptiveSparseGroupLasso,
        AdaptiveSparseGroupLassoCV,
        Lasso,
        LassoCV,
        UniLasso,
        UniLassoCV,
    )

    # Common parameters
    params = {
        "standardize": True,
        "fit_intercept": True,
    }

    if algo_name == "nlasso":
        params.update({
            "lambda_ridge": config.get("lambda_ridge", 10.0),
            "lambda_": config.get("lambda_", 0.01),
            "gamma": config.get("gamma", 0.3),
            "s": config.get("s", 1.0),
            "group_threshold": config.get("group_threshold", 0.7),
            "group_min_size": config.get("group_min_size", 2),
            "group_max_size": config.get("group_max_size", 10),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name == "nlasso_cv":
        params.update({
            "cv": config.get("cv_folds", 5),
            "param_grid": {
                'lambda_ridge': [1.0, 10.0],
                'gamma': [0.3, 0.5],
                's': [0.5, 1.0],
                'group_threshold': [0.7],
            },
            "random_state": config.get("random_state", 42),
        })
    elif algo_name == "adaptive_flipped_lasso":
        params.update({
            "lambda_ridge": config.get("lambda_ridge", 10.0),
            "lambda_": config.get("lambda_", 0.01),
            "gamma": config.get("gamma", 1.0),
            "alpha_min_ratio": config.get("alpha_min_ratio", 1e-4),
            "n_alpha": config.get("n_alpha", 50),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name == "aflclassifier":
        # Same as adaptive_flipped_lasso
        params.update({
            "lambda_ridge": config.get("lambda_ridge", 10.0),
            "lambda_": config.get("lambda_", 0.01),
            "gamma": config.get("gamma", 1.0),
            "alpha_min_ratio": config.get("alpha_min_ratio", 1e-4),
            "n_alpha": config.get("n_alpha", 50),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name == "aflclassifier_cv":
        params.update({
            "lambda_ridge_list": config.get("lambda_ridge_list", [0.1, 1.0, 10.0]),
            "gamma_list": config.get("gamma_list", [0.5, 1.0, 2.0]),
            "cv": config.get("cv_folds", 5),
            "alpha_min_ratio": config.get("alpha_min_ratio", 1e-4),
            "n_alpha": config.get("n_alpha", 50),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name == "adaptive_flipped_lasso_cv":
        params.update({
            "lambda_ridge_list": config.get("lambda_ridge_list", [0.1, 1.0, 10.0]),
            "gamma_list": config.get("gamma_list", [0.5, 1.0, 2.0]),
            "cv": config.get("cv_folds", 5),
            "alpha_min_ratio": config.get("alpha_min_ratio", 1e-4),
            "n_alpha": config.get("n_alpha", 50),
            "max_iter": config.get("max_iter", 2000),
            "tol": config.get("tol", 0.0001),
            "standardize": config.get("standardize", True),
            "fit_intercept": config.get("fit_intercept", True),
            "random_state": config.get("random_state", 42),
            "verbose": config.get("verbose", False),
            "use_post_ols_debiasing": config.get("use_post_ols_debiasing", False),
            "auto_tune_collinearity": config.get("auto_tune_collinearity", True),
            "weight_clip_max": config.get("weight_clip_max", 100.0),
        })
    elif algo_name == "adaptive_flipped_lasso_ebic":
        params.update({
            "lambda_ridge_list": config.get("lambda_ridge_list", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
            "gamma_list": config.get("gamma_list", [0.3, 0.5, 0.7, 1.0]),
            "ebic_gamma": config.get("ebic_gamma", 0.5),
            "alpha_min_ratio": config.get("alpha_min_ratio", 1e-4),
            "n_alpha": config.get("n_alpha", 100),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name == "aflclassifier_ebic":
        params.update({
            "lambda_ridge_list": config.get("lambda_ridge_list", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
            "gamma_list": config.get("gamma_list", [0.3, 0.5, 0.7, 1.0]),
            "ebic_gamma": config.get("ebic_gamma", 0.5),
            "alpha_min_ratio": config.get("alpha_min_ratio", 1e-4),
            "n_alpha": config.get("n_alpha", 100),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name == "adaptive_lasso":
        params.update({
            "alpha": config.get("lambda_", 0.01),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name == "adaptive_lasso_cv":
        params.update({
            "alphas": np.logspace(-4, 1, 30),
            "gammas": [0.5, 1.0, 2.0],
            "cv": config.get("cv_folds", 5),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
            "use_1se": True,  # Enable 1-SE rule
        })
    elif algo_name == "lasso":
        params.update({
            "alpha": config.get("lambda_", 0.01),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name == "lasso_cv":
        params.update({
            "alphas": np.logspace(-4, 1, 30),
            "cv": config.get("cv_folds", 5),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
            "random_state": config.get("random_state", 42),
            "use_1se": True,  # Enable 1-SE rule
        })
    elif algo_name == "fused_lasso":
        params.update({
            "alpha": config.get("lambda_", 0.01),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name == "group_lasso":
        params.update({
            "alpha": config.get("lambda_", 0.01),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name == "group_lasso_cv":
        params.update({
            "alphas": np.logspace(-4, 1, 30),
            "cv": config.get("cv_folds", 5),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name == "unilasso":
        params.update({
            "lambda_1": config.get("lambda_1", 0.01),
            "lambda_2": config.get("lambda_2", 0.01),
            "group_threshold": config.get("group_threshold", 0.7),
            "standardize": config.get("standardize", True),
            "fit_intercept": config.get("fit_intercept", True),
            "family": config.get("family", "gaussian"),
        })
    elif algo_name == "unilasso_cv":
        params.update({
            "lambda_1": config.get("lambda_1", 0.01),
            "lambda_2": config.get("lambda_2", 0.01),
            "standardize": config.get("standardize", True),
            "fit_intercept": config.get("fit_intercept", True),
            "family": config.get("family", "gaussian"),
            "n_folds": config.get("cv_folds", 5),
            "use_1se": True,  # Enable 1-SE rule
        })
    elif algo_name == "fused_lasso_cv":
        params.update({
            "alphas": np.logspace(-4, 1, 30),
            "cv": config.get("cv_folds", 5),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name == "elasticnet_1se":
        # ElasticNet1SE has its own specific parameters
        params.update({
            "cv_folds": config.get("cv_folds", 5),
            "l1_ratios": [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
            "max_iter": config.get("max_iter", 5000),
            "random_state": config.get("random_state", 42),
            "verbose": False,
        })
    elif algo_name == "relaxed_lasso_1se":
        # RelaxedLassoCV1SE has its own specific parameters
        params.update({
            "cv": config.get("cv_folds", 5),
            "random_state": config.get("random_state", 42),
            "eps": 1e-3,
            "n_alphas": 100,
            "verbose": False,
        })
    else:
        params.update({
            "alpha": config.get("lambda_", 0.01),
        })

    return params


def run_benchmark_trial(config, parent_dir):
    """Run a single benchmark trial with correct params for each algorithm."""
    from experiments.factory.run import ALGO_REGISTRY
    from experiments.modules import (
        MetricCalculator,
        CrossValidator,
        DataGenerator,
    )
    import time

    n_folds = config.get("cv_folds", 5)
    n_repeats = config.get("n_repeats", 1)

    # Use correct params for each algo
    algo_name = config["algo"].lower()
    algo_params = get_benchmark_algo_params(algo_name, config)

    metrics_list = []

    for repeat in range(n_repeats):
        random_state_base = config.get("random_state_base", 42)
        random_state = random_state_base + repeat

        # Generate data
        data_gen = DataGenerator(random_state=random_state)
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

        # Get algorithm
        algo_class = ALGO_REGISTRY.get(algo_name)
        if algo_class is None:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        # Initialize algorithm with correct params
        algo = algo_class(**algo_params)

        if n_folds == 1:
            # Single split (no CV): use train_test_split equivalent
            from sklearn.model_selection import train_test_split
            train_idx, test_idx = train_test_split(
                np.arange(len(y)),
                test_size=0.2,
                random_state=random_state,
                shuffle=True
            )
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit
            start_time = time.time()
            algo.fit(X_train, y_train)
            train_time = time.time() - start_time

            # Predict
            y_pred = algo.predict(X_test)

            # Calculate metrics
            metrics_calc = MetricCalculator()
            fold_metrics = metrics_calc.calculate(
                y_true=y_test,
                y_pred=y_pred,
                beta_true=beta_true,
                beta_est=algo.coef_,
            )
            fold_metrics["fold"] = repeat
            fold_metrics["train_time"] = train_time
            # Capture CV parameters if available (AFL CV with internal CV)
            if hasattr(algo, 'best_gamma_'):
                fold_metrics["best_gamma"] = algo.best_gamma_
            if hasattr(algo, 'best_alpha_'):
                fold_metrics["best_alpha"] = algo.best_alpha_
            if hasattr(algo, 'best_lambda_ridge_'):
                fold_metrics["best_lambda_ridge"] = algo.best_lambda_ridge_
            if hasattr(algo, 'cv_score_'):
                fold_metrics["cv_score"] = algo.cv_score_
            metrics_list.append(fold_metrics)
        else:
            # CV
            cv = CrossValidator(n_folds=n_folds, shuffle=True, random_state=42 + repeat)
            repeat_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Initialize fresh algo instance for each fold
                algo_fold = algo_class(**algo_params)

                # Fit
                start_time = time.time()
                algo_fold.fit(X_train, y_train)
                train_time = time.time() - start_time

                # Predict
                y_pred = algo_fold.predict(X_test)

                # Calculate metrics
                metrics_calc = MetricCalculator()
                fold_metrics = metrics_calc.calculate(
                    y_true=y_test,
                    y_pred=y_pred,
                    beta_true=beta_true,
                    beta_est=algo_fold.coef_,
                )
                fold_metrics["fold"] = fold_idx
                fold_metrics["train_time"] = train_time
                # Capture CV parameters if available (AFL CV with internal CV)
                if hasattr(algo_fold, 'best_gamma_'):
                    fold_metrics["best_gamma"] = algo_fold.best_gamma_
                if hasattr(algo_fold, 'best_alpha_'):
                    fold_metrics["best_alpha"] = algo_fold.best_alpha_
                if hasattr(algo_fold, 'best_lambda_ridge_'):
                    fold_metrics["best_lambda_ridge"] = algo_fold.best_lambda_ridge_
                if hasattr(algo_fold, 'cv_score_'):
                    fold_metrics["cv_score"] = algo_fold.cv_score_
                repeat_metrics.append(fold_metrics)

            # Average across folds
            metrics_df = pd.DataFrame(repeat_metrics)
            avg_metrics = metrics_df.mean().to_dict()
            metrics_list.append(avg_metrics)

    # Average across repeats
    final_metrics = pd.DataFrame(metrics_list).mean().to_dict()

    return {
        "n_folds_completed": n_folds,
        "metrics": final_metrics,
        "individual_metrics": metrics_list,
        "n_repeats_completed": n_repeats,
    }


def generate_benchmark_summary(results_df, exp_dir):
    """Generate summary statistics for benchmark by model and sigma."""
    # Overall summary by model
    model_summary = results_df.groupby('model').agg({
        'f1': ['mean', 'std'],
        'mse': ['mean', 'std'],
        'tpr': ['mean', 'std'],
        'fdr': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'sparsity': ['mean', 'std'],
        'n_selected': ['mean', 'std'],
    }).round(4)
    model_summary.columns = ['_'.join(col) for col in model_summary.columns]
    model_summary = model_summary.reset_index()

    # Summary by sigma and model
    sigma_model_summary = results_df.groupby(['sigma', 'snr', 'model']).agg({
        'f1': ['mean', 'std'],
        'mse': ['mean', 'std'],
        'tpr': ['mean', 'std'],
        'fdr': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'r2': ['mean', 'std'],
    }).round(4)
    sigma_model_summary.columns = ['_'.join(col) for col in sigma_model_summary.columns]
    sigma_model_summary = sigma_model_summary.reset_index()

    # Save summaries
    model_summary_path = exp_dir / "summary_by_model.csv"
    model_summary.to_csv(model_summary_path, index=False)

    sigma_model_summary_path = exp_dir / "summary_by_sigma_model.csv"
    sigma_model_summary.to_csv(sigma_model_summary_path, index=False)

    return {
        "by_model": model_summary,
        "by_sigma_model": sigma_model_summary,
    }


def generate_benchmark_rankings(results_df, exp_dir):
    """Generate rankings for benchmark metrics."""
    rankings = {}

    # Rank by overall F1 (higher is better)
    model_f1 = results_df.groupby('model')['f1'].mean().sort_values(ascending=False)
    rankings['by_f1'] = {f"rank_{i+1}": {"model": m, "f1": round(f1, 4)} for i, (m, f1) in enumerate(model_f1.items())}

    # Rank by overall MSE (lower is better)
    model_mse = results_df.groupby('model')['mse'].mean().sort_values(ascending=True)
    rankings['by_mse'] = {f"rank_{i+1}": {"model": m, "mse": round(mse, 4)} for i, (m, mse) in enumerate(model_mse.items())}

    # Rank by FDR (lower is better)
    model_fdr = results_df.groupby('model')['fdr'].mean().sort_values(ascending=True)
    rankings['by_fdr'] = {f"rank_{i+1}": {"model": m, "fdr": round(fdr, 4)} for i, (m, fdr) in enumerate(model_fdr.items())}

    # Rank by TPR (higher is better)
    model_tpr = results_df.groupby('model')['tpr'].mean().sort_values(ascending=False)
    rankings['by_tpr'] = {f"rank_{i+1}": {"model": m, "tpr": round(tpr, 4)} for i, (m, tpr) in enumerate(model_tpr.items())}

    # Save rankings
    rankings_path = exp_dir / "rankings.yaml"
    with open(rankings_path, "w") as f:
        yaml.dump(rankings, f, default_flow_style=False, sort_keys=False)

    return rankings


def generate_benchmark_report(config, exp_dir, results_df, summary, rankings):
    """Generate comprehensive benchmark report in markdown."""
    report_path = exp_dir / "benchmark_report.md"

    with open(report_path, "w") as f:
        f.write(f"# AdaptiveFlippedLasso Benchmark Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Experiment**: {config['experiment']}\n")
        f.write(f"**Data**: Exp2 (AR(1), rho=0.8, n=300, p=500, 20 non-zero)\n")
        f.write(f"**Repeats**: {config.get('n_repeats', 5)} per configuration\n\n")

        f.write("## 1. AdaptiveFlippedLasso Optimal Parameters\n\n")
        f.write("From Stage1 grid search:\n")
        f.write(f"- lambda_ridge: {config.get('lambda_ridge', 1.0)}\n")
        f.write(f"- lambda_: {config.get('lambda_', 1.0)}\n")
        f.write(f"- gamma: {config.get('gamma', 0.5)}\n\n")

        f.write("## 2. Compared Models\n\n")
        models = results_df['model'].unique()
        for m in models:
            use_cv = results_df[results_df['model'] == m]['use_cv'].iloc[0]
            f.write(f"- **{m}** ({'CV-tuned' if use_cv else 'fixed params'})\n")
        f.write("\n")

        f.write("## 3. Overall Performance by Model\n\n")
        f.write("### 3.1 F1 Score (higher is better)\n\n")
        model_f1 = results_df.groupby('model')['f1'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        f.write("| Model | F1 Mean | F1 Std |\n")
        f.write("|-------|---------|--------|\n")
        for model, row in model_f1.iterrows():
            f.write(f"| {model} | {row['mean']:.4f} | {row['std']:.4f} |\n")
        f.write("\n")

        f.write("### 3.2 MSE (lower is better)\n\n")
        model_mse = results_df.groupby('model')['mse'].agg(['mean', 'std']).sort_values('mean', ascending=True)
        f.write("| Model | MSE Mean | MSE Std |\n")
        f.write("|-------|----------|--------|\n")
        for model, row in model_mse.iterrows():
            f.write(f"| {model} | {row['mean']:.4f} | {row['std']:.4f} |\n")
        f.write("\n")

        f.write("### 3.3 TPR - True Positive Rate (higher is better)\n\n")
        model_tpr = results_df.groupby('model')['tpr'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        f.write("| Model | TPR Mean | TPR Std |\n")
        f.write("|-------|----------|--------|\n")
        for model, row in model_tpr.iterrows():
            f.write(f"| {model} | {row['mean']:.4f} | {row['std']:.4f} |\n")
        f.write("\n")

        f.write("### 3.4 FDR - False Discovery Rate (lower is better)\n\n")
        model_fdr = results_df.groupby('model')['fdr'].agg(['mean', 'std']).sort_values('mean', ascending=True)
        f.write("| Model | FDR Mean | FDR Std |\n")
        f.write("|-------|----------|--------|\n")
        for model, row in model_fdr.iterrows():
            f.write(f"| {model} | {row['mean']:.4f} | {row['std']:.4f} |\n")
        f.write("\n")

        f.write("## 4. Performance Across SNR Levels\n\n")
        f.write("### 4.1 F1 by Sigma\n\n")
        sigma_model_f1 = results_df.pivot_table(values='f1', index='sigma', columns='model', aggfunc='mean')
        sigma_model_f1 = sigma_model_f1.round(4)
        f.write(sigma_model_f1.to_markdown() + "\n\n")

        f.write("### 4.2 MSE by Sigma\n\n")
        sigma_model_mse = results_df.pivot_table(values='mse', index='sigma', columns='model', aggfunc='mean')
        sigma_model_mse = sigma_model_mse.round(4)
        f.write(sigma_model_mse.to_markdown() + "\n\n")

        f.write("### 4.3 TPR by Sigma\n\n")
        sigma_model_tpr = results_df.pivot_table(values='tpr', index='sigma', columns='model', aggfunc='mean')
        sigma_model_tpr = sigma_model_tpr.round(4)
        f.write(sigma_model_tpr.to_markdown() + "\n\n")

        f.write("### 4.4 FDR by Sigma\n\n")
        sigma_model_fdr = results_df.pivot_table(values='fdr', index='sigma', columns='model', aggfunc='mean')
        sigma_model_fdr = sigma_model_fdr.round(4)
        f.write(sigma_model_fdr.to_markdown() + "\n\n")

        f.write("## 5. Complete Metrics Summary\n\n")
        f.write("| sigma | SNR | Model | F1 | MSE | TPR | FDR | Precision | Recall | R2 |\n")
        f.write("|-------|-----|-------|-----|-----|-----|-----|----------|--------|-----|\n")
        for sigma in sorted(results_df['sigma'].unique()):
            for model in models:
                subset = results_df[(results_df['sigma'] == sigma) & (results_df['model'] == model)]
                if len(subset) > 0:
                    snr = subset['snr'].iloc[0]
                    f1 = subset['f1'].mean()
                    mse = subset['mse'].mean()
                    tpr = subset['tpr'].mean()
                    fdr = subset['fdr'].mean()
                    prec = subset['precision'].mean()
                    rec = subset['recall'].mean()
                    r2 = subset['r2'].mean()
                    f.write(f"| {sigma} | {snr} | {model} | {f1:.4f} | {mse:.4f} | {tpr:.4f} | {fdr:.4f} | {prec:.4f} | {rec:.4f} | {r2:.4f} |\n")
        f.write("\n")

        f.write("## 6. Rankings Summary\n\n")
        f.write("### 6.1 By F1 (higher is better)\n\n")
        for rank, data in rankings['by_f1'].items():
            f.write(f"- {rank}: {data['model']} (F1={data['f1']:.4f})\n")
        f.write("\n")

        f.write("### 6.2 By MSE (lower is better)\n\n")
        for rank, data in rankings['by_mse'].items():
            f.write(f"- {rank}: {data['model']} (MSE={data['mse']:.4f})\n")
        f.write("\n")

        f.write("## 7. Key Findings\n\n")
        best_f1_model = rankings['by_f1']['rank_1']['model']
        best_f1_val = rankings['by_f1']['rank_1']['f1']
        best_mse_model = rankings['by_mse']['rank_1']['model']
        best_mse_val = rankings['by_mse']['rank_1']['mse']

        f.write(f"1. **Best F1**: {best_f1_model} with F1={best_f1_val:.4f}\n")
        f.write(f"2. **Best MSE**: {best_mse_model} with MSE={best_mse_val:.4f}\n")

        # SNR sensitivity analysis
        high_snr = results_df[results_df['sigma'] <= 0.5]
        low_snr = results_df[results_df['sigma'] >= 2.0]
        if len(high_snr) > 0 and len(low_snr) > 0:
            high_snr_f1 = high_snr.groupby('model')['f1'].mean()
            low_snr_f1 = low_snr.groupby('model')['f1'].mean()
            f.write("\n3. **SNR Sensitivity**:\n")
            for model in models:
                if model in high_snr_f1 and model in low_snr_f1:
                    drop = high_snr_f1[model] - low_snr_f1[model]
                    f.write(f"   - {model}: F1 drop = {drop:.4f} (high SNR to low SNR)\n")

        f.write("\n---\n")
        f.write(f"*Report generated: {datetime.now().isoformat()}*\n")

    return report_path


def run_benchmark(config_path, output_dir=None, dry_run=False):
    """Run benchmark comparison."""
    config = load_config(config_path)
    if output_dir:
        config["output_dir"] = output_dir
    config["config_path"] = config_path

    if dry_run:
        print("[sweep:benchmark] Dry-run mode - validation only")
        print(f"[sweep:benchmark] Config: {config['experiment']}")
        print(f"[sweep:benchmark] SNR values: {config.get('search_space', {}).get('sigma', [])}")
        print(f"[sweep:benchmark] Models to compare: {len(config.get('compare_models', []))}")
        return {"status": "dry_run", "config": config}

    return benchmark_study(config)


def main():
    args = parse_args()

    try:
        if args.stage == "stage1":
            result = run_stage1(
                config_path=args.config,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
            )
        elif args.stage == "stage2":
            result = run_stage2(
                config_path=args.config,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
            )
        elif args.stage == "benchmark":
            result = run_benchmark(
                config_path=args.config,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
            )
        else:  # snr
            result = run_snr(
                config_path=args.config,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
            )

        print(f"\n[sweep] {args.stage} completed!")
        return 0

    except Exception as e:
        print(f"[sweep] FATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
