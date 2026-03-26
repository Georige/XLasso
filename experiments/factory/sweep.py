#!/usr/bin/env python
"""
Two-Stage Hyperparameter Tuning Driver
======================================
Stage1: Grid search to identify optimal structural region
Stage2: Fine-grained search with CV within the optimal region

Usage:
    # Stage 1: Coarse grid search
    python factory/sweep.py stage1 --config configs/stage1/example.yaml

    # Stage 2: Fine CV search within Stage1's optimal region
    python factory/sweep.py stage2 --config configs/stage2/example_cv.yaml
"""

import argparse
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
        choices=["stage1", "stage2"],
        help="Stage to run: stage1 (grid search) or stage2 (fine CV tuning)",
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
    config["output_dir"] = config.get("output_dir", "results/stage1")
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
                # Flatten params into individual columns
                result_entry = {"idx": idx}
                for k, v in param_dict.items():
                    result_entry[f"params_{k}"] = v
                result_entry.update(result["metrics"])
                all_results.append(result_entry)

                # Track best
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
        random_state = 42 + repeat

        # Generate data
        data_gen = DataGenerator(random_state=random_state)
        X, y, beta_true = data_gen.generate(
            n_samples=config["n_samples"],
            n_features=config["n_features"],
            n_nonzero=config["n_nonzero"],
            sigma=config.get("sigma", 1.0),
            correlation_type=config.get("correlation_type", "pairwise"),
            rho=config.get("rho", 0.5),
        )

        # Get algorithm
        algo_name = config["algo"].lower()
        algo_class = ALGO_REGISTRY.get(algo_name)
        if algo_class is None:
            raise ValueError(f"Unknown algorithm: {algo_name}")

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
    config["output_dir"] = config.get("output_dir", "results/stage2")
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


def main():
    args = parse_args()

    try:
        if args.stage == "stage1":
            result = run_stage1(
                config_path=args.config,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
            )
        else:
            result = run_stage2(
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
