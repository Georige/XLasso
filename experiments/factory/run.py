#!/usr/bin/env python
"""
Single Experiment Runner
========================
Executes a single experiment with a given config, outputs the "four pieces":
- config.yaml: experiment configuration copy
- status.json: running status (supports resume from checkpoint)
- raw.csv: complete iteration record
- summary.csv: summary statistics

Usage:
    python factory/run.py --config configs/stage1/example.yaml
    python factory/run.py --config /path/to/config.yaml --output-dir results/pilot/run
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add XLasso root directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.modules import (
    NLasso,
    NLassoClassifier,
    NLassoCV,
    NLassoClassifierCV,
    AdaptiveLasso,
    AdaptiveLassoCV,
    FusedLasso,
    FusedLassoCV,
    GroupLasso,
    GroupLassoCV,
    AdaptiveSparseGroupLasso,
    AdaptiveSparseGroupLassoCV,
    DataGenerator,
    MetricCalculator,
    CrossValidator,
)


# Algorithm registry with proper class references
ALGO_REGISTRY = {
    # NLasso family
    "nlasso": NLasso,
    "nlclassifier": NLassoClassifier,
    "nlasso_cv": NLassoCV,
    "nlclassifier_cv": NLassoClassifierCV,
    # Other Lasso variants
    "adaptive_lasso": AdaptiveLasso,
    "adaptive_lasso_cv": AdaptiveLassoCV,
    "fused_lasso": FusedLasso,
    "fused_lasso_cv": FusedLassoCV,
    "group_lasso": GroupLasso,
    "group_lasso_cv": GroupLassoCV,
    "adaptive_sparse_group_lasso": AdaptiveSparseGroupLasso,
    "adaptive_sparse_group_lasso_cv": AdaptiveSparseGroupLassoCV,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run single experiment")
    parser.add_argument(
        "--config", "-c", required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Override output directory (default: from config)",
    )
    parser.add_argument(
        "--fold",
        "-f",
        type=int,
        default=None,
        help="Run only specific fold (for resume)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running",
    )
    return parser.parse_args()


def load_config(config_path):
    """Load and validate config from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required = ["experiment", "algo", "n_samples", "n_features", "n_nonzero"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    # Set defaults
    config.setdefault("sigma", 1.0)
    config.setdefault("correlation_type", "pairwise")
    config.setdefault("rho", 0.5)
    config.setdefault("n_repeats", 3)
    config.setdefault("cv_folds", 5)
    config.setdefault("output_dir", "results/pilot/run")
    config.setdefault("search_space", {})
    config.setdefault("lambda_1", 0.01)
    config.setdefault("lambda_2", 0.01)
    config.setdefault("group_threshold", 0.7)

    return config


def generate_experiment_dir(config, base_dir=None):
    """Generate unique experiment directory with timestamp."""
    if base_dir is None:
        base_dir = config["output_dir"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config["experiment"]
    dir_name = f"{exp_name}__{timestamp}"

    exp_dir = Path(base_dir) / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def save_config(config, exp_dir):
    """Save config to experiment directory."""
    config_path = exp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


def load_status(exp_dir):
    """Load status.json if exists."""
    status_path = exp_dir / "status.json"
    if status_path.exists():
        with open(status_path, "r") as f:
            return json.load(f)
    return None


def save_status(status, exp_dir):
    """Save status.json."""
    status_path = exp_dir / "status.json"
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)


def init_status(config, exp_dir, n_folds):
    """Initialize status.json for new experiment."""
    status = {
        "status": "running",
        "experiment": config["experiment"],
        "algo": config["algo"],
        "created_at": datetime.now().isoformat(),
        "n_folds": n_folds,
        "completed_folds": [],
        "current_fold": 0,
        "config_path": str(exp_dir / "config.yaml"),
    }
    save_status(status, exp_dir)
    return status


def update_status(status, exp_dir, fold=None, completed=False):
    """Update status during experiment."""
    if fold is not None:
        status["current_fold"] = fold
        if fold not in status["completed_folds"]:
            status["completed_folds"].append(fold)

    if completed:
        status["status"] = "completed"
        status["completed_at"] = datetime.now().isoformat()

    save_status(status, exp_dir)


def get_algo_params(algo_name, config):
    """Get algorithm-specific parameters from config."""
    # Common parameters
    params = {
        "standardize": True,
        "fit_intercept": True,
    }

    if algo_name in ["nlasso", "nlclassifier"]:
        # NLasso parameters
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
    elif algo_name in ["nlasso_cv", "nlclassifier_cv"]:
        # NLasso CV parameters
        params.update({
            "lambda_ridge": config.get("lambda_ridge", 10.0),
            "n_lambda": config.get("n_lambda", 50),
            "cv_folds": config.get("cv_folds", 5),
            "gamma": config.get("gamma", 0.3),
            "s": config.get("s", 1.0),
            "group_threshold": config.get("group_threshold", 0.7),
        })
    elif algo_name.startswith("adaptive_lasso"):
        # AdaptiveLasso parameters (sklearn-compatible)
        params.update({
            "alpha": config.get("lambda_", 0.01),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name.startswith("fused_lasso"):
        # FusedLasso parameters
        params.update({
            "alpha": config.get("lambda_", 0.01),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name.startswith("group_lasso"):
        # GroupLasso parameters
        params.update({
            "alpha": config.get("lambda_", 0.01),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    elif algo_name.startswith("adaptive_sparse_group_lasso"):
        # AdaptiveSparseGroupLasso parameters
        params.update({
            "alpha": config.get("lambda_", 0.01),
            "max_iter": config.get("max_iter", 1000),
            "tol": config.get("tol", 1e-4),
        })
    else:
        # Default fallback
        params.update({
            "alpha": config.get("lambda_", 0.01),
        })

    return params


def run_single_fold(
    algo_class, config, X, y, beta_true, fold_idx, train_idx, test_idx
):
    """Run a single fold of cross-validation."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Get algorithm name and parameters
    algo_name = config["algo"].lower()

    # Initialize algorithm with appropriate parameters
    algo_params = get_algo_params(algo_name, config)
    algo = algo_class(**algo_params)

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
    fold_metrics["fold"] = fold_idx
    fold_metrics["train_time"] = train_time

    return fold_metrics, algo.coef_


def save_raw_results(all_fold_metrics, exp_dir):
    """Save raw.csv with all fold results."""
    df = pd.DataFrame(all_fold_metrics)
    raw_path = exp_dir / "raw.csv"
    df.to_csv(raw_path, index=False)
    return raw_path


def save_summary(all_fold_metrics, exp_dir):
    """Save summary.csv with aggregated statistics."""
    df = pd.DataFrame(all_fold_metrics)

    # Calculate statistics across folds
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    agg_funcs = ["mean", "std", "min", "max"]

    # Build summary with proper column names: metric_aggregation (e.g., mse_mean)
    summary_dict = {}
    for col in numeric_cols:
        for func in agg_funcs:
            summary_dict[f"{col}_{func}"] = [df[col].agg(func)]

    summary = pd.DataFrame(summary_dict)

    summary_path = exp_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def check_resume(status, fold):
    """Check if this fold was already completed."""
    if status is None:
        return False
    return fold in status.get("completed_folds", [])


def run_experiment(config_path, output_dir=None, fold=None, dry_run=False):
    """Main experiment runner."""
    # Load config
    config = load_config(config_path)

    if output_dir is not None:
        config["output_dir"] = output_dir

    # Generate experiment directory
    exp_dir = generate_experiment_dir(config)

    # Save config
    save_config(config, exp_dir)
    print(f"[run] Experiment: {config['experiment']}")
    print(f"[run] Algorithm: {config['algo']}")
    print(f"[run] Output: {exp_dir}")

    if dry_run:
        print("[run] Dry-run mode - validation only, no execution")
        return {"status": "dry_run", "exp_dir": str(exp_dir), "config": config}

    # Check for resume
    status = load_status(exp_dir)
    n_folds = config.get("cv_folds", 5)

    if status is None:
        status = init_status(config, exp_dir, n_folds)
    else:
        print(f"[run] Resuming experiment from fold {status.get('current_fold', 0)}")
        print(f"[run] Completed folds: {status.get('completed_folds', [])}")

    # Generate data
    print(f"[run] Generating data: n={config['n_samples']}, p={config['n_features']}")
    data_gen = DataGenerator(random_state=config.get("random_state", 42))
    X, y, beta_true = data_gen.generate(
        n_samples=config["n_samples"],
        n_features=config["n_features"],
        n_nonzero=config["n_nonzero"],
        sigma=config.get("sigma", 1.0),
        correlation_type=config.get("correlation_type", "pairwise"),
        rho=config.get("rho", 0.5),
    )
    config["beta_true"] = beta_true.tolist() if isinstance(beta_true, np.ndarray) else beta_true

    # Get algorithm class
    algo_name = config["algo"].lower()
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(
            f"Unknown algorithm: {algo_name}. Available: {list(ALGO_REGISTRY.keys())}"
        )
    algo_class = ALGO_REGISTRY[algo_name]

    # Cross-validation
    cv = CrossValidator(n_folds=n_folds, shuffle=True, random_state=42)
    cv_splits = cv.split(X)

    all_fold_metrics = []

    # Check if we should resume from a specific fold
    start_fold = 0
    if status and status.get("completed_folds"):
        start_fold = max(status["completed_folds"]) + 1
        print(f"[run] Starting from fold {start_fold} (completed: {status['completed_folds']})")

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        if fold is not None and fold_idx != fold:
            continue
        if fold_idx < start_fold:
            continue

        print(f"[run] Running fold {fold_idx + 1}/{n_folds}...")

        try:
            if check_resume(status, fold_idx):
                print(f"[run] Fold {fold_idx} already completed, skipping")
                continue

            update_status(status, exp_dir, fold=fold_idx)

            fold_metrics, coefs = run_single_fold(
                algo_class, config, X, y, beta_true, fold_idx, train_idx, test_idx
            )
            all_fold_metrics.append(fold_metrics)

            print(
                f"[run] Fold {fold_idx + 1} - "
                f"MSE: {fold_metrics.get('mse', 0):.4f}, "
                f"F1: {fold_metrics.get('f1', 0):.4f}, "
                f"TPR: {fold_metrics.get('tpr', 0):.4f}"
            )

        except Exception as e:
            print(f"[run] Error in fold {fold_idx}: {e}")
            traceback.print_exc()
            continue

    # Save results
    if all_fold_metrics:
        save_raw_results(all_fold_metrics, exp_dir)
        save_summary(all_fold_metrics, exp_dir)
        update_status(status, exp_dir, completed=True)
        print(f"[run] Results saved to {exp_dir}")
    else:
        print("[run] WARNING: No fold metrics collected!")

    # Final status
    final_status = load_status(exp_dir)
    return {
        "status": final_status["status"] if final_status else "unknown",
        "exp_dir": str(exp_dir),
        "n_folds_completed": len(all_fold_metrics),
        "config": config,
    }


def main():
    args = parse_args()

    try:
        result = run_experiment(
            config_path=args.config,
            output_dir=args.output_dir,
            fold=args.fold,
            dry_run=args.dry_run,
        )
        print(f"\n[run] Experiment completed with status: {result['status']}")
        return 0 if result["status"] == "completed" else 1

    except Exception as e:
        print(f"[run] FATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
