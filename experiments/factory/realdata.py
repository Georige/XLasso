#!/usr/bin/env python
"""
Real Dataset Experiment Runner
==============================
Evaluates feature selection models on real-world datasets.

Two evaluation modes:
1. Random Split Mode (default for cross-sectional data):
   - Random shuffle data, split 70% train / 30% test
   - Model does internal CV on training data
   - Evaluate on held-out test set

2. Rolling Window Mode (for time series data):
   - Use a rolling training window (e.g., 120 months)
   - Train on window data, predict next month
   - Roll forward by 1 month, repeat
   - Compute RMSE/MAE over all out-of-sample predictions

Usage:
    # Rolling window (time series) - FRED-MD
    python factory/realdata.py --dataset fred_md --mode rolling --window-size 120 --algo elasticnet_1se

    # Random split (cross-sectional)
    python factory/realdata.py --dataset riboflavin --mode random --algo elasticnet_1se --n-iter 100

    # Compare multiple models on time series
    python factory/realdata.py --dataset fred_md --mode rolling --window-size 120 --compare elasticnet_1se,lasso,nlasso
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.factory.run import ALGO_REGISTRY, generate_experiment_dir, save_config


# Real dataset registry
DATASET_REGISTRY = {
    "riboflavin": {
        "path": "experiments/dataset/riboflavin.npz",
        "description": "Riboflavin gene expression (n=71, p=4088)",
        "task": "regression",
    },
    "tecator": {
        "path": "experiments/dataset/tecator.npz",
        "description": "Tecator near-infrared spectra (n=133, p=1024)",
        "task": "regression",
    },
    "fred_md": {
        "path": "experiments/dataset/fred_md.npz",
        "description": "FRED-MD macroeconomic indicators (n=441, p=76)",
        "task": "regression",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run real data experiments")
    parser.add_argument(
        "--dataset", "-d", required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Real dataset to use",
    )
    parser.add_argument(
        "--algo", "-a",
        help="Single algorithm to use (mutually exclusive with --compare)",
    )
    parser.add_argument(
        "--compare", "-c",
        help="Comma-separated list of algorithms to compare (mutually exclusive with --algo)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="experiments/results/realdata",
        help="Output directory",
    )
    # Evaluation mode
    parser.add_argument(
        "--mode", "-m",
        choices=["random", "rolling"],
        default="random",
        help="Evaluation mode: random (shuffle split) or rolling (time series window)",
    )
    # Rolling window parameters
    parser.add_argument(
        "--window-size", "-w",
        type=int, default=120,
        help="Rolling training window size (months) for rolling mode",
    )
    # Random split parameters
    parser.add_argument(
        "--train-ratio", type=float, default=0.7,
        help="Training set ratio for random mode (default: 0.7)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=100,
        help="Number of iterations for random mode (default: 100)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running",
    )
    return parser.parse_args()


def load_dataset(dataset_name):
    """Load a real dataset from npz file."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_info = DATASET_REGISTRY[dataset_name]
    path = Path(dataset_info["path"])

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path)
    X = data["X"]
    y = data["y"]

    if hasattr(X, 'toarray'):
        X = X.toarray()

    print(f"[realdata] Loaded {dataset_name}: X={X.shape}, y={y.shape}")
    print(f"[realdata] Description: {dataset_info['description']}")

    return X, y, dataset_info


def _get_cv_algo_name(algo_name):
    """Map algorithm name to its CV version for real data experiments."""
    cv_mapping = {
        "nlasso": "nlasso_cv",
        "elasticnet_1se": "elasticnet_1se",
        "relaxed_lasso_1se": "relaxed_lasso_1se",
        "lasso": "lasso_cv",
        "adaptive_flipped_lasso": "adaptive_flipped_lasso_cv",
    }
    return cv_mapping.get(algo_name, algo_name)


def get_cv_algo_params(algo_name):
    """Get default CV-based algorithm parameters."""
    params = {
        "standardize": True,
        "fit_intercept": True,
    }

    algo_name = _get_cv_algo_name(algo_name)

    if algo_name == "elasticnet_1se":
        params.update({
            "cv_folds": 5,
            "l1_ratios": [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
            "max_iter": 5000,
            "random_state": 42,
        })
    elif algo_name == "relaxed_lasso_1se":
        params.update({
            "cv": 5,
            "random_state": 42,
            "eps": 1e-3,
            "n_alphas": 100,
        })
    elif algo_name == "lasso_cv":
        params.update({
            "alphas": np.logspace(-4, 1, 30),
            "cv": 5,
            "max_iter": 20000,
            "random_state": 42,
        })
    elif algo_name == "nlasso_cv":
        params.update({
            "lambda_ridge": 10.0,
            "n_lambda": 50,
            "cv_folds": 5,
            "gamma": 0.3,
            "s": 1.0,
            "group_threshold": 0.7,
        })
    elif algo_name == "adaptive_lasso_cv":
        params.update({
            "alphas": np.logspace(-4, 1, 30),
            "gammas": [0.5, 1.0, 2.0],
            "cv": 5,
            "max_iter": 5000,
        })
    elif algo_name == "unilasso_cv":
        params.update({
            "lambda_1": 0.01,
            "lambda_2": 0.01,
            "group_threshold": 0.7,
            "standardize": True,
            "fit_intercept": True,
            "family": "gaussian",
            "n_folds": 5,
        })
    elif algo_name == "adaptive_flipped_lasso":
        params.update({
            "lambda_ridge": 10.0,
            "lambda_": 0.01,
            "gamma": 1.0,
            "max_iter": 2000,
            "tol": 1e-4,
        })
    elif algo_name == "adaptive_flipped_lasso_cv":
        params.update({
            "lambda_ridge_list": (10.0, 50.0, 100.0, 500.0),
            "gamma_list": (0.3, 0.5, 1.0),
            "cv": 5,
            "alpha_min_ratio": 1e-2,
            "n_alpha": 30,
            "max_iter": 10000,
            "tol": 1e-4,
            "standardize": True,
            "fit_intercept": True,
            "verbose": False,
        })
    elif algo_name == "nlasso":
        params.update({
            "lambda_ridge": 10.0,
            "n_lambda": 50,
            "cv_folds": 5,
            "gamma": 0.3,
            "s": 1.0,
            "group_threshold": 0.7,
        })
    elif algo_name == "lasso":
        params.update({
            "alpha": 0.01,
            "max_iter": 5000,
        })

    return params


def run_single_iteration(algo_class, algo_params, X_train, y_train, X_test, y_test, store_coefs=False):
    """Train model and evaluate on test set.

    Args:
        store_coefs: if True, returns the coefficient vector for frequency analysis
    """
    algo = algo_class(**algo_params)
    start_time = time.time()
    algo.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = algo.predict(X_test)
    test_mse = np.mean((y_test - y_pred) ** 2)
    coef = algo.coef_
    nonzero_mask = np.abs(coef) > 1e-6
    n_selected = int(np.sum(nonzero_mask))
    selected_features = np.where(nonzero_mask)[0].tolist()  # List of selected feature indices

    info = {"train_time": train_time, "n_selected": n_selected}
    if hasattr(algo, 'best_alpha_'):
        info["best_alpha"] = algo.best_alpha_
    if hasattr(algo, 'best_gamma_'):
        info["best_gamma"] = algo.best_gamma_
    if hasattr(algo, 'best_lambda_ridge_'):
        info["best_lambda_ridge"] = algo.best_lambda_ridge_

    result = {
        "test_mse": test_mse,
        "model_size": n_selected,
        "sparsity": 1 - n_selected / len(coef) if len(coef) > 0 else 0,
        "selected_features": selected_features,  # Raw feature indices for each iteration
        **info,
    }

    if store_coefs:
        result["coef"] = coef.copy()

    return result


# ============================================================================
# Rolling Window Mode (Time Series)
# ============================================================================

def run_rolling_window(X, y, algo_class, algo_params, window_size=120):
    """
    Rolling window evaluation for time series data.

    Workflow:
        1. Train on window [i, i+window_size)
        2. Predict at time i+window_size
        3. Roll forward by 1, repeat
        4. Collect all out-of-sample predictions
    """
    n_samples = len(y)
    n_predictions = n_samples - window_size

    if n_predictions <= 0:
        raise ValueError(f"Window size {window_size} is too large for {n_samples} samples")

    print(f"[rolling] Window size: {window_size}, Total predictions: {n_predictions}")

    all_predictions = []
    all_actuals = []
    all_results = []

    for i in range(n_predictions):
        train_start = i
        train_end = i + window_size
        test_idx = train_end  # Predict next month

        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[test_idx:test_idx+1]
        y_test = y[test_idx]

        try:
            algo = algo_class(**algo_params)
            algo.fit(X_train, y_train)
            y_pred = algo.predict(X_test)[0]

            n_selected = int(np.sum(np.abs(algo.coef_) > 1e-6))
            abs_error = abs(y_test - y_pred)
            squared_error = (y_test - y_pred) ** 2

            result = {
                "step": i,
                "train_start": train_start,
                "train_end": train_end,
                "predict_at": test_idx,
                "y_true": y_test,
                "y_pred": y_pred,
                "abs_error": abs_error,
                "squared_error": squared_error,
                "model_size": n_selected,
            }

            all_results.append(result)
            all_predictions.append(y_pred)
            all_actuals.append(y_test)

        except Exception as e:
            print(f"  Warning at step {i}: {e}")
            continue

        if (i + 1) % 20 == 0 or i == n_predictions - 1:
            recent_mse = np.mean([r["squared_error"] for r in all_results[-20:]])
            recent_size = np.mean([r["model_size"] for r in all_results[-20:]])
            print(f"  Step {i+1}/{n_predictions}: MSE={recent_mse:.4f}, Model Size={recent_size:.1f}")

    if not all_results:
        raise RuntimeError("No successful predictions")

    return all_results


def run_experiment_rolling(dataset_name, algo_name, window_size=120, random_seed=42):
    """Run rolling window experiment for time series evaluation."""
    print(f"[realdata] Starting rolling window experiment")
    print(f"[realdata] Dataset: {dataset_name}")
    print(f"[realdata] Algorithm: {algo_name}")
    print(f"[realdata] Window size: {window_size}")

    X, y, dataset_info = load_dataset(dataset_name)

    cv_algo_name = _get_cv_algo_name(algo_name)
    algo_class = ALGO_REGISTRY.get(cv_algo_name)
    if algo_class is None:
        raise ValueError(f"Unknown algorithm: {cv_algo_name}")

    algo_params = get_cv_algo_params(algo_name)
    print(f"[realdata] Algorithm params: {algo_params}")

    all_results = run_rolling_window(X, y, algo_class, algo_params, window_size)

    results_df = pd.DataFrame(all_results)

    rmse = np.sqrt(results_df["squared_error"].mean())
    mae = results_df["abs_error"].mean()
    avg_model_size = results_df["model_size"].mean()

    summary = {
        "dataset": dataset_name,
        "algo": algo_name,
        "mode": "rolling",
        "window_size": window_size,
        "n_predictions": len(results_df),
        "rmse": float(rmse),
        "mae": float(mae),
        "model_size": {
            "mean": float(avg_model_size),
            "std": float(results_df["model_size"].std()),
        },
    }

    return {
        "results": results_df,
        "summary": summary,
    }


def run_comparison_rolling(dataset_name, algo_names, window_size=120, random_seed=42):
    """Run rolling window comparison for multiple algorithms."""
    print(f"[realdata] Starting rolling window comparison")
    print(f"[realdata] Dataset: {dataset_name}")
    print(f"[realdata] Algorithms: {algo_names}")
    print(f"[realdata] Window size: {window_size}")

    X, y, dataset_info = load_dataset(dataset_name)

    all_results = {}
    summaries = {}

    for algo_name in algo_names:
        print(f"\n{'='*60}")
        print(f"[realdata] Training: {algo_name}")
        print(f"{'='*60}")

        cv_algo_name = _get_cv_algo_name(algo_name)
        algo_class = ALGO_REGISTRY.get(cv_algo_name)
        if algo_class is None:
            print(f"[realdata] WARNING: Unknown algorithm {cv_algo_name}, skipping")
            continue

        algo_params = get_cv_algo_params(algo_name)
        print(f"[realdata] Algorithm params: {algo_params}")

        try:
            results = run_rolling_window(X, y, algo_class, algo_params, window_size)
            results_df = results["results"]
            all_results[algo_name] = results_df

            rmse = np.sqrt(results_df["squared_error"].mean())
            mae = results_df["abs_error"].mean()
            avg_model_size = results_df["model_size"].mean()

            summaries[algo_name] = {
                "rmse": float(rmse),
                "mae": float(mae),
                "model_size": {
                    "mean": float(avg_model_size),
                    "std": float(results_df["model_size"].std()),
                },
            }
        except Exception as e:
            print(f"[realdata] ERROR with {algo_name}: {e}")
            traceback.print_exc()
            continue

    if not summaries:
        raise RuntimeError("No successful algorithms")

    return {
        "results": all_results,
        "summaries": summaries,
    }


# ============================================================================
# Random Split Mode (Cross-sectional)
# ============================================================================

def run_experiment_random(dataset_name, algo_name, n_iter=100, train_ratio=0.7, random_seed=42):
    """Run random split experiment for cross-sectional evaluation."""
    print(f"[realdata] Starting random split experiment")
    print(f"[realdata] Dataset: {dataset_name}")
    print(f"[realdata] Algorithm: {algo_name}")
    print(f"[realdata] Iterations: {n_iter}")

    X, y, dataset_info = load_dataset(dataset_name)
    n_features = X.shape[1]

    cv_algo_name = _get_cv_algo_name(algo_name)
    algo_class = ALGO_REGISTRY.get(cv_algo_name)
    if algo_class is None:
        raise ValueError(f"Unknown algorithm: {cv_algo_name}")

    algo_params = get_cv_algo_params(algo_name)
    print(f"[realdata] Algorithm params: {algo_params}")

    all_results = []
    coef_matrix = []  # Store coefficients for frequency analysis

    for i in range(n_iter):
        rng = np.random.RandomState(random_seed + i)
        n_samples = len(y)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        n_train = int(n_samples * train_ratio)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        try:
            result = run_single_iteration(
                algo_class, algo_params, X[train_idx], y[train_idx], X[test_idx], y[test_idx],
                store_coefs=True
            )
            result["iteration"] = i
            all_results.append(result)
            coef_matrix.append(result["coef"])

            if (i + 1) % 10 == 0 or i == 0:
                recent_mse = np.mean([r["test_mse"] for r in all_results[-10:]])
                recent_size = np.mean([r["model_size"] for r in all_results[-10:]])
                print(f"  Iter {i+1}/{n_iter}: MSE={recent_mse:.4f}, Size={recent_size:.1f}")

        except Exception as e:
            print(f"  ERROR in iteration {i}: {e}")
            continue

    if not all_results:
        raise RuntimeError("No successful iterations")

    results_df = pd.DataFrame(all_results)

    # Compute feature selection frequency
    coef_matrix = np.array(coef_matrix)
    selection_matrix = np.abs(coef_matrix) > 1e-6
    selection_freq = np.mean(selection_matrix, axis=0)  # Frequency for each feature
    selection_count = np.sum(selection_matrix, axis=0)  # Count for each feature

    # Sort features by selection frequency
    freq_order = np.argsort(selection_freq)[::-1]
    top_features = [(i, selection_freq[i], selection_count[i]) for i in freq_order[:20]]

    print(f"\n[realdata] Top 20 most frequently selected features:")
    for rank, (feat_idx, freq, count) in enumerate(top_features, 1):
        print(f"  {rank}. Feature {feat_idx}: selected {count}/{n_iter} times ({freq*100:.1f}%)")

    summary = {
        "dataset": dataset_name,
        "algo": algo_name,
        "mode": "random",
        "n_iter": n_iter,
        "train_ratio": train_ratio,
        "n_features": n_features,
        "test_mse": {
            "mean": float(results_df["test_mse"].mean()),
            "std": float(results_df["test_mse"].std()),
        },
        "model_size": {
            "mean": float(results_df["model_size"].mean()),
            "std": float(results_df["model_size"].std()),
        },
        "selection_frequency": {
            "top_features": top_features,
            "all_frequencies": {i: float(selection_freq[i]) for i in range(n_features)},
            "all_counts": {i: int(selection_count[i]) for i in range(n_features)},
        },
    }

    return {
        "results": results_df,
        "summary": summary,
        "selection_freq": selection_freq,
        "selection_count": selection_count,
        "coef_matrix": coef_matrix,
    }


def run_comparison_random(dataset_name, algo_names, n_iter=100, train_ratio=0.7, random_seed=42):
    """Run random split comparison for multiple algorithms."""
    print(f"[realdata] Starting random split comparison")
    print(f"[realdata] Dataset: {dataset_name}")
    print(f"[realdata] Algorithms: {algo_names}")

    X, y, dataset_info = load_dataset(dataset_name)
    n_features = X.shape[1]

    all_results = {}
    all_selection_freq = {}
    all_coef_matrices = {}
    summaries = {}

    for algo_name in algo_names:
        print(f"\n{'='*60}")
        print(f"[realdata] Training: {algo_name}")
        print(f"{'='*60}")

        cv_algo_name = _get_cv_algo_name(algo_name)
        algo_class = ALGO_REGISTRY.get(cv_algo_name)
        if algo_class is None:
            print(f"[realdata] WARNING: Unknown algorithm {cv_algo_name}, skipping")
            continue

        algo_params = get_cv_algo_params(algo_name)
        print(f"[realdata] Algorithm params: {algo_params}")

        algo_results = []
        coef_matrix = []

        for i in range(n_iter):
            rng = np.random.RandomState(random_seed + i)
            n_samples = len(y)
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            n_train = int(n_samples * train_ratio)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]

            try:
                result = run_single_iteration(
                    algo_class, algo_params, X[train_idx], y[train_idx], X[test_idx], y[test_idx],
                    store_coefs=True
                )
                result["iteration"] = i
                result["algo"] = algo_name
                algo_results.append(result)
                coef_matrix.append(result["coef"])

                if (i + 1) % 10 == 0 or i == 0:
                    recent_mse = np.mean([r["test_mse"] for r in algo_results[-10:]])
                    recent_size = np.mean([r["model_size"] for r in algo_results[-10:]])
                    print(f"  Iter {i+1}/{n_iter}: MSE={recent_mse:.4f}, Size={recent_size:.1f}")

            except Exception as e:
                print(f"  ERROR in iteration {i}: {e}")
                continue

        if algo_results:
            results_df = pd.DataFrame(algo_results)
            all_results[algo_name] = results_df

            # Compute selection frequency
            coef_matrix = np.array(coef_matrix)
            selection_matrix = np.abs(coef_matrix) > 1e-6
            selection_freq = np.mean(selection_matrix, axis=0)
            selection_count = np.sum(selection_matrix, axis=0)
            all_selection_freq[algo_name] = selection_freq
            all_coef_matrices[algo_name] = coef_matrix

            # Top features
            freq_order = np.argsort(selection_freq)[::-1]
            top_features = [(i, selection_freq[i], int(selection_count[i])) for i in freq_order[:20]]

            print(f"\n[realdata] Top 10 most frequently selected features for {algo_name}:")
            for rank, (feat_idx, freq, count) in enumerate(top_features[:10], 1):
                print(f"  {rank}. Feature {feat_idx}: selected {count}/{n_iter} times ({freq*100:.1f}%)")

            summaries[algo_name] = {
                "test_mse": {
                    "mean": float(results_df["test_mse"].mean()),
                    "std": float(results_df["test_mse"].std()),
                },
                "model_size": {
                    "mean": float(results_df["model_size"].mean()),
                    "std": float(results_df["model_size"].std()),
                },
                "top_features": top_features,
            }

    if not summaries:
        raise RuntimeError("No successful algorithms")

    return {
        "results": all_results,
        "summaries": summaries,
        "selection_freq": all_selection_freq,
        "coef_matrices": all_coef_matrices,
    }


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(result, exp_dir, config, is_comparison=False):
    """Generate markdown report."""
    report_path = exp_dir / "report.md"

    with open(report_path, "w") as f:
        f.write(f"# Real Dataset Experiment Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Dataset**: {config['dataset']} - {DATASET_REGISTRY[config['dataset']]['description']}\n")
        f.write(f"**Mode**: {config['mode']}\n")

        if config['mode'] == 'rolling':
            f.write(f"**Window Size**: {config['window_size']} months\n")
            n_pred = result['summary'].get('n_predictions', 'N/A')
            f.write(f"**Number of Predictions**: {n_pred}\n\n")

            if is_comparison:
                f.write("## Model Comparison (Rolling Window)\n\n")
                f.write("| Model | RMSE | MAE | Model Size (mean±std) |\n")
                f.write("|-------|------|-----|----------------------|\n")
                for algo_name, summary in result["summaries"].items():
                    rmse = summary['rmse']
                    mae = summary['mae']
                    size = f"{summary['model_size']['mean']:.1f} ± {summary['model_size']['std']:.1f}"
                    f.write(f"| {algo_name} | {rmse:.4f} | {mae:.4f} | {size} |\n")

                f.write("\n### Key Findings\n\n")
                best_rmse = min(result["summaries"].keys(), key=lambda x: result["summaries"][x]['rmse'])
                best_size = min(result["summaries"].keys(), key=lambda x: result["summaries"][x]['model_size']['mean'])
                f.write(f"- **Best RMSE**: {best_rmse} ({result['summaries'][best_rmse]['rmse']:.4f})\n")
                f.write(f"- **Most Sparse**: {best_size} ({result['summaries'][best_size]['model_size']['mean']:.1f} features)\n")
            else:
                summary = result["summary"]
                f.write(f"## Algorithm: {config['algo']}\n\n")
                f.write("### Performance Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| RMSE | {summary['rmse']:.4f} |\n")
                f.write(f"| MAE | {summary['mae']:.4f} |\n")
                f.write(f"| Model Size (mean±std) | {summary['model_size']['mean']:.1f} ± {summary['model_size']['std']:.1f} |\n")

        else:  # random mode
            f.write(f"**Iterations**: {config['n_iter']}\n")
            f.write(f"**Train Ratio**: {config['train_ratio']}\n\n")

            if is_comparison:
                f.write("## Model Comparison (Random Split)\n\n")
                f.write("| Model | Test MSE (mean±std) | Model Size (mean±std) |\n")
                f.write("|-------|---------------------|----------------------|\n")
                for algo_name, summary in result["summaries"].items():
                    mse = f"{summary['test_mse']['mean']:.4f} ± {summary['test_mse']['std']:.4f}"
                    size = f"{summary['model_size']['mean']:.1f} ± {summary['model_size']['std']:.1f}"
                    f.write(f"| {algo_name} | {mse} | {size} |\n")

                f.write("\n### Key Findings\n\n")
                best_mse = min(result["summaries"].keys(), key=lambda x: result["summaries"][x]['test_mse']['mean'])
                best_size = min(result["summaries"].keys(), key=lambda x: result["summaries"][x]['model_size']['mean'])
                f.write(f"- **Best Test MSE**: {best_mse} ({result['summaries'][best_mse]['test_mse']['mean']:.4f})\n")
                f.write(f"- **Most Sparse**: {best_size} ({result['summaries'][best_size]['model_size']['mean']:.1f} features)\n")

                # Feature Selection Frequency
                if "selection_freq" in result and result["selection_freq"]:
                    f.write("\n## Feature Selection Frequency Analysis\n\n")
                    f.write("### Top 10 Most Frequently Selected Features by Model\n\n")
                    for algo_name in result["summaries"].keys():
                        top_feats = result["summaries"][algo_name].get("top_features", [])[:10]
                        if top_feats:
                            f.write(f"**{algo_name}**:\n")
                            for rank, (feat_idx, freq, count) in enumerate(top_feats, 1):
                                n_iter = config['n_iter']
                                f.write(f"  {rank}. Feature {feat_idx}: {count}/{n_iter} ({freq*100:.1f}%)\n")
                            f.write("\n")
            else:
                summary = result["summary"]
                f.write(f"## Algorithm: {config['algo']}\n\n")
                f.write("### Performance Metrics\n\n")
                f.write("| Metric | Mean ± Std |\n")
                f.write("|--------|------------|\n")
                f.write(f"| Test MSE | {summary['test_mse']['mean']:.4f} ± {summary['test_mse']['std']:.4f} |\n")
                f.write(f"| Model Size | {summary['model_size']['mean']:.1f} ± {summary['model_size']['std']:.1f} |\n")

                # Feature Selection Frequency
                if "selection_freq" in result:
                    f.write("\n## Feature Selection Frequency Analysis\n\n")
                    top_feats = summary.get("selection_frequency", {}).get("top_features", [])[:20]
                    if top_feats:
                        f.write("### Top 20 Most Frequently Selected Features\n\n")
                        f.write("| Rank | Feature | Count | Frequency |\n")
                        f.write("|------|---------|-------|-----------|\n")
                        for rank, (feat_idx, freq, count) in enumerate(top_feats, 1):
                            f.write(f"| {rank} | {feat_idx} | {count} | {freq*100:.1f}% |\n")

        f.write("\n---\n")
        f.write(f"*Report generated: {datetime.now().isoformat()}*\n")

    print(f"[realdata] Report saved to {report_path}")
    return report_path


def _make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_results(result, exp_dir, config, is_comparison=False):
    """Save experiment results to files."""
    if is_comparison:
        for algo_name, df in result["results"].items():
            raw_path = exp_dir / f"raw_{algo_name}.csv"
            df.to_csv(raw_path, index=False)

            # Save selected features per iteration as JSON
            features_path = exp_dir / f"selected_features_{algo_name}.json"
            features_data = []
            if "selected_features" in df.columns:
                features_data = df["selected_features"].tolist()
            with open(features_path, "w") as f:
                json.dump(features_data, f, indent=2)

        summary_path = exp_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(_make_json_serializable(result["summaries"]), f, indent=2)
    else:
        raw_path = exp_dir / "raw.csv"
        result["results"].to_csv(raw_path, index=False)

        # Save selected features per iteration as JSON
        features_path = exp_dir / "selected_features.json"
        features_data = []
        if "selected_features" in result["results"].columns:
            features_data = result["results"]["selected_features"].tolist()
        with open(features_path, "w") as f:
            json.dump(features_data, f, indent=2)

        summary_path = exp_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(_make_json_serializable(result["summary"]), f, indent=2)
    print(f"[realdata] Results saved to {exp_dir}")


def plot_selection_frequency(result, exp_dir, config, is_comparison=False):
    """Generate bar chart of feature selection frequency.

    Creates a bar chart showing how often each feature was selected
    across all iterations. This demonstrates feature selection stability.

    Visualization strategy:
    1. Find all features with >10% selection frequency in ANY model
    2. Sort by the first model's frequency (establishing "home advantage")
    3. Stack vertically for comparison - reveals signal vs noise clearly
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[realdata] matplotlib not available, skipping frequency plot")
        return

    if is_comparison:
        # Multi-model comparison with shared X-axis sorted by first model
        selection_freq = result.get("selection_freq", {})
        if not selection_freq:
            return

        n_iter = config['n_iter']
        model_names = list(selection_freq.keys())

        # Step 1: Find union of features with >10% frequency in ANY model
        threshold = 0.10
        union_features = set()
        for algo_name, freq in selection_freq.items():
            above_threshold = set(np.where(freq > threshold)[0])
            union_features.update(above_threshold)

        if not union_features:
            print("[realdata] No features above 10% threshold in any model")
            return

        union_features = sorted(union_features)
        n_shared = len(union_features)
        print(f"[realdata] Shared features (>10%% in any model): {n_shared}")

        # Step 2: Sort by first model's frequency (home advantage)
        first_model = model_names[0]
        first_freq = selection_freq[first_model]

        # Create feature index -> sort key mapping
        # Features with higher freq in first model come first
        sort_key = {f: first_freq[f] for f in union_features}
        sorted_features = sorted(union_features, key=lambda f: sort_key[f], reverse=True)

        # Build frequency matrix for plotting
        freq_matrix = np.zeros((len(model_names), n_shared))
        for i, algo_name in enumerate(model_names):
            for j, feat_idx in enumerate(sorted_features):
                freq_matrix[i, j] = selection_freq[algo_name][feat_idx] * 100

        # Step 3: Create figure with stacked subplots
        fig, axes = plt.subplots(len(model_names), 1, figsize=(16, 4 * len(model_names)), squeeze=False)

        # Color scheme - use distinct colors for each model
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

        for i, (algo_name, ax) in enumerate(zip(model_names, axes[:, 0])):
            bars = ax.bar(range(n_shared), freq_matrix[i], color=colors[i % len(colors)], alpha=0.85)

            ax.set_ylabel('Selection Freq (%)', fontsize=10)
            ax.set_title(f'{algo_name}', fontsize=12, fontweight='bold')
            ax.set_xlim(-0.5, n_shared - 0.5)
            ax.set_ylim(0, 105)
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.4, linewidth=1)
            ax.axhline(y=90, color='green', linestyle='--', alpha=0.4, linewidth=1)

            # X-axis: show every 5th label to avoid crowding
            tick_positions = list(range(0, n_shared, max(1, n_shared // 20)))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f'F{sorted_features[k]}' for k in tick_positions], rotation=45, fontsize=8)

            # Mark "pillar" features (90%+) and "noise" features (0% in first model)
            first_model_freq = freq_matrix[0]
            pillar_count = np.sum(first_model_freq >= 90)
            noise_count = np.sum(first_model_freq == 0)

            # Add annotation for pillars
            if pillar_count > 0:
                ax.annotate(f'{pillar_count} pillars\n(90%+)', xy=(0.02, 0.95),
                           xycoords='axes fraction', fontsize=9, color='green', va='top')
            if noise_count > 0:
                ax.annotate(f'{noise_count} noise\n(0% in 1st)', xy=(0.98, 0.95),
                           xycoords='axes fraction', fontsize=9, color='red', va='top', ha='right')

        axes[-1, 0].set_xlabel('Features (sorted by 1st model frequency, left=highest)', fontsize=10)

        plt.suptitle(f'Feature Selection Frequency Comparison\n(n={n_iter} iterations, {n_shared} shared features)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(exp_dir / "selection_frequency_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[realdata] Frequency plot saved to {exp_dir / 'selection_frequency_comparison.png'}")

    else:
        # Single model: bar chart of all features (or top N)
        selection_freq = result.get("selection_freq")
        if selection_freq is None:
            return

        n_features = len(selection_freq)
        n_iter = config['n_iter']

        # For high-dimensional data, only show top features
        if n_features > 100:
            top_n = 100
            top_indices = np.argsort(selection_freq)[::-1][:top_n]
            top_freq = selection_freq[top_indices]

            fig, ax = plt.subplots(figsize=(14, 6))
            bars = ax.bar(range(top_n), top_freq * 100, color='steelblue', alpha=0.8)
            ax.set_ylabel('Selection Frequency (%)')
            ax.set_xlabel('Feature Rank')
            ax.set_title(f'{config["algo"]} - Top {top_n} Feature Selection Frequency ({n_iter} iterations)')
            ax.set_xticks(range(top_n))
            ax.set_xticklabels([f'F{i}' for i in top_indices], rotation=45, fontsize=8)
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
            ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% threshold (stable)')
            ax.legend()
        else:
            # Show all features
            fig, ax = plt.subplots(figsize=(max(14, n_features * 0.3), 6))
            bars = ax.bar(range(n_features), selection_freq * 100, color='steelblue', alpha=0.8)
            ax.set_ylabel('Selection Frequency (%)')
            ax.set_xlabel('Feature Index')
            ax.set_title(f'{config["algo"]} - Feature Selection Frequency ({n_iter} iterations)')
            ax.set_xticks(range(n_features))
            ax.set_xticklabels([f'{i}' for i in range(n_features)], rotation=90, fontsize=6)
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        fig.savefig(exp_dir / "selection_frequency.png", dpi=150)
        plt.close()
        print(f"[realdata] Frequency plot saved to {exp_dir / 'selection_frequency.png'}")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # Parse algorithm(s)
    if args.compare:
        algo_names = [a.strip() for a in args.compare.split(",")]
        is_comparison = len(algo_names) > 1
        experiment_name = f"{args.dataset}_{args.mode}_{'-'.join(algo_names)}"
    elif args.algo:
        algo_names = [args.algo]
        is_comparison = False
        experiment_name = f"{args.dataset}_{args.mode}_{args.algo}"
    else:
        print("[realdata] ERROR: Must specify either --algo or --compare")
        return 1

    config = {
        "experiment": experiment_name,
        "dataset": args.dataset,
        "algo": algo_names[0] if not is_comparison else ",".join(algo_names),
        "mode": args.mode,
        "window_size": args.window_size,
        "n_iter": args.n_iter,
        "train_ratio": args.train_ratio,
        "random_seed": args.random_seed,
        "output_dir": args.output_dir,
    }

    if args.dry_run:
        print("[realdata] Dry-run mode - validation only")
        print(f"[realdata] Config: {config}")
        return 0

    try:
        exp_dir = generate_experiment_dir(config)
        print(f"[realdata] Output directory: {exp_dir}")
        save_config(config, exp_dir)

        # Run experiment based on mode
        if args.mode == "rolling":
            if is_comparison:
                result = run_comparison_rolling(
                    dataset_name=args.dataset,
                    algo_names=algo_names,
                    window_size=args.window_size,
                    random_seed=args.random_seed,
                )
            else:
                result = run_experiment_rolling(
                    dataset_name=args.dataset,
                    algo_name=algo_names[0],
                    window_size=args.window_size,
                    random_seed=args.random_seed,
                )
        else:  # random mode
            if is_comparison:
                result = run_comparison_random(
                    dataset_name=args.dataset,
                    algo_names=algo_names,
                    n_iter=args.n_iter,
                    train_ratio=args.train_ratio,
                    random_seed=args.random_seed,
                )
            else:
                result = run_experiment_random(
                    dataset_name=args.dataset,
                    algo_name=algo_names[0],
                    n_iter=args.n_iter,
                    train_ratio=args.train_ratio,
                    random_seed=args.random_seed,
                )

        save_results(result, exp_dir, config, is_comparison=is_comparison)
        generate_report(result, exp_dir, config, is_comparison=is_comparison)

        # Generate feature selection frequency plot (only for random mode)
        if args.mode == "random":
            try:
                plot_selection_frequency(result, exp_dir, config, is_comparison=is_comparison)
            except Exception as e:
                print(f"[realdata] Warning: Could not generate frequency plot: {e}")

        print(f"\n[realdata] Completed!")
        print(f"[realdata] Results: {exp_dir}")
        return 0

    except Exception as e:
        print(f"[realdata] FATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
