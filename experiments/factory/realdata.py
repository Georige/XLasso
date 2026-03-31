#!/usr/bin/env python
"""
Real Dataset Experiment Runner
==============================
Evaluates feature selection models on real-world datasets.

Data Flow (per repeat):
    1. Load data with preprocessing
    2. seed = 42 + repeat
    3. 80/20 train/test split using seed
    4. Generate 10-fold CV splits on training data using seed
    5. Pass (X_train, y_train, cv_splits) to algorithm
    6. Algorithm handles CV internally
    7. Evaluate on held-out test set

Usage:
    python factory/realdata.py --config configs/realdata/my_experiment.yaml
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
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.factory.run import ALGO_REGISTRY, generate_experiment_dir, save_config


# Real dataset registry
DATASET_REGISTRY = {
    "riboflavin": {
        "path": "dataset/riboflavin.npz",
        "description": "Riboflavin gene expression (n=71, p=4088)",
        "task": "regression",
    },
    "tecator": {
        "path": "dataset/tecator.npz",
        "description": "Tecator near-infrared spectra (n=133, p=1024)",
        "task": "regression",
    },
    "fred_md": {
        "path": "dataset/fred_md.npz",
        "description": "FRED-MD macroeconomic indicators (n=441, p=76)",
        "task": "regression",
    },
}


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Run real data experiments")
    parser.add_argument(
        "--config", "-C",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=list(DATASET_REGISTRY.keys()),
        help="Real dataset to use",
    )
    parser.add_argument(
        "--algo", "-a",
        help="Single algorithm to use (mutually exclusive with --compare)",
    )
    parser.add_argument(
        "--compare", "-c",
        help="Comma-separated list of algorithms to compare",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="experiments/results/realdata",
        help="Output directory",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["random", "rolling"],
        default="random",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--window-size", "-w",
        type=int, default=120,
        help="Rolling training window size",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7,
        help="Training set ratio",
    )
    parser.add_argument(
        "--n-iter", type=int, default=100,
        help="Number of iterations",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed base",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running",
    )
    return parser.parse_args()


def load_dataset(dataset_name, preprocess=None):
    """Load a real dataset from npz file with optional preprocessing."""
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

    # Apply preprocessing
    if preprocess:
        X = apply_preprocessing(X, y, preprocess)

    print(f"[realdata] Loaded {dataset_name}: X={X.shape}, y={y.shape}")
    print(f"[realdata] Description: {dataset_info['description']}")

    return X, y, dataset_info


def apply_preprocessing(X, y, preprocess):
    """Apply preprocessing to features."""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    method = preprocess.get("method", "standard")
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "none":
        return X
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

    X = scaler.fit_transform(X)
    return X


def _get_cv_algo_name(algo_name):
    """Map algorithm name to its CV version."""
    cv_mapping = {
        "nlasso": "nlasso_cv",
        "elasticnet_1se": "elasticnet_1se",
        "relaxed_lasso_1se": "relaxed_lasso_1se",
        "lasso": "lasso_cv",
        "adaptive_flipped_lasso": "adaptive_flipped_lasso_cv",
    }
    return cv_mapping.get(algo_name, algo_name)


def get_cv_algo_params(algo_name, custom_params=None):
    """Get default CV-based algorithm parameters, merged with custom params if provided."""
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
    elif algo_name == "cv_unilasso":
        params.update({
            "n_folds": 5,
            "lmda_min_ratio": 0.0001,
            "use_1se": True,
            "random_state": 42,
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
    elif algo_name == "pfl_regressor_cv":
        params.update({
            "cv": 5,
            "lambda_ridge_list": [0.1, 1.0, 10.0, 100.0],
            "alpha_min_ratio": 0.0001,
            "n_alpha": 100,
            "max_iter": 10000,
            "tol": 0.0001,
            "random_state": 2026,
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

    # Merge custom params (higher priority)
    if custom_params:
        params.update(custom_params)

    return params


def run_single_repeat(algo_class, algo_params, X, y, repeat_idx, train_ratio, seed_base, cv_folds):
    """
    Run a single repeat with proper data flow:
    1. seed = seed_base + repeat_idx
    2. 80/20 train/test split using seed
    3. Generate 10-fold CV splits on training data using seed
    4. Pass (X_train, y_train, cv_splits) to algorithm
    5. Evaluate on held-out test set

    Returns metrics dict or None on error.
    """
    import inspect
    from sklearn.model_selection import KFold, train_test_split

    seed = seed_base + repeat_idx

    # Step 1: 80/20 train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        train_size=train_ratio,
        test_size=1 - train_ratio,
        random_state=seed,
        shuffle=True
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Step 2: Generate 10-fold CV splits on training data
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_splits = list(kfold.split(X_train))

    # Step 3: Initialize algorithm and check if it supports cv_splits
    algo = algo_class(**algo_params)
    supports_cv_splits = 'cv_splits' in inspect.signature(algo.fit).parameters

    # Step 4: Train with CV splits
    start_time = time.time()
    if supports_cv_splits:
        try:
            algo.fit(X_train, y_train, cv_splits=cv_splits)
        except TypeError:
            # Some algorithms don't accept cv_splits
            algo.fit(X_train, y_train)
    else:
        algo.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Step 5: Evaluate on test set
    y_pred = algo.predict(X_test)
    test_mse = np.mean((y_test - y_pred) ** 2)
    coef = algo.coef_
    nonzero_mask = np.abs(coef) > 1e-6
    n_selected = int(np.sum(nonzero_mask))

    result = {
        "repeat": repeat_idx,
        "seed": seed,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "test_mse": test_mse,
        "model_size": n_selected,
        "sparsity": 1 - n_selected / len(coef) if len(coef) > 0 else 0,
        "train_time": train_time,
        "n_selected": n_selected,
        "selected_features": np.where(nonzero_mask)[0].tolist(),
    }

    # Capture CV parameters if available
    if hasattr(algo, 'best_alpha_'):
        result["best_alpha"] = algo.best_alpha_
    if hasattr(algo, 'best_gamma_'):
        result["best_gamma"] = algo.best_gamma_
    if hasattr(algo, 'best_lambda_ridge_'):
        result["best_lambda_ridge"] = algo.best_lambda_ridge_
    if hasattr(algo, 'cv_score_'):
        result["cv_score"] = algo.cv_score_

    return result


def run_experiment_random(dataset_name, algo_name, algo_params=None, n_iter=100,
                          train_ratio=0.8, seed_base=42, cv_folds=10,
                          preprocess=None):
    """Run random split experiment for cross-sectional evaluation."""
    print(f"[realdata] Starting random split experiment")
    print(f"[realdata] Dataset: {dataset_name}")
    print(f"[realdata] Algorithm: {algo_name}")
    print(f"[realdata] Iterations: {n_iter}, CV folds: {cv_folds}")
    print(f"[realdata] Train ratio: {train_ratio}, Seed base: {seed_base}")

    X, y, dataset_info = load_dataset(dataset_name, preprocess=preprocess)
    n_features = X.shape[1]

    cv_algo_name = _get_cv_algo_name(algo_name)
    algo_class = ALGO_REGISTRY.get(cv_algo_name)
    if algo_class is None:
        raise ValueError(f"Unknown algorithm: {cv_algo_name}")

    algo_params = get_cv_algo_params(algo_name, custom_params=algo_params)
    print(f"[realdata] Algorithm params: {algo_params}")

    all_results = []
    coef_matrix = []

    for i in range(n_iter):
        try:
            result = run_single_repeat(
                algo_class=algo_class,
                algo_params=algo_params,
                X=X, y=y,
                repeat_idx=i,
                train_ratio=train_ratio,
                seed_base=seed_base,
                cv_folds=cv_folds
            )
            result["iteration"] = i
            all_results.append(result)

            if (i + 1) % 10 == 0 or i == 0:
                recent_mse = np.mean([r["test_mse"] for r in all_results[-10:]])
                recent_size = np.mean([r["model_size"] for r in all_results[-10:]])
                print(f"  Iter {i+1}/{n_iter}: MSE={recent_mse:.4f}, Size={recent_size:.1f}")

        except Exception as e:
            print(f"  ERROR in iteration {i}: {e}")
            traceback.print_exc()
            continue

    if not all_results:
        raise RuntimeError("No successful iterations")

    results_df = pd.DataFrame(all_results)

    # Compute feature selection frequency
    coef_matrix = np.array(coef_matrix) if coef_matrix else np.array([])
    if len(coef_matrix) > 0:
        selection_matrix = np.abs(coef_matrix) > 1e-6
        selection_freq = np.mean(selection_matrix, axis=0)
        selection_count = np.sum(selection_matrix, axis=0)
        freq_order = np.argsort(selection_freq)[::-1]
        top_features = [(i, selection_freq[i], selection_count[i]) for i in freq_order[:20]]
    else:
        selection_freq = None
        selection_count = None
        top_features = []

    print(f"\n[realdata] Top 20 most frequently selected features:")
    for rank, (feat_idx, freq, count) in enumerate(top_features, 1):
        print(f"  {rank}. Feature {feat_idx}: selected {count}/{n_iter} times ({freq*100:.1f}%)")

    summary = {
        "dataset": dataset_name,
        "algo": algo_name,
        "mode": "random",
        "n_iter": n_iter,
        "train_ratio": train_ratio,
        "cv_folds": cv_folds,
        "n_features": n_features,
        "test_mse": {
            "mean": float(results_df["test_mse"].mean()),
            "std": float(results_df["test_mse"].std()),
        },
        "model_size": {
            "mean": float(results_df["model_size"].mean()),
            "std": float(results_df["model_size"].std()),
        },
    }

    return {
        "results": results_df,
        "summary": summary,
        "selection_freq": selection_freq,
        "selection_count": selection_count,
    }


def run_comparison_random(dataset_name, algo_names, algo_params_map=None, n_iter=100,
                           train_ratio=0.8, seed_base=42, cv_folds=10,
                           preprocess=None):
    """Run random split comparison for multiple algorithms."""
    print(f"[realdata] Starting random split comparison")
    print(f"[realdata] Dataset: {dataset_name}")
    print(f"[realdata] Algorithms: {algo_names}")
    print(f"[realdata] Iterations: {n_iter}, CV folds: {cv_folds}")

    if algo_params_map is None:
        algo_params_map = {}

    X, y, dataset_info = load_dataset(dataset_name, preprocess=preprocess)
    n_features = X.shape[1]

    all_results = {}
    all_selection_freq = {}
    all_selection_count = {}
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

        custom_params = algo_params_map.get(algo_name, {})
        algo_params = get_cv_algo_params(algo_name, custom_params=custom_params)
        print(f"[realdata] Algorithm params: {algo_params}")

        algo_results = []
        all_selected_features = []

        for i in range(n_iter):
            try:
                result = run_single_repeat(
                    algo_class=algo_class,
                    algo_params=algo_params,
                    X=X, y=y,
                    repeat_idx=i,
                    train_ratio=train_ratio,
                    seed_base=seed_base,
                    cv_folds=cv_folds
                )
                result["iteration"] = i
                result["algo"] = algo_name
                algo_results.append(result)
                all_selected_features.append(result["selected_features"])

                if (i + 1) % max(1, n_iter // 10) == 0 or i == 0 or i == n_iter - 1:
                    recent_mse = np.mean([r["test_mse"] for r in algo_results[-10:]])
                    recent_size = np.mean([r["model_size"] for r in algo_results[-10:]])
                    print(f"  Iter {i+1}/{n_iter}: MSE={recent_mse:.4f}, Size={recent_size:.1f}")

            except Exception as e:
                print(f"  ERROR in iteration {i}: {e}")
                continue

        if algo_results:
            results_df = pd.DataFrame(algo_results)
            all_results[algo_name] = results_df

            # Compute selection frequency across all iterations
            max_features = n_features
            selection_matrix = np.zeros((n_iter, max_features), dtype=bool)
            for i, feats in enumerate(all_selected_features):
                for f in feats:
                    if f < max_features:
                        selection_matrix[i, f] = True

            selection_freq = np.mean(selection_matrix, axis=0)
            selection_count = np.sum(selection_matrix, axis=0)
            all_selection_freq[algo_name] = selection_freq
            all_selection_count[algo_name] = selection_count

            # Top features
            freq_order = np.argsort(selection_freq)[::-1]
            top_features = [(i, selection_freq[i], int(selection_count[i])) for i in freq_order[:20]]

            print(f"\n[realdata] Top 10 most frequently selected features for {algo_name}:")
            for rank, (feat_idx, freq, count) in enumerate(top_features[:10], 1):
                print(f"  {rank}. Feature {feat_idx}: {count}/{n_iter} ({freq*100:.1f}%)")

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
        "selection_count": all_selection_count,
    }


def run_experiment_rolling(dataset_name, algo_name, algo_params=None, window_size=120, seed_base=42):
    """Run rolling window experiment for time series evaluation."""
    print(f"[realdata] Starting rolling window experiment")
    print(f"[realdata] Dataset: {dataset_name}")
    print(f"[realdata] Algorithm: {algo_name}")
    print(f"[realdata] Window size: {window_size}")

    X, y, dataset_info = load_dataset(dataset_name)
    n_samples = len(y)
    n_predictions = n_samples - window_size

    if n_predictions <= 0:
        raise ValueError(f"Window size {window_size} is too large for {n_samples} samples")

    cv_algo_name = _get_cv_algo_name(algo_name)
    algo_class = ALGO_REGISTRY.get(cv_algo_name)
    if algo_class is None:
        raise ValueError(f"Unknown algorithm: {cv_algo_name}")

    algo_params = get_cv_algo_params(algo_name, custom_params=algo_params)
    print(f"[realdata] Algorithm params: {algo_params}")

    print(f"[rolling] Window size: {window_size}, Total predictions: {n_predictions}")

    all_results = []

    for i in range(n_predictions):
        train_start = i
        train_end = i + window_size
        test_idx = train_end

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

        except Exception as e:
            print(f"  Warning at step {i}: {e}")
            continue

        if (i + 1) % 20 == 0 or i == n_predictions - 1:
            recent_mse = np.mean([r["squared_error"] for r in all_results[-20:]])
            recent_size = np.mean([r["model_size"] for r in all_results[-20:]])
            print(f"  Step {i+1}/{n_predictions}: MSE={recent_mse:.4f}, Model Size={recent_size:.1f}")

    if not all_results:
        raise RuntimeError("No successful predictions")

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
        f.write(f"**CV Folds**: {config.get('cv_folds', 'N/A')}\n")
        f.write(f"**Train Ratio**: {config.get('train_ratio', 'N/A')}\n")
        f.write(f"**Iterations**: {config.get('n_iter', 'N/A')}\n\n")

        if config['mode'] == 'rolling':
            f.write(f"**Window Size**: {config.get('window_size', 120)} months\n")
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
            if is_comparison:
                f.write("## Model Comparison (Random Split with Internal CV)\n\n")
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
                                n_iter = config.get('n_iter', 'N/A')
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


def save_selection_frequency(result, exp_dir, config, is_comparison=False):
    """Save feature selection frequency data for visualization.

    Saves:
    - selection_frequency.csv: feature index, selection count, selection frequency for each algo
    - selection_frequency_long.csv: long format (melted) for ggplot2/seaborn faceting
    """
    if is_comparison:
        selection_freq = result.get("selection_freq", {})
        selection_count = result.get("selection_count", {})
        n_iter = config.get("n_iter", 100)

        if not selection_freq:
            return

        algo_names = list(selection_freq.keys())
        n_features = len(selection_freq[algo_names[0]])

        # Wide format: rows=features, cols=algorithms
        freq_data = []
        for feat_idx in range(n_features):
            row = {"feature": feat_idx}
            for algo_name in algo_names:
                row[f"{algo_name}_freq"] = selection_freq[algo_name][feat_idx]
                row[f"{algo_name}_count"] = selection_count[algo_name][feat_idx]
            freq_data.append(row)

        freq_df = pd.DataFrame(freq_data)
        freq_path = exp_dir / "selection_frequency.csv"
        freq_df.to_csv(freq_path, index=False)

        # Long format for faceted plotting (ggplot2, seaborn, plotly)
        long_data = []
        for algo_name in algo_names:
            for feat_idx in range(n_features):
                long_data.append({
                    "feature": feat_idx,
                    "algo": algo_name,
                    "freq": selection_freq[algo_name][feat_idx],
                    "count": selection_count[algo_name][feat_idx],
                    "n_iter": n_iter,
                })
        long_df = pd.DataFrame(long_data)
        long_path = exp_dir / "selection_frequency_long.csv"
        long_df.to_csv(long_path, index=False)

        # Top features summary (sorted by avg frequency across algorithms)
        avg_freq = np.mean([selection_freq[a] for a in algo_names], axis=0)
        top_n = min(50, n_features)
        top_indices = np.argsort(avg_freq)[::-1][:top_n]

        top_data = []
        for rank, feat_idx in enumerate(top_indices, 1):
            row = {"rank": rank, "feature": feat_idx, "avg_freq": avg_freq[feat_idx]}
            for algo_name in algo_names:
                row[f"{algo_name}_freq"] = selection_freq[algo_name][feat_idx]
                row[f"{algo_name}_count"] = selection_count[algo_name][feat_idx]
            top_data.append(row)

        top_df = pd.DataFrame(top_data)
        top_path = exp_dir / "selection_frequency_top50.csv"
        top_df.to_csv(top_path, index=False)

        print(f"[realdata] Selection frequency saved:")
        print(f"  - {freq_path} (wide format)")
        print(f"  - {long_path} (long format for faceting)")
        print(f"  - {top_path} (top 50 features by avg frequency)")

    else:
        # Single algorithm mode
        selection_freq = result.get("selection_freq")
        selection_count = result.get("selection_count")
        n_iter = config.get("n_iter", 100)
        algo_name = config.get("algo", "unknown")

        if selection_freq is None:
            return

        n_features = len(selection_freq)

        # Wide format
        freq_data = []
        for feat_idx in range(n_features):
            freq_data.append({
                "feature": feat_idx,
                "freq": selection_freq[feat_idx],
                "count": selection_count[feat_idx],
                "n_iter": n_iter,
            })
        freq_df = pd.DataFrame(freq_data)
        freq_path = exp_dir / "selection_frequency.csv"
        freq_df.to_csv(freq_path, index=False)

        # Top features
        top_n = min(50, n_features)
        top_indices = np.argsort(selection_freq)[::-1][:top_n]
        top_data = []
        for rank, feat_idx in enumerate(top_indices, 1):
            top_data.append({
                "rank": rank,
                "feature": feat_idx,
                "freq": selection_freq[feat_idx],
                "count": selection_count[feat_idx],
                "n_iter": n_iter,
            })
        top_df = pd.DataFrame(top_data)
        top_path = exp_dir / "selection_frequency_top50.csv"
        top_df.to_csv(top_path, index=False)

        print(f"[realdata] Selection frequency saved:")
        print(f"  - {freq_path}")
        print(f"  - {top_path} (top 50)")


def save_results(result, exp_dir, config, is_comparison=False):
    """Save experiment results to files."""
    if is_comparison:
        for algo_name, df in result["results"].items():
            raw_path = exp_dir / f"raw_{algo_name}.csv"
            df.to_csv(raw_path, index=False)

            features_path = exp_dir / f"selected_features_{algo_name}.json"
            features_data = []
            if "selected_features" in df.columns:
                features_data = df["selected_features"].tolist()
            with open(features_path, "w") as f:
                json.dump(features_data, f, indent=2)

        summary_path = exp_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(_make_json_serializable(result["summaries"]), f, indent=2)

        # Save selection frequency for visualization
        save_selection_frequency(result, exp_dir, config, is_comparison=True)

    else:
        raw_path = exp_dir / "raw.csv"
        result["results"].to_csv(raw_path, index=False)

        features_path = exp_dir / "selected_features.json"
        features_data = []
        if "selected_features" in result["results"].columns:
            features_data = result["results"]["selected_features"].tolist()
        with open(features_path, "w") as f:
            json.dump(features_data, f, indent=2)

        summary_path = exp_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(_make_json_serializable(result["summary"]), f, indent=2)

        # Save selection frequency for visualization
        save_selection_frequency(result, exp_dir, config, is_comparison=False)

    print(f"[realdata] Results saved to {exp_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # Load from config file if provided
    if args.config:
        config = load_config(args.config)
        if "dataset" not in config:
            print("[realdata] ERROR: Config file must specify 'dataset'")
            return 1
        if "compare_models" not in config and "algo" not in config:
            print("[realdata] ERROR: Config file must specify 'compare_models' or 'algo'")
            return 1

        experiment_name = config.get("experiment", f"realdata_{config['dataset']}")
        algo_configs = config.get("compare_models", [])
        if algo_configs:
            algo_names = [a["algo"] for a in algo_configs]
            is_comparison = len(algo_names) > 1
            algo_params_map = {a["algo"]: a.get("params", {}) for a in algo_configs}
        else:
            algo_names = [config["algo"]]
            is_comparison = False
            algo_params_map = {config["algo"]: config.get("algo_params", {})}

        mode = config.get("mode", "random")
        window_size = config.get("window_size", 120)
        n_iter = config.get("n_iter", 100)
        train_ratio = config.get("train_ratio", 0.8)
        seed_base = config.get("seed_base", 42)
        cv_folds = config.get("cv_folds", 10)
        output_dir = config.get("output_dir", "experiments/results/realdata")
        preprocess = config.get("preprocess", None)

        config_for_save = {
            "experiment": experiment_name,
            "dataset": config["dataset"],
            "algo": ",".join(algo_names) if is_comparison else algo_names[0],
            "mode": mode,
            "window_size": window_size,
            "n_iter": n_iter,
            "train_ratio": train_ratio,
            "seed_base": seed_base,
            "cv_folds": cv_folds,
            "output_dir": output_dir,
            "compare_models": algo_configs,
        }
        if preprocess:
            config_for_save["preprocess"] = preprocess
        config_for_save.update(config)

        dataset = config["dataset"]
        dry_run = args.dry_run

    else:
        if args.compare:
            algo_names = [a.strip() for a in args.compare.split(",")]
            is_comparison = len(algo_names) > 1
            experiment_name = f"{args.dataset}_{args.mode}_{'-'.join(algo_names)}"
        elif args.algo:
            algo_names = [args.algo]
            is_comparison = False
            experiment_name = f"{args.dataset}_{args.mode}_{args.algo}"
        else:
            print("[realdata] ERROR: Must specify either --algo/--compare or --config")
            return 1

        config_for_save = {
            "experiment": experiment_name,
            "dataset": args.dataset,
            "algo": algo_names[0] if not is_comparison else ",".join(algo_names),
            "mode": args.mode,
            "window_size": args.window_size,
            "n_iter": args.n_iter,
            "train_ratio": args.train_ratio,
            "seed_base": args.random_seed,
            "cv_folds": 10,
            "output_dir": args.output_dir,
        }
        algo_params_map = {}
        dataset = args.dataset
        mode = args.mode
        window_size = args.window_size
        n_iter = args.n_iter
        train_ratio = args.train_ratio
        seed_base = args.random_seed
        cv_folds = 10
        output_dir = args.output_dir
        preprocess = None
        dry_run = args.dry_run

    if dry_run:
        print("[realdata] Dry-run mode - validation only")
        print(f"[realdata] Config: {config_for_save}")
        if algo_params_map:
            print(f"[realdata] Algorithm params: {algo_params_map}")
        return 0

    try:
        exp_dir = generate_experiment_dir(config_for_save)
        print(f"[realdata] Output directory: {exp_dir}")
        save_config(config_for_save, exp_dir)

        if mode == "rolling":
            if is_comparison:
                print("[realdata] Comparison mode not supported for rolling window")
                return 1
            result = run_experiment_rolling(
                dataset_name=dataset,
                algo_name=algo_names[0],
                algo_params=algo_params_map.get(algo_names[0], {}),
                window_size=window_size,
                seed_base=seed_base,
            )
        else:  # random mode
            if is_comparison:
                result = run_comparison_random(
                    dataset_name=dataset,
                    algo_names=algo_names,
                    algo_params_map=algo_params_map,
                    n_iter=n_iter,
                    train_ratio=train_ratio,
                    seed_base=seed_base,
                    cv_folds=cv_folds,
                    preprocess=preprocess,
                )
            else:
                result = run_experiment_random(
                    dataset_name=dataset,
                    algo_name=algo_names[0],
                    algo_params=algo_params_map.get(algo_names[0], {}),
                    n_iter=n_iter,
                    train_ratio=train_ratio,
                    seed_base=seed_base,
                    cv_folds=cv_folds,
                    preprocess=preprocess,
                )

        save_results(result, exp_dir, config_for_save, is_comparison=is_comparison)
        generate_report(result, exp_dir, config_for_save, is_comparison=is_comparison)

        print(f"\n[realdata] Completed!")
        print(f"[realdata] Results: {exp_dir}")
        return 0

    except Exception as e:
        print(f"[realdata] FATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
