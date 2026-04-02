#!/usr/bin/env python3
"""
BAFL Coefficient Path Plotting Script
=====================================
Command-line tool to generate publication-quality BAFL coefficient path plots.

Usage:
    python factory/plot_bafl_path.py --exp 6 --seed 42
    python factory/plot_bafl_path.py --exp 6 --seed 42 --output ./plots/bafl_path.pdf
    python factory/plot_bafl_path.py --exp 6 --seed 42 --n-alphas 200 --cv-folds 10

Examples:
    Exp6 (Decoy Trap):
        python factory/plot_bafl_path.py --exp 6 --seed 42

    Exp1 (Basic):
        python factory/plot_bafl_path.py --exp 1 --seed 42

    Exp4 (Twin):
        python factory/plot_bafl_path.py --exp 4 --seed 42
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.model_selection import KFold

# Add XLasso root to path
xlasso_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, xlasso_root)

# Add experiments directory for viz imports
sys.path.insert(0, os.path.join(xlasso_root, 'experiments'))
from viz import plot_bafl_coefficient_path, plot_bafl_cv_error_path
from modules import DataGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot BAFL coefficient path for synthetic experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--exp', type=int, default=6,
                        help='Experiment number (1-7, default: 6)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--n-samples', type=int, default=300,
                        help='Number of samples (default: 300)')
    parser.add_argument('--n-features', type=int, default=500,
                        help='Number of features (default: 500)')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Noise standard deviation (default: 1.0)')
    parser.add_argument('--rho', type=float, default=0.8,
                        help='Correlation parameter (default: 0.8)')
    parser.add_argument('--family', type=str, default='gaussian',
                        choices=['gaussian', 'binomial'],
                        help='Response family (default: gaussian)')
    parser.add_argument('--n-alphas', type=int, default=100,
                        help='Number of alpha points (default: 100)')
    parser.add_argument('--alpha-min-ratio', type=float, default=1e-4,
                        help='Minimum alpha as fraction of maximum (default: 1e-4)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of CV folds for 1-SE selection (default: 5)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='BAFL weight exponent gamma (default: 1.0)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: auto-generated)')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show plot interactively (default: False)')
    parser.add_argument('--no-1se', action='store_true', default=False,
                        help='Skip 1-SE optimal line (default: False)')
    parser.add_argument('--plot', type=str, default='coef',
                        choices=['coef', 'cv_error', 'both'],
                        help='Plot type: coef (coefficient path), '
                             'cv_error (CV error path), both (default: coef)')
    return parser.parse_args()


def generate_data(exp_type, n_samples, n_features, sigma, rho, family, seed):
    """Generate synthetic data using DataGenerator."""
    correlation_map = {
        1: 'experiment1',
        2: 'experiment2',
        3: 'experiment3',
        4: 'experiment4',
        5: 'experiment5',
        6: 'experiment6',
        7: 'experiment7',
    }

    data_gen = DataGenerator(random_state=seed)
    X, y, beta_true = data_gen.generate(
        n_samples=n_samples,
        n_features=n_features,
        sigma=sigma,
        correlation_type=correlation_map[exp_type],
        rho=rho,
        family=family
    )
    return X, y, beta_true


def get_decoy_indices(exp_type):
    """Get noise decoy indices for each experiment type.

    Exp6 groups: [(0,1,2), (3,4,5), (6,7,8), (12,13,14), (15,16,17)]
    - First 2 in each group are true signals (β=1.0)
    - 3rd in each group is noise decoy (β=0, correlated with group signals)
    """
    if exp_type == 6:
        return [2, 5, 8, 14, 17]  # Exp6: The Decoy Trap
    return []


def _fista_logistic_path(X, y, alphas, max_iter=2000, tol=1e-5, random_state=42):
    """FISTA logistic regression path solver for classification.

    Parameters
    ----------
    X : np.ndarray (N, p)
        Feature matrix (standardized, sign-flipped, weighted).
    y : np.ndarray (N,)
        Labels {0, 1}.
    alphas : np.ndarray
        Regularization parameter sequence.

    Returns
    -------
    coefs_path : np.ndarray (p, n_alphas)
        Coefficient path.
    """
    from scipy.special import expit

    N, p = X.shape
    n_alphas = len(alphas)
    coefs_path = np.zeros((p, n_alphas))

    # Lipschitz constant estimation via power iteration
    rng = np.random.RandomState(random_state)
    v = rng.randn(p)
    for _ in range(5):
        v = X.T @ (X @ v)
        v_norm = np.linalg.norm(v)
        if v_norm > 0:
            v = v / v_norm
    L = np.linalg.norm(X.T @ (X @ v)) / (4.0 * N)
    step_size = 1.0 / (L + 1e-8)

    theta = np.zeros(p)
    b = 0.0
    y_k = theta.copy()
    b_k = b
    t_k = 1.0

    for i, alpha in enumerate(alphas):
        for iteration in range(max_iter):
            theta_old = theta.copy()
            b_old = b

            sigma = expit(X @ y_k + b_k)
            err = sigma - y
            grad_theta = (X.T @ err) / N
            grad_b = np.mean(err)

            theta_step = y_k - step_size * grad_theta
            b_step = b_k - step_size * grad_b

            theta = np.maximum(0.0, theta_step - step_size * alpha)
            b = b_step

            t_k_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
            momentum = (t_k - 1.0) / t_k_next
            y_k = theta + momentum * (theta - theta_old)
            b_k = b + momentum * (b - b_old)

            if (np.max(np.abs(theta - theta_old)) < tol and
                    np.abs(b - b_old) < tol):
                break

        coefs_path[:, i] = theta
        t_k = 1.0
        y_k = theta.copy()
        b_k = b

    return coefs_path


def extract_bafl_path(X, y, alpha_min_ratio, n_alphas, gamma, cv_folds, seed, family='gaussian'):
    """Extract BAFL coefficient path with 1-SE selection.

    Parameters
    ----------
    family : str
        'gaussian' for regression, 'binomial' for classification.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss

    n = len(y)
    is_classification = (family == 'binomial')

    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    if is_classification:
        # For classification: use marginal correlation for signs
        y_centered = y - np.mean(y)
        marginal_corr = X_std.T @ y_centered
        signs = np.sign(marginal_corr)
        signs[signs == 0] = 1.0
    else:
        # For regression: use Ridge CV for signs
        ridge_cv = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=None)
        ridge_cv.fit(X_std, y)
        beta_ridge = ridge_cv.coef_
        signs = np.sign(beta_ridge)
        signs[signs == 0] = 1.0

    # Weights
    eps = 1e-10
    weights = 1.0 / (np.abs(signs * (X_std.T @ y) if is_classification else np.abs(beta_ridge)) + eps) ** gamma
    weights = weights / np.min(weights)

    # Adaptive space
    X_adaptive = (X_std * signs) / weights

    # Alpha grid
    y_for_alpha = y.astype(float)
    alpha_max = np.max(np.abs(X_adaptive.T @ y_for_alpha)) / len(y_for_alpha)
    alpha_min = alpha_max * alpha_min_ratio
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alphas)[::-1]

    # Coefficient path
    if is_classification:
        # Use FISTA logistic path
        coefs_path = _fista_logistic_path(
            X_adaptive, y.astype(int),
            alphas, max_iter=2000, tol=1e-5, random_state=seed
        )
    else:
        # Use lasso path for regression
        _, coefs_path, _ = lasso_path(
            X_adaptive, y,
            alphas=alphas,
            positive=True,
            max_iter=10000,
            tol=1e-4
        )

    # Convert back to original space
    coefs_original = coefs_path.T * signs / weights  # (n_alphas, p)

    # K-Fold CV for 1-SE selection
    if is_classification:
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    else:
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    error_matrix = np.full((n_alphas, cv_folds), np.inf)
    nselected_matrix = np.zeros((n_alphas, cv_folds), dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_adaptive, y if is_classification else None)):
        X_tr = X_adaptive[train_idx]
        X_va = X_adaptive[val_idx]
        y_tr = y[train_idx]
        y_va = y[val_idx]

        if is_classification:
            y_tr_binary = (y_tr > 0.5).astype(int)
            coefs_tr = _fista_logistic_path(
                X_tr, y_tr_binary, alphas, max_iter=2000, tol=1e-5, random_state=seed
            )
            # Log-loss
            from scipy.special import expit
            for i in range(n_alphas):
                prob_va = expit(X_va @ coefs_tr[:, i])
                prob_va = np.clip(prob_va, 1e-15, 1.0 - 1e-15)
                error_matrix[i, fold_idx] = log_loss(y_va, prob_va)
        else:
            y_tr_mean = np.mean(y_tr)
            y_tr_centered = y_tr - y_tr_mean
            _, coefs_tr, _ = lasso_path(
                X_tr, y_tr_centered,
                alphas=alphas,
                positive=True,
                max_iter=10000,
                tol=1e-4
            )
            preds_va = X_va @ coefs_tr + y_tr_mean
            mse_path = np.mean((y_va[:, np.newaxis] - preds_va) ** 2, axis=0)
            error_matrix[:, fold_idx] = mse_path

        nselected_matrix[:, fold_idx] = np.sum(coefs_tr != 0, axis=0)

    # 1-SE rule (for classification, lower log-loss is better)
    mean_error = np.mean(error_matrix, axis=1)
    std_error = np.std(error_matrix, axis=1) / np.sqrt(cv_folds)

    min_error_idx = np.argmin(mean_error)
    se_threshold = mean_error[min_error_idx] + std_error[min_error_idx]

    within_se = np.where(mean_error <= se_threshold)[0]
    mean_nselected = np.mean(nselected_matrix, axis=1)

    best_idx = within_se[np.argmin(mean_nselected[within_se])]
    if mean_nselected[best_idx] == 0:
        best_idx = min_error_idx

    # Return CV data for error path plotting
    return coefs_original, alphas, alphas[best_idx], alphas[min_error_idx], mean_error, std_error, mean_nselected


def main():
    args = parse_args()

    print("=" * 60)
    print(f"BAFL Coefficient Path Plotting")
    print(f"Experiment: Exp{args.exp}")
    print(f"Random Seed: {args.seed}")
    print("=" * 60)

    # Generate data
    print("\n[1/4] Generating data...")
    X, y, beta_true = generate_data(
        args.exp, args.n_samples, args.n_features,
        args.sigma, args.rho, args.family, args.seed
    )
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  True signals: {np.sum(beta_true != 0)}")

    true_signal_indices = np.where(beta_true != 0)[0].tolist()
    decoy_indices = get_decoy_indices(args.exp)

    # Extract path
    print("\n[2/4] Extracting BAFL coefficient path...")
    coefs, alphas, optimal_alpha, min_alpha, mean_error, std_error, mean_nselected = extract_bafl_path(
        X, y, args.alpha_min_ratio, args.n_alphas,
        args.gamma, args.cv_folds, args.seed, args.family
    )
    print(f"  Alpha range: [{alphas[-1]:.2e}, {alphas[0]:.2e}]")
    print(f"  Min MSE alpha: {min_alpha:.6e}")
    print(f"  1-SE optimal alpha: {optimal_alpha:.6e}")

    # Print statistics
    print("\n[3/4] Path statistics:")
    for i in range(0, len(alphas), len(alphas) // 5):
        n_nz = np.sum(coefs[i] != 0)
        print(f"  alpha[{i}]: {n_nz} non-zero coefficients")

    # Determine output path
    if args.output is None:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_base = os.path.join(xlasso_root, 'experiments', 'results', 'plots')
        os.makedirs(os.path.join(output_base, 'bafl_paths'), exist_ok=True)
        os.makedirs(os.path.join(output_base, 'cv_error'), exist_ok=True)
    else:
        output_base = None

    optimal = None if args.no_1se else optimal_alpha

    # Plot
    print(f"\n[4/4] Generating plot...")

    if args.plot in ['coef', 'both']:
        coef_output = args.output if output_base is None else os.path.join(output_base, 'bafl_paths', f'bafl_coef_path_exp{args.exp}_seed{args.seed}.pdf')
        fig, ax = plot_bafl_coefficient_path(
            coefs=coefs,
            alphas=alphas,
            true_signal_indices=true_signal_indices,
            decoy_indices=decoy_indices,
            optimal_alpha=optimal,
            min_alpha=min_alpha,
            save_path=coef_output,
            show=args.show,
        )
        print(f"  Coefficient path saved to: {coef_output}")

    if args.plot in ['cv_error', 'both']:
        cv_output = args.output if output_base is None else os.path.join(output_base, 'cv_error', f'bafl_cv_error_exp{args.exp}_seed{args.seed}.pdf')
        fig, ax1, ax2 = plot_bafl_cv_error_path(
            alphas=alphas,
            mean_error=mean_error,
            std_error=std_error,
            nselected=mean_nselected,
            optimal_alpha=optimal,
            min_alpha=min_alpha,
            family=args.family,
            save_path=cv_output,
            show=args.show,
        )
        print(f"  CV error path saved to: {cv_output}")

    if args.plot == 'coef':
        output_path = coef_output
    elif args.plot == 'cv_error':
        output_path = cv_output
    else:
        output_path = coef_output  # for 'both', coef is the primary

    print(f"\nPlot saved to: {output_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    opt_coefs = coefs[np.argmax(alphas == optimal_alpha)]
    selected = np.where(opt_coefs != 0)[0]
    tp = len([x for x in selected if x in true_signal_indices])
    fp = len([x for x in selected if x not in true_signal_indices])
    print(f"1-SE optimal model:")
    print(f"  Total selected: {len(selected)}")
    print(f"  True positives: {tp}/{len(true_signal_indices)}")
    print(f"  False positives: {fp}")
    if decoy_indices:
        decoy_selected = len([x for x in selected if x in decoy_indices])
        print(f"  Decoys selected: {decoy_selected}/{len(decoy_indices)}")


if __name__ == '__main__':
    main()
