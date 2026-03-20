"""
Experiment 2: AR(1) correlated sparse regression
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
from unilasso.uni_lasso import cv_unilasso, cv_uni, extract_cv

def generate_ar1_cov(p, rho=0.8):
    """Generate AR(1) covariance matrix: Σ_ij = rho^|i-j|"""
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = rho ** abs(i - j)
    return cov

def generate_data(n=300, p=1000, sigma=1.0, rho=0.8):
    """
    Generate AR(1) correlated sparse regression data
    True coefficients: odd indices first 50 variables are non-zero, ~U(0.5, 2)
    """
    # Generate AR(1) covariance matrix
    cov = generate_ar1_cov(p, rho)
    # Generate X: multivariate normal
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)
    # Generate true coefficients
    beta = np.zeros(p)
    # Odd indices 1,3,5,...,99 (0-based: 0,2,4,...,98)
    true_indices = np.arange(0, 100, 2)
    beta[true_indices] = np.random.uniform(0.5, 2.0, size=len(true_indices))
    # Generate response
    y = X @ beta + np.random.normal(0, sigma, size=n)
    # Generate test data for evaluation
    X_test = np.random.multivariate_normal(np.zeros(p), cov, size=1000)
    y_test = X_test @ beta + np.random.normal(0, sigma, size=1000)
    return X, y, X_test, y_test, beta, true_indices

def evaluate_model(y_true, y_pred, beta_true, beta_pred, true_indices):
    """Evaluate model performance"""
    # Prediction MSE
    mse = np.mean((y_true - y_pred) ** 2)
    # Variable selection metrics
    selected = np.where(np.abs(beta_pred) > 1e-8)[0]
    true_positive = len(np.intersect1d(selected, true_indices))
    false_positive = len(selected) - true_positive
    tpr = true_positive / len(true_indices) if len(true_indices) > 0 else 0
    fdr = false_positive / len(selected) if len(selected) > 0 else 0
    f1 = 2 * (tpr * (1 - fdr)) / (tpr + (1 - fdr)) if (tpr + (1 - fdr)) > 0 else 0
    n_selected = len(selected)
    return {
        'mse': mse,
        'tpr': tpr,
        'fdr': fdr,
        'f1': f1,
        'n_selected': n_selected
    }

def run_experiment(n_repeats=10, n_train=300, p=1000, rho=0.8):
    """Run AR(1) experiment with 3 SNR scenarios"""
    scenarios = [
        ('Low SNR (σ=0.5)', 0.5),
        ('Medium SNR (σ=1.0)', 1.0),
        ('High SNR (σ=2.5)', 2.5)
    ]

    methods = [
        ('UniLasso', 'unilasso'),
        ('Soft Constraint', 'soft'),
        ('Soft + Adaptive', 'soft_adaptive'),
        ('Soft + Group', 'soft_group'),
        ('Full XLasso', 'full')
    ]

    results = []

    for scenario_name, sigma in scenarios:
        print(f"\n{'='*80}")
        print(f"Running {scenario_name}, sigma={sigma}")
        print(f"{'='*80}")

        for repeat in tqdm(range(n_repeats), desc=f"Repeats for {scenario_name}"):
            # Generate data
            X_train, y_train, X_test, y_test, beta_true, true_indices = generate_data(
                n=n_train, p=p, sigma=sigma, rho=rho
            )

            for method_name, method_key in methods:
                start_time = time.time()

                try:
                    if method_key == 'unilasso':
                        # Original UniLasso with non-negative constraint
                        cv_result = cv_unilasso(
                            X_train, y_train, family="gaussian", n_folds=5, verbose=False
                        )
                        fit = extract_cv(cv_result)
                        beta_pred = fit.coefs
                        intercept = fit.intercept
                    else:
                        # XLasso series methods
                        if method_key == 'soft':
                            adaptive = False
                            group = False
                            alpha = 0.0
                            beta_param = 1.0
                        elif method_key == 'soft_adaptive':
                            adaptive = True
                            group = False
                            alpha = 1.0
                            beta_param = 1.0
                        elif method_key == 'soft_group':
                            adaptive = False
                            group = True
                            alpha = 0.0
                            beta_param = 1.0
                        else:  # full
                            adaptive = True
                            group = True
                            alpha = 1.0
                            beta_param = 1.0

                        cv_result = cv_uni(
                            X_train, y_train, family="gaussian", n_folds=5,
                            adaptive_weighting=adaptive,
                            enable_group_constraint=group,
                            alpha=alpha,
                            beta=beta_param,
                            backend="numba",
                            verbose=False
                        )
                        fit = extract_cv(cv_result)
                        beta_pred = fit.coefs
                        intercept = fit.intercept

                    # Predict
                    y_pred = X_test @ beta_pred + intercept

                    # Evaluate
                    metrics = evaluate_model(y_test, y_pred, beta_true, beta_pred, true_indices)
                    run_time = time.time() - start_time

                    results.append({
                        'scenario': scenario_name,
                        'sigma': sigma,
                        'repeat': repeat,
                        'method': method_name,
                        'mse': metrics['mse'],
                        'tpr': metrics['tpr'],
                        'fdr': metrics['fdr'],
                        'f1': metrics['f1'],
                        'n_selected': metrics['n_selected'],
                        'time': run_time
                    })

                except Exception as e:
                    print(f"Error in {method_name}, repeat {repeat}: {str(e)}")
                    continue

    # Convert to DataFrame
    df = pd.DataFrame(results)
    # Save raw results
    df.to_csv('lab/result/exp_002/ar1_experiment_raw.csv', index=False)

    # Generate summary table
    summary = df.groupby(['scenario', 'method']).agg({
        'mse': ['mean', 'std'],
        'tpr': ['mean', 'std'],
        'fdr': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'n_selected': ['mean', 'std'],
        'time': ['mean', 'std']
    }).round(3)

    summary.to_csv('lab/result/exp_002/ar1_experiment_summary.csv')
    print("\nExperiment completed! Results saved to lab/result/exp_002/")

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(summary)

    return df, summary

if __name__ == "__main__":
    import os
    os.makedirs('lab/result/exp_002', exist_ok=True)
    df, summary = run_experiment(n_repeats=10)
