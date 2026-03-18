"""
GLM family simulation experiments for XLasso.

Evaluates the performance of group constraint and adaptive weighting
across all GLM families:
- Gaussian (linear regression)
- Binomial (logistic regression)
- Poisson (count regression)
- Multinomial (multi-class classification)
- Cox proportional hazards (survival analysis)

Each family compares the same four configurations as the linear experiment:
1. Baseline (no constraint, no adaptive)
2. Adaptive only
3. Grouping only
4. Full XLasso (both)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional, Any

from unilasso.uni_lasso import cv_uni
from experiments.base_experiment import BaseSimulationExperiment, compute_variable_selection_metrics
from experiments import visualization

# Data generators
from unilasso.utils import simulate_gaussian_data, simulate_binomial_data, simulate_cox_data
from data_generators import simulate_poisson_data, simulate_multinomial_data


class GLMExperiment(BaseSimulationExperiment):
    """
    GLM experiment for a specific family.
    """

    def __init__(
        self,
        family: str = 'gaussian',
        n_repeats: int = 20,
        test_size: float = 0.3,
        random_seed: int = 42,
        n: int = 500,
        p: int = 100,
        sparsity: int = 10,
        correlation: str = 'block',
        rho: float = 0.7
    ):
        """
        Initialize experiment.

        Parameters
        ----------
        family : str
            GLM family: 'gaussian', 'binomial', 'poisson', 'multinomial', 'cox'
        n_repeats : int
            Number of repetitions.
        test_size : float
            Test fraction.
        random_seed : int
            Base random seed.
        n : int
            Number of samples.
        p : int
            Number of features.
        sparsity : int
            Number of true non-zero coefficients.
        correlation : str
            Correlation structure.
        rho : float
            Correlation coefficient.
        """
        super().__init__(n_repeats=n_repeats, test_size=test_size, random_seed=random_seed)
        self.family = family
        self.n = n
        self.p = p
        self.sparsity = sparsity
        self.correlation = correlation
        self.rho = rho

    def generate_data(self, repeat_seed: int):
        """Generate data for one repetition with train/test split."""
        n = self.n
        p = self.p
        sparsity = self.sparsity
        rho = self.rho

        if self.family == 'gaussian':
            # Use block correlation structure
            from data_generators import generate_highdim_correlated_data
            X, y, beta_true, _ = generate_highdim_correlated_data(
                n=n, p=p, sparsity=sparsity, correlation=rho, snr_level='medium', seed=repeat_seed
            )
        elif self.family == 'binomial':
            # Generate with block correlation
            X = self._generate_correlated_features(n, p, rho, repeat_seed)
            beta_true = np.zeros(p)
            nonzero_indices = np.random.RandomState(repeat_seed).choice(p, sparsity, replace=False)
            beta_vals = np.random.RandomState(repeat_seed).uniform(-2, 2, sparsity)
            beta_true[nonzero_indices] = beta_vals
            eta = X @ beta_true
            # Add intercept
            eta = eta - eta.mean()
            p = 1 / (1 + np.exp(-eta))
            y = np.random.RandomState(repeat_seed).binomial(1, p)
        elif self.family == 'poisson':
            X, y, beta_true = simulate_poisson_data(
                n=n, p=p, sparsity=sparsity, correlation=self.correlation, rho=rho,
                seed=repeat_seed
            )
        elif self.family == 'multinomial':
            X, y, B_true = simulate_multinomial_data(
                n=n, p=p, n_classes=3, sparsity_per_class=sparsity//3,
                correlation=self.correlation, rho=rho, seed=repeat_seed
            )
            beta_true = B_true  # For multinomial, this is (p, n_classes)
        elif self.family == 'cox':
            # Use simulate_cox_data from utils
            beta_true = np.zeros(p)
            nonzero_indices = np.random.RandomState(repeat_seed).choice(p, sparsity, replace=False)
            beta_vals = np.random.RandomState(repeat_seed).uniform(0.3, 1.5, sparsity)
            beta_vals = beta_vals * np.random.RandomState(repeat_seed).choice([-1, 1], sparsity)
            beta_true[nonzero_indices] = beta_vals
            X = self._generate_correlated_features(n, p, rho, repeat_seed)
            # Use the existing simulate_cox_data with precomputed beta
            X, y, status = simulate_cox_data(n=n, p=p, beta=beta_true, seed=repeat_seed)
            # Cox returns y as (time, status), combine for our API
            # For evaluation, we still need beta_true
            y = (y, status)
        else:
            raise ValueError(f"Unknown family: {self.family}")

        # Train/test split
        if self.family != 'cox':
            X_train, X_test, y_train, y_test = self._split_data(X, y, repeat_seed)
        else:
            # Cox y is (time, status), need special handling
            X_train, X_test, y_train, y_test = self._split_data_cox(X, y, repeat_seed)

        return X_train, y_train, X_test, y_test, beta_true

    def _generate_correlated_features(
        self, n: int, p: int, rho: float, seed: int
    ) -> np.ndarray:
        """Generate correlated features with block structure."""
        rs = np.random.RandomState(seed)
        if self.correlation == 'block':
            block_size = 10
            n_blocks = p // block_size
            cov_blocks = []
            for _ in range(n_blocks):
                block = np.ones((block_size, block_size)) * rho
                np.fill_diagonal(block, 1.0)
                cov_blocks.append(block)
            cov_matrix = np.block([[cov_blocks[i] if i == j else np.zeros((block_size, block_size))
                                    for j in range(n_blocks)] for i in range(n_blocks)])
            if p % block_size != 0:
                remaining = p % block_size
                remaining_cov = np.eye(remaining)
                cov_matrix = np.block([[cov_matrix, np.zeros((cov_matrix.shape[0], remaining))],
                                       [np.zeros((remaining, cov_matrix.shape[1])), remaining_cov]])
        elif self.correlation == 'ar1':
            cov_matrix = np.power(rho, np.abs(np.arange(p)[:, None] - np.arange(p)[None, :]))
        else:
            cov_matrix = np.eye(p)

        return rs.multivariate_normal(np.zeros(p), cov_matrix, size=n)

    def _split_data(self, X: np.ndarray, y: np.ndarray, seed: int) -> Tuple:
        """Split into train and test."""
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=self.test_size, random_state=seed)

    def _split_data_cox(self, X: np.ndarray, y, seed: int) -> Tuple:
        """Split Cox data where y is (time, status)."""
        from sklearn.model_selection import train_test_split
        time, status = y
        idx = np.arange(len(X))
        idx_train, idx_test = train_test_split(idx, test_size=self.test_size, random_state=seed)
        X_train = X[idx_train]
        X_test = X[idx_test]
        y_train = (time[idx_train], status[idx_train])
        y_test = (time[idx_test], status[idx_test])
        return X_train, X_test, y_train, y_test

    def get_model_configurations(self) -> Dict[str, dict]:
        """Return the four configurations to compare."""
        return {
            'Baseline': {
                'enable_group_constraint': False,
                'adaptive_weighting': False
            },
            'Adaptive only': {
                'enable_group_constraint': False,
                'adaptive_weighting': True
            },
            'Grouping only': {
                'enable_group_constraint': True,
                'adaptive_weighting': False
            },
            'Full XLasso': {
                'enable_group_constraint': True,
                'adaptive_weighting': True
            },
        }

    def fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        enable_group_constraint: bool,
        adaptive_weighting: bool
    ):
        """Fit model with cross-validation."""
        import time
        start_time = time.time()

        if self.family == 'cox':
            # y_train is (time, status)
            model = cv_uni(
                X_train, y_train,
                family='cox',
                enable_group_constraint=enable_group_constraint,
                adaptive_weighting=adaptive_weighting,
                n_lmdas=50,
                n_folds=5,
                verbose=False,
                seed=None
            )
        else:
            model = cv_uni(
                X_train, y_train,
                family=self.family,
                enable_group_constraint=enable_group_constraint,
                adaptive_weighting=adaptive_weighting,
                n_lmdas=50,
                n_folds=5,
                verbose=False,
                seed=None
            )

        runtime = time.time() - start_time
        return model, runtime

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        beta_true: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        metrics = {}

        # Get coefficients for best lambda from CV
        coef = model.coefs[model.best_idx]
        intercept = model.intercept[model.best_idx]

        # Get selected variables
        if self.family != 'multinomial':
            selected = coef != 0
            true_nonzero = beta_true != 0
        else:
            # Multinomial: handle different shapes
            if coef.ndim == 1 and beta_true.ndim == 2:
                # beta_true is (p, K), check if coef size matches p or p*K
                K = beta_true.shape[1]
                p = beta_true.shape[0]
                if coef.size == p * K:
                    coef = coef.reshape(p, K)
                # else: if size is already p, keep as 1D - this means model
                # has different parameterization than we thought
            # Multinomial: feature is selected if any coefficient is non-zero
            if coef.ndim == 1:
                selected = coef != 0
            else:
                selected = np.any(coef != 0, axis=1)
            if beta_true.ndim == 1:
                true_nonzero = beta_true != 0
            else:
                true_nonzero = np.any(beta_true != 0, axis=1)

        # Variable selection metrics
        vs_metrics = compute_variable_selection_metrics(selected, true_nonzero)
        metrics.update(vs_metrics)
        metrics['n_selected'] = np.sum(selected)

        # Prediction metrics by family
        if self.family == 'gaussian':
            y_pred = X_test @ coef + intercept
            mse = np.mean((y_test - y_pred) ** 2)
            metrics['mse'] = mse
            metrics['rmse'] = np.sqrt(mse)

        elif self.family == 'binomial':
            eta = X_test @ coef + intercept
            # sigmoid
            eta = np.clip(eta, -10, 10)
            y_pred_proba = 1 / (1 + np.exp(-eta))
            # Ensure in [0, 1]
            y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
            # Deviance: -2 log likelihood
            deviance = -2 * np.mean(y_test * np.log(y_pred_proba) + (1 - y_test) * np.log(1 - y_pred_proba))
            # Accuracy
            y_pred = (y_pred_proba >= 0.5).astype(int)
            accuracy = np.mean(y_pred == y_test)
            # AUC
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = np.nan
            metrics['deviance'] = deviance
            metrics['accuracy'] = accuracy
            metrics['auc'] = auc

        elif self.family == 'poisson':
            eta = X_test @ coef + intercept
            eta = np.clip(eta, -10, 10)
            lambda_pred = np.exp(eta)
            # Deviance for Poisson
            # D = 2 * sum(y_test log(y_test / λ) - (y_test - λ))
            # Where y_test=0, term is 2λ
            deviance = 0.0
            for yi, lam in zip(y_test, lambda_pred):
                if yi == 0:
                    deviance += 2 * lam
                else:
                    deviance += 2 * (yi * np.log(yi / lam) - (yi - lam))
            deviance = deviance / len(y_test)
            rmse = np.sqrt(np.mean((y_test - lambda_pred) ** 2))
            metrics['deviance'] = deviance
            metrics['rmse'] = rmse

        elif self.family == 'multinomial':
            # Compute probabilities manually
            eta = X_test @ coef + intercept
            # Softmax with numerical stability
            if eta.ndim > 1 and eta.shape[1] > 1:
                eta = eta - np.max(eta, axis=1, keepdims=True)
                exp_eta = np.exp(eta)
                y_pred_proba = exp_eta / np.sum(exp_eta, axis=1, keepdims=True)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                # Binary case or single dimension case
                y_pred_proba = 1 / (1 + np.exp(-eta))
                y_pred = (y_pred_proba >= 0.5).astype(int)
                y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
            accuracy = np.mean(y_pred == y_test)
            # Multi-class AUC (one-vs-rest)
            from sklearn.metrics import roc_auc_score
            try:
                if len(np.unique(y_test)) > 2:
                    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                else:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                auc = np.nan
            metrics['accuracy'] = accuracy
            metrics['auc'] = auc

        elif self.family == 'cox':
            # Concordance index
            from sksurv.metrics import concordance_index_ipcw
            from sklearn.model_selection import train_test_split
            # y_test is (time, status)
            time_test, status_test = y_test
            # Compute linear predictor
            risk_score = X_test @ coef
            # Need training data to estimate survival function for C-index
            # Just use concordance from lifelines directly
            from lifelines.utils import concordance_index
            try:
                cindex = concordance_index(time_test, -risk_score, status_test)
            except:
                cindex = np.nan
            metrics['cindex'] = cindex

        return metrics

    def plot_results(self, output_dir: str):
        """Generate plots for this experiment."""
        os.makedirs(output_dir, exist_ok=True)

        raw_df = self.get_raw_results()

        # Select appropriate metric based on family
        if self.family == 'gaussian':
            metric, ylabel, higher_is_better = 'mse', 'MSE (lower is better)', False
        elif self.family == 'binomial':
            metric, ylabel, higher_is_better = 'auc', 'AUC (higher is better)', True
        elif self.family == 'poisson':
            metric, ylabel, higher_is_better = 'deviance', 'Deviance (lower is better)', False
        elif self.family == 'multinomial':
            metric, ylabel, higher_is_better = 'accuracy', 'Accuracy (higher is better)', True
        elif self.family == 'cox':
            metric, ylabel, higher_is_better = 'cindex', 'C-index (higher is better)', True
        else:
            metric, ylabel, higher_is_better = 'f1', 'F1 (higher is better)', True

        # Boxplot for main metric
        fig = visualization.plot_method_comparison_boxplot(
            raw_df, metric,
            title=f'{self.family.capitalize()} - {metric}',
            ylabel=ylabel,
            ascending=not higher_is_better
        )
        visualization.save_figure(
            fig, os.path.join(output_dir, f'{self.family}_{metric}_boxplot')
        )

        # Boxplot for F1
        fig = visualization.plot_method_comparison_boxplot(
            raw_df, 'f1',
            title=f'{self.family.capitalize()} - F1 Score (Variable Selection)',
            ylabel='F1 (higher is better)',
            ascending=False
        )
        visualization.save_figure(
            fig, os.path.join(output_dir, f'{self.family}_f1_boxplot')
        )

        # Summary bar chart for F1
        agg_df = self.aggregate_results()
        mean_f1 = agg_df[('f1', 'mean')]
        std_f1 = agg_df[('f1', 'std')]
        fig = visualization.plot_summary_bar_chart(
            mean_f1, std_f1,
            title=f'F1 Comparison - {self.family.capitalize()}',
            ylabel='F1 (higher is better)'
        )
        visualization.save_figure(
            fig, os.path.join(output_dir, f'{self.family}_f1_summary')
        )

        print(f"Plots saved to {output_dir}")


def run_all_glm_experiments(
    output_dir: str = 'experiments/results',
    n_repeats_override: Optional[int] = None,
    save_plots: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run GLM experiments for all families.

    Parameters
    ----------
    output_dir : str
        Directory to save results.
    n_repeats_override : int, optional
        Override number of repetitions.
    save_plots : bool
        Whether to generate plots.

    Returns
    -------
    all_results : Dict[str, pd.DataFrame]
        Aggregated results for each family.
    """
    families = ['gaussian', 'binomial', 'poisson', 'multinomial', 'cox']

    all_results = {}
    os.makedirs(output_dir, exist_ok=True)

    for family in families:
        print(f"\n{'='*60}")
        print(f"Running GLM experiment: family = {family}")
        print(f"{'='*60}")

        n_reps = 20 if n_repeats_override is None else n_repeats_override
        exp = GLMExperiment(
            family=family,
            n_repeats=n_reps,
            random_seed=42,
            correlation='block',
            rho=0.7
        )
        exp.run()
        agg = exp.aggregate_results()
        all_results[family] = agg
        exp.save_results(os.path.join(output_dir, f'{family}_results.csv'))
        if save_plots:
            exp.plot_results(output_dir)

    # Create summary table
    summary_rows = []
    for family, df in all_results.items():
        for method in df.index:
            row = {
                'family': family,
                'method': method,
                'f1_mean': df.loc[method, ('f1', 'mean')],
                'f1_std': df.loc[method, ('f1', 'std')],
                'tpr_mean': df.loc[method, ('tpr', 'mean')],
                'fpr_mean': df.loc[method, ('fpr', 'mean')],
            }
            # Add family-specific metric
            if family == 'gaussian':
                row['perf_mean'] = df.loc[method, ('mse', 'mean')]
                row['perf_std'] = df.loc[method, ('mse', 'std')]
            elif family == 'binomial':
                row['perf_mean'] = df.loc[method, ('auc', 'mean')]
                row['perf_std'] = df.loc[method, ('auc', 'std')]
            elif family == 'poisson':
                row['perf_mean'] = df.loc[method, ('deviance', 'mean')]
                row['perf_std'] = df.loc[method, ('deviance', 'std')]
            elif family == 'multinomial':
                row['perf_mean'] = df.loc[method, ('accuracy', 'mean')]
                row['perf_std'] = df.loc[method, ('accuracy', 'std')]
            elif family == 'cox':
                row['perf_mean'] = df.loc[method, ('cindex', 'mean')]
                row['perf_std'] = df.loc[method, ('cindex', 'std')]
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'glm_summary.csv'), index=False)
    print(f"\nAll GLM experiments completed. Summary saved to {output_dir}/glm_summary.csv")

    return all_results
