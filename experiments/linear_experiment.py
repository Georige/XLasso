"""
Linear Gaussian simulation experiment for XLasso.

Systematically evaluates the effect of:
- Group constraint (grouped selection for correlated features)
- Adaptive weighting (penalty adjustment based on univariate significance)

Compares four configurations:
1. No constraint, no adaptive weighting (baseline, similar to standard Lasso)
2. No constraint, with adaptive weighting
3. With constraint, no adaptive weighting
4. With constraint, with adaptive weighting (full XLasso)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional, Any

from unilasso.uni_lasso import cv_uni
from experiments.base_experiment import BaseSimulationExperiment, compute_variable_selection_metrics, compute_group_accuracy
from experiments import visualization

# Data generators
from data_generators import (
    generate_ar1_data,
    generate_highdim_correlated_data,
    generate_sign_inconsistent_data,
    generate_factor_model_data
)


class LinearGaussianExperiment(BaseSimulationExperiment):
    """
    Linear Gaussian simulation experiment comparing different XLasso configurations.
    """

    def __init__(
        self,
        scenario: str = 'independent',
        n_repeats: int = 30,
        test_size: float = 0.3,
        random_seed: int = 42,
        n: int = None,
        p: int = None,
        sparsity: int = None,
        rho: float = None,
        snr_level: str = 'medium'
    ):
        """
        Initialize experiment for a specific scenario.

        Parameters
        ----------
        scenario : str
            Which scenario to run:
            - 'independent': independent features
            - 'ar1': AR(1) correlation
            - 'block': block structure correlation
            - 'block_false': block structure with only one true variable per block
            - 'sign_inconsistent': two highly correlated true variables with opposite signs
            - 'factor': factor model correlation
            - 'highdim_block_low': high-dimensional low SNR block structure
            - 'highdim_block_medium': high-dimensional medium SNR block structure
            - 'highdim_block_high': high-dimensional high SNR block structure
        n_repeats : int
            Number of repetitions.
        test_size : float
            Test fraction.
        random_seed : int
            Base random seed.
        n : int, optional
            Override number of samples.
        p : int, optional
            Override number of features.
        sparsity : int, optional
            Override sparsity.
        rho : float, optional
            Override correlation coefficient.
        snr_level : str
            SNR level for high-dimensional case.
        """
        super().__init__(n_repeats=n_repeats, test_size=test_size, random_seed=random_seed)
        self.scenario = scenario
        self._set_scenario_parameters(n, p, sparsity, rho, snr_level)

    def _set_scenario_parameters(
        self,
        n_override: int,
        p_override: int,
        sparsity_override: int,
        rho_override: float,
        snr_level: str
    ):
        """Set parameters based on scenario."""
        # Default parameters by scenario
        params = {
            'independent': {'n': 500, 'p': 100, 'sparsity': 10, 'rho': 0.0, 'correlation': 'independent'},
            'ar1': {'n': 500, 'p': 100, 'sparsity': 10, 'rho': 0.7, 'correlation': 'ar1'},
            'block': {'n': 500, 'p': 100, 'sparsity': 10, 'rho': 0.7, 'correlation': 'block'},
            'block_false': {'n': 500, 'p': 100, 'sparsity': 5, 'rho': 0.7, 'correlation': 'block'},
            'sign_inconsistent': {'n': 300, 'p': 20, 'sparsity': 2, 'rho': 0.95, 'correlation': 'sign_inconsistent'},
            'factor': {'n': 300, 'p': 50, 'sparsity': 10, 'rho': None, 'correlation': 'factor'},
            'highdim_block': {'n': 300, 'p': 1000, 'sparsity': 20, 'rho': 0.5, 'correlation': 'block'},
        }

        self.params = params[self.scenario].copy()

        # Apply overrides
        if n_override is not None:
            self.params['n'] = n_override
        if p_override is not None:
            self.params['p'] = p_override
        if sparsity_override is not None:
            self.params['sparsity'] = sparsity_override
        if rho_override is not None:
            self.params['rho'] = rho_override

        self.snr_level = snr_level

    def generate_data(self, repeat_seed: int):
        """Generate data for one repetition with train/test split."""
        corr = self.params['correlation']
        n = self.params['n']
        p = self.params['p']
        sparsity = self.params['sparsity']
        rho = self.params['rho']

        if corr == 'independent':
            # Independent features - use AR1 with rho=0
            X, y, beta_true = generate_ar1_data(
                n=n, p=p, rho=0.0, sparsity=sparsity, seed=repeat_seed
            )
        elif corr == 'ar1':
            X, y, beta_true = generate_ar1_data(
                n=n, p=p, rho=rho, sparsity=sparsity, seed=repeat_seed
            )
        elif corr == 'block' or corr == 'block_false':
            # Block structure: true variables one per block
            X, y, beta_true, snr = generate_highdim_correlated_data(
                n=n, p=p, sparsity=sparsity, correlation=rho, snr_level=self.snr_level, seed=repeat_seed
            )
        elif corr == 'sign_inconsistent':
            X, y, beta_true = generate_sign_inconsistent_data(
                n=n, p=p, seed=repeat_seed
            )
        elif corr == 'factor':
            X, y, beta_true, _ = generate_factor_model_data(
                n=n, p=p, sparsity=sparsity, seed=repeat_seed
            )
        else:
            raise ValueError(f"Unknown correlation: {corr}")

        # Train/test split
        X_train, X_test, y_train, y_test = self._split_data(X, y, repeat_seed)

        return X_train, y_train, X_test, y_test, beta_true

    def _split_data(self, X: np.ndarray, y: np.ndarray, seed: int) -> Tuple:
        """Split into train and test."""
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=seed
        )
        return X_train, X_test, y_train, y_test

    def get_model_configurations(self) -> Dict[str, dict]:
        """Return the four configurations to compare."""
        return {
            'XLasso (baseline)': {
                'enable_group_constraint': False,
                'adaptive_weighting': False
            },
            'XLasso + adaptive': {
                'enable_group_constraint': False,
                'adaptive_weighting': True
            },
            'XLasso + grouping': {
                'enable_group_constraint': True,
                'adaptive_weighting': False
            },
            'XLasso (full)': {
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
        """Fit XLasso with cross-validation."""
        import time
        start_time = time.time()

        model = cv_uni(
            X_train, y_train,
            family='gaussian',
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
        # Get coefficients for best lambda from CV
        coef = model.coefs[model.best_idx]
        intercept = model.intercept[model.best_idx]

        # Predict on test
        y_pred = X_test @ coef + intercept

        # Out-of-sample MSE
        mse = np.mean((y_test - y_pred) ** 2)

        # Coefficient MSE
        coef_mse = np.mean((coef - beta_true) ** 2)

        # Variable selection metrics
        selected = coef != 0
        true_nonzero = beta_true != 0

        vs_metrics = compute_variable_selection_metrics(selected, true_nonzero)

        # Compute grouping accuracy if groups exist in the model
        group_acc = np.nan
        if hasattr(model, 'groups') and model.groups is not None:
            group_acc = compute_group_accuracy(selected, model.groups, true_nonzero)

        result = {
            'mse': mse,
            'coef_mse': coef_mse,
            'group_acc': group_acc,
            'n_selected': np.sum(selected),
            'tpr': vs_metrics['tpr'],
            'fpr': vs_metrics['fpr'],
            'precision': vs_metrics['precision'],
            'f1': vs_metrics['f1'],
        }

        return result

    def plot_results(self, output_dir: str):
        """Generate plots for this experiment."""
        os.makedirs(output_dir, exist_ok=True)

        raw_df = self.get_raw_results()
        agg_df = self.aggregate_results()

        # Boxplot for MSE
        fig = visualization.plot_method_comparison_boxplot(
            raw_df, 'mse',
            title=f'MSE Out-of-Sample - Scenario: {self.scenario}',
            ylabel='MSE (lower is better)',
            ascending=True
        )
        visualization.save_figure(
            fig, os.path.join(output_dir, f'{self.scenario}_mse_boxplot')
        )

        # Boxplot for F1
        fig = visualization.plot_method_comparison_boxplot(
            raw_df, 'f1',
            title=f'F1 Score (Variable Selection) - Scenario: {self.scenario}',
            ylabel='F1 (higher is better)',
            ascending=False
        )
        visualization.save_figure(
            fig, os.path.join(output_dir, f'{self.scenario}_f1_boxplot')
        )

        # Boxplot for TPR
        fig = visualization.plot_method_comparison_boxplot(
            raw_df, 'tpr',
            title=f'True Positive Rate - Scenario: {self.scenario}',
            ylabel='TPR (higher is better)',
            ascending=False
        )
        visualization.save_figure(
            fig, os.path.join(output_dir, f'{self.scenario}_tpr_boxplot')
        )

        # Boxplot for FPR
        fig = visualization.plot_method_comparison_boxplot(
            raw_df, 'fpr',
            title=f'False Positive Rate - Scenario: {self.scenario}',
            ylabel='FPR (lower is better)',
            ascending=True
        )
        visualization.save_figure(
            fig, os.path.join(output_dir, f'{self.scenario}_fpr_boxplot')
        )

        # Bar chart with error bars for F1
        mean_f1 = agg_df[('f1', 'mean')]
        std_f1 = agg_df[('f1', 'std')]
        fig = visualization.plot_summary_bar_chart(
            mean_f1, std_f1,
            title=f'F1 Score Comparison - {self.scenario}',
            ylabel='F1 (higher is better)'
        )
        visualization.save_figure(
            fig, os.path.join(output_dir, f'{self.scenario}_f1_summary')
        )

        print(f"Plots saved to {output_dir}")


def run_all_linear_scenarios(
    output_dir: str = 'experiments/results',
    n_repeats_override: Optional[int] = None,
    save_plots: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run all linear Gaussian scenarios and collect results.

    Parameters
    ----------
    output_dir : str
        Directory to save results.
    n_repeats_override : int, optional
        Override number of repetitions (use smaller for testing).
    save_plots : bool
        Whether to generate and save plots.

    Returns
    -------
    all_results : Dict[str, pd.DataFrame]
        Aggregated results for each scenario.
    """
    scenarios = [
        'independent',
        'ar1',
        'block',
        'block_false',
        'sign_inconsistent',
        'factor',
        'highdim_block'
    ]

    snr_levels = {
        'highdim_block_low': 'low',
        'highdim_block_medium': 'medium',
        'highdim_block_high': 'high'
    }

    all_results = {}

    os.makedirs(output_dir, exist_ok=True)

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario}")
        print(f"{'='*60}")

        if scenario == 'highdim_block':
            # Run three SNR levels
            for name, snr in snr_levels.items():
                n_reps = 20 if n_repeats_override is None else n_repeats_override
                exp = LinearGaussianExperiment(
                    scenario='highdim_block',
                    n_repeats=n_reps,
                    snr_level=snr,
                    random_seed=42
                )
                exp.run()
                agg = exp.aggregate_results()
                all_results[name] = agg
                exp.save_results(os.path.join(output_dir, f'{name}_results.csv'))
                if save_plots:
                    exp.plot_results(output_dir)
        else:
            n_reps = 30 if n_repeats_override is None else n_repeats_override
            exp = LinearGaussianExperiment(
                scenario=scenario,
                n_repeats=n_reps,
                random_seed=42
            )
            exp.run()
            agg = exp.aggregate_results()
            all_results[scenario] = agg
            exp.save_results(os.path.join(output_dir, f'{scenario}_results.csv'))
            if save_plots:
                exp.plot_results(output_dir)

    # Save combined summary
    summary = []
    for scenario, df in all_results.items():
        for method in df.index:
            row = {
                'scenario': scenario,
                'method': method,
                'f1_mean': df.loc[method, ('f1', 'mean')],
                'f1_std': df.loc[method, ('f1', 'std')],
                'tpr_mean': df.loc[method, ('tpr', 'mean')],
                'tpr_std': df.loc[method, ('tpr', 'std')],
                'fpr_mean': df.loc[method, ('fpr', 'mean')],
                'fpr_std': df.loc[method, ('fpr', 'std')],
                'mse_mean': df.loc[method, ('mse', 'mean')],
                'mse_std': df.loc[method, ('mse', 'std')],
            }
            summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'linear_summary.csv'), index=False)
    print(f"\nAll linear experiments completed. Summary saved to {output_dir}/linear_summary.csv")

    return all_results
