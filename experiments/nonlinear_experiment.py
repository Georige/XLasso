"""
Nonlinear simulation experiments for XLasso.

Evaluates the performance of group constraint and adaptive weighting
with different univariate models:
- linear: linear basis
- spline: B-spline basis for non-linear effects
- tree: decision tree univariate model

Two-dimensional comparison:
1. Across univariate model types (linear vs spline vs tree)
2. Across four method configurations (no constraint/no adaptive vs ... vs both)

Data scenarios: pure nonlinear, mixed linear+nonlinear, block correlated nonlinear, null model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional, Any

from unilasso.uni_lasso import cv_uni
from experiments.base_experiment import BaseSimulationExperiment, compute_variable_selection_metrics
from experiments import visualization

from data_generators import (
    simulate_nonlinear_gaussian_data,
    simulate_nonlinear_glm_data,
    simulate_mixed_data
)


class NonlinearExperiment(BaseSimulationExperiment):
    """
    Nonlinear simulation experiment comparing different univariate models
    and XLasso configurations.
    """

    def __init__(
        self,
        scenario: str = 'pure_nonlinear',
        nonlinear_type: str = 'mixed',
        n_repeats: int = 20,
        test_size: float = 0.3,
        random_seed: int = 42,
        n: int = 500,
        p: int = 100,
        n_nonlinear: int = 8,
        correlation: str = 'block',
        rho: float = 0.7
    ):
        """
        Initialize experiment.

        Parameters
        ----------
        scenario : str
            Data scenario:
            - 'pure_nonlinear': all relevant features are nonlinear
            - 'mixed_nonlinear': mix of linear, nonlinear, and irrelevant
            - 'block_nonlinear': true nonlinear features are in correlated blocks
            - 'null_model': all features are irrelevant (test FDR control)
        nonlinear_type : str
            Type of nonlinear functions: 'sine', 'quadratic', 'step', 'mixed'
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
        n_nonlinear : int
            Number of nonlinear relevant features.
        correlation : str
            Correlation structure.
        rho : float
            Correlation coefficient.
        """
        super().__init__(n_repeats=n_repeats, test_size=test_size, random_seed=random_seed)
        self.scenario = scenario
        self.nonlinear_type = nonlinear_type
        self.n = n
        self.p = p
        self.n_nonlinear = n_nonlinear
        self.correlation = correlation
        self.rho = rho

    def generate_data(self, repeat_seed: int):
        """Generate nonlinear data for one repetition."""
        rs = np.random.RandomState(repeat_seed)

        if self.scenario == 'pure_nonlinear':
            X, y, active_true, true_functions = simulate_nonlinear_gaussian_data(
                n=self.n, p=self.p, n_nonlinear=self.n_nonlinear,
                nonlinear_type=self.nonlinear_type,
                correlation=self.correlation, rho=self.rho,
                seed=repeat_seed
            )
            beta_true = active_true  # 1 for active, 0 for inactive
        elif self.scenario == 'mixed_nonlinear':
            n_linear = self.n_nonlinear // 2
            X, y, beta_true, info = simulate_mixed_data(
                n=self.n, p=self.p, n_linear=n_linear, n_nonlinear=self.n_nonlinear - n_linear,
                correlation=self.correlation, rho=self.rho,
                seed=repeat_seed
            )
            # True active: any non-zero coefficient
            active_true = beta_true != 0
        elif self.scenario == 'block_nonlinear':
            X, y, active_true, true_functions = simulate_nonlinear_gaussian_data(
                n=self.n, p=self.p, n_nonlinear=self.n_nonlinear,
                nonlinear_type=self.nonlinear_type,
                correlation='block', rho=self.rho,
                seed=repeat_seed
            )
            beta_true = active_true
        elif self.scenario == 'null_model':
            # All features are irrelevant, y is just noise
            if self.correlation == 'block':
                block_size = 10
                n_blocks = self.p // block_size
                cov_blocks = []
                for _ in range(n_blocks):
                    block = np.ones((block_size, block_size)) * self.rho
                    np.fill_diagonal(block, 1.0)
                    cov_blocks.append(block)
                cov_matrix = np.block([[cov_blocks[i] if i == j else np.zeros((block_size, block_size))
                                        for j in range(n_blocks)] for i in range(n_blocks)])
                if self.p % block_size != 0:
                    remaining = self.p % block_size
                    remaining_cov = np.eye(remaining)
                    cov_matrix = np.block([[cov_matrix, np.zeros((cov_matrix.shape[0], remaining))],
                                           [np.zeros((remaining, cov_matrix.shape[1])), remaining_cov]])
            else:
                cov_matrix = np.eye(self.p)

            X = rs.multivariate_normal(np.zeros(self.p), cov_matrix, size=self.n)
            y = rs.normal(0, 1, size=self.n)
            beta_true = np.zeros(self.p)
            active_true = np.zeros(self.p).astype(bool)
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=repeat_seed
        )

        # True active is what we need for evaluation
        true_params = (beta_true, active_true.astype(bool))

        return X_train, y_train, X_test, y_test, true_params

    def get_model_configurations(self) -> Dict[str, dict]:
        """
        Return all configurations to test:
        - All combinations of (linear, spline, tree) × four method configurations
        """
        configs = {}

        univariate_models = ['linear', 'spline', 'tree']
        for model_type in univariate_models:
            # Four configurations for each univariate model type
            configs[f'{model_type} - baseline'] = {
                'univariate': model_type,
                'enable_group_constraint': False,
                'adaptive_weighting': False
            }
            configs[f'{model_type} - adaptive'] = {
                'univariate': model_type,
                'enable_group_constraint': False,
                'adaptive_weighting': True
            }
            configs[f'{model_type} - grouping'] = {
                'univariate': model_type,
                'enable_group_constraint': True,
                'adaptive_weighting': False
            }
            configs[f'{model_type} - full'] = {
                'univariate': model_type,
                'enable_group_constraint': True,
                'adaptive_weighting': True
            }

        return configs

    def fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        univariate: str,
        enable_group_constraint: bool,
        adaptive_weighting: bool
    ):
        """Fit model with given configuration."""
        import time
        start_time = time.time()

        model = cv_uni(
            X_train, y_train,
            family='gaussian',
            univariate_model=univariate,
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
        true_params
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        beta_true, active_true = true_params

        # Get coefficients for best lambda from CV
        coef = model.coefs[model.best_idx]
        intercept = model.intercept[model.best_idx]

        # Predict on test
        y_pred = X_test @ coef + intercept
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)

        # Feature selection evaluation
        selected = coef != 0
        # For nonlinear models, any coefficient != 0 means feature selected
        if selected.ndim > 1:
            # (p, df_basis) -> any non-zero in basis means feature selected
            selected = np.any(selected != 0, axis=1)

        vs_metrics = compute_variable_selection_metrics(selected, active_true)

        # Group accuracy if groups exist
        group_acc = np.nan
        if hasattr(model, 'groups') and model.groups is not None:
            from experiments.base_experiment import compute_group_accuracy
            group_acc = compute_group_accuracy(selected, model.groups, active_true)

        result = {
            'mse': mse,
            'rmse': rmse,
            'group_acc': group_acc,
            'n_selected': np.sum(selected),
            'tpr': vs_metrics['tpr'],
            'fpr': vs_metrics['fpr'],
            'precision': vs_metrics['precision'],
            'f1': vs_metrics['f1'],
        }

        return result

    def plot_results(self, output_dir: str):
        """Generate plots for nonlinear experiment."""
        os.makedirs(output_dir, exist_ok=True)
        raw_df = self.get_raw_results()

        # Plot MSE comparison across all configurations
        fig = visualization.plot_method_comparison_boxplot(
            raw_df, 'mse',
            title=f'Out-of-Sample MSE - Scenario: {self.scenario}',
            ylabel='MSE (lower is better)',
            ascending=True
        )
        visualization.save_figure(
            fig, os.path.join(output_dir, f'{self.scenario}_mse_boxplot')
        )

        # Plot F1 comparison
        fig = visualization.plot_method_comparison_boxplot(
            raw_df, 'f1',
            title=f'F1 Score (Feature Selection) - Scenario: {self.scenario}',
            ylabel='F1 (higher is better)',
            ascending=False
        )
        visualization.save_figure(
            fig, os.path.join(output_dir, f'{self.scenario}_f1_boxplot')
        )

        # Summarize by univariate model type: compare best configuration
        agg_df = self.aggregate_results()

        # Extract full configurations for each univariate type
        mean_f1 = []
        for model_type in ['linear', 'spline', 'tree']:
            key = f'{model_type} - full'
            if key in agg_df.index:
                mean_f1.append((model_type, agg_df.loc[key, ('f1', 'mean')], agg_df.loc[key, ('f1', 'std')]))

        model_types, m_f1, s_f1 = zip(*mean_f1)
        s = pd.Series(m_f1, index=model_types)
        s_std = pd.Series(s_f1, index=model_types)

        fig = visualization.plot_summary_bar_chart(
            s, s_std,
            title=f'Full XLasso F1 by Univariate Model - {self.scenario}',
            ylabel='F1 (higher is better)'
        )
        visualization.save_figure(
            fig, os.path.join(output_dir, f'{self.scenario}_model_type_comparison')
        )

        print(f"Plots saved to {output_dir}")


def run_all_nonlinear_experiments(
    output_dir: str = 'experiments/results',
    n_repeats_override: Optional[int] = None,
    save_plots: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run all nonlinear experiments across scenarios.

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
        Aggregated results for each scenario.
    """
    scenarios = [
        ('pure_nonlinear', 'mixed'),
        ('mixed_nonlinear', 'mixed'),
        ('block_nonlinear', 'mixed'),
        ('null_model', 'mixed')
    ]

    all_results = {}
    os.makedirs(output_dir, exist_ok=True)

    for scenario_name, nonlinear_type in scenarios:
        print(f"\n{'='*60}")
        print(f"Running nonlinear experiment: scenario = {scenario_name}")
        print(f"{'='*60}")

        n_reps = 20 if n_repeats_override is None else n_repeats_override
        exp = NonlinearExperiment(
            scenario=scenario_name,
            nonlinear_type=nonlinear_type,
            n_repeats=n_reps,
            n=500,
            p=100,
            n_nonlinear=8,
            correlation='block',
            rho=0.7,
            random_seed=42
        )
        exp.run()
        agg = exp.aggregate_results()
        all_results[scenario_name] = agg
        exp.save_results(os.path.join(output_dir, f'{scenario_name}_results.csv'))
        if save_plots:
            exp.plot_results(output_dir)

    # Create summary
    summary_rows = []
    for scenario, df in all_results.items():
        for method in df.index:
            row = {
                'scenario': scenario,
                'method': method,
                'f1_mean': df.loc[method, ('f1', 'mean')],
                'f1_std': df.loc[method, ('f1', 'std')],
                'mse_mean': df.loc[method, ('mse', 'mean')],
                'mse_std': df.loc[method, ('mse', 'std')],
                'tpr_mean': df.loc[method, ('tpr', 'mean')],
                'fpr_mean': df.loc[method, ('fpr', 'mean')],
            }
            # Extract univariate model type
            univariate_type = method.split(' - ')[0]
            config_type = method.split(' - ')[1]
            row['univariate'] = univariate_type
            row['config'] = config_type
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'nonlinear_summary.csv'), index=False)
    print(f"\nAll nonlinear experiments completed. Summary saved to {output_dir}/nonlinear_summary.csv")

    return all_results


def compare_nonlinear_models(
    results_dir: str = 'experiments/results',
    output_dir: str = 'experiments/results'
):
    """
    Generate summary comparison of different univariate models.

    Parameters
    ----------
    results_dir : str
        Directory with result files.
    output_dir : str
        Directory to save summary.
    """
    df = pd.read_csv(os.path.join(results_dir, 'nonlinear_summary.csv'))

    # Average across scenarios: full configuration by model type
    summary = df[df['config'] == 'full'].groupby('univariate').agg({
        'f1_mean': 'mean',
        'mse_mean': 'mean'
    }).round(3)

    print("\nNonlinear model comparison (full configuration):")
    print(summary)

    latex_table = """
\\begin{table}[ht]
\\centering
\\caption{Comparison of univariate models on nonlinear scenarios (average across all scenarios)}
\\label{tab:nonlinear_comparison}
\\begin{tabular}{lcc}
\\hline
Univariate Model & Avg F1 & Avg Out-of-sample MSE \\\\
\\hline
"""
    for idx, row in summary.iterrows():
        latex_table += f"{idx} & {row['f1_mean']:.3f} & {row['mse_mean']:.3f} \\\\\n"

    latex_table += """\\hline
\\end{tabular}
\\end{table}
"""

    with open(os.path.join(output_dir, 'nonlinear_model_comparison.tex'), 'w') as f:
        f.write(latex_table)

    print(f"\nLaTeX table saved to {output_dir}/nonlinear_model_comparison.tex")
