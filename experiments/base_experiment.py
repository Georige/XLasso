"""
Base experiment class for all UniLasso simulation experiments.
Provides common infrastructure:
- Data splitting
- Repeated experiments
- Results aggregation
- Saving and plotting
"""

import numpy as np
import pandas as pd
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from sklearn.model_selection import train_test_split


class BaseSimulationExperiment(ABC):
    """Base class for all simulation experiments."""

    def __init__(
        self,
        n_repeats: int = 20,
        test_size: float = 0.3,
        random_seed: int = 42
    ):
        """
        Initialize the experiment.

        Parameters
        ----------
        n_repeats : int
            Number of independent repetitions of the experiment.
        test_size : float
            Fraction of data to use for testing (out-of-sample evaluation).
        random_seed : int
            Base random seed for reproducibility.
        """
        self.n_repeats = n_repeats
        self.test_size = test_size
        self.random_seed = random_seed
        self.results: List[Dict[str, Any]] = []
        self.aggregated_results: Optional[pd.DataFrame] = None

    @abstractmethod
    def generate_data(self, repeat_seed: int):
        """
        Generate synthetic data for one repetition.

        Parameters
        ----------
        repeat_seed : int
            Random seed for this repetition.

        Returns
        -------
        X_train : np.ndarray
        y_train : np.ndarray
        X_test : np.ndarray
        y_test : np.ndarray
        true_params : Any
            True parameters (beta_true, etc.) for evaluation.
        """
        raise NotImplementedError

    @abstractmethod
    def fit_model(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        Fit model with given configuration.

        Parameters
        ----------
        X_train : np.ndarray
        y_train : np.ndarray
        **kwargs : dict
            Model configuration parameters.

        Returns
        -------
        model : fitted model object
            Must have `coef_` attribute for variable selection.
        runtime : float
            Fitting time in seconds.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        true_params
    ) -> Dict[str, float]:
        """
        Evaluate fitted model on test data using the true parameters.

        Parameters
        ----------
        model : fitted model
        X_test : np.ndarray
        y_test : np.ndarray
        true_params : True parameters (beta_true, etc.)

        Returns
        -------
        metrics : Dict[str, float]
            Dictionary of evaluation metrics.
        """
        raise NotImplementedError

    def run(self):
        """
        Run the full experiment with all repetitions.
        """
        np.random.seed(self.random_seed)
        seeds = np.random.randint(0, 100000, size=self.n_repeats)

        print(f"Starting experiment with {self.n_repeats} repetitions...")

        for i, seed in enumerate(seeds):
            print(f"  Repeat {i+1}/{self.n_repeats} (seed={seed})", end="", flush=True)

            # Generate data with train/test split
            data = self.generate_data(int(seed))
            if len(data) == 5:
                X_train, y_train, X_test, y_test, true_params = data
            else:
                raise ValueError("generate_data must return (X_train, y_train, X_test, y_test, true_params)")

            # Get all model configurations to test
            configs = self.get_model_configurations()

            repeat_results = {}
            for config_name, config_kwargs in configs.items():
                print(f" [{config_name}]", end="", flush=True)

                # Fit model and measure time
                model, runtime = self.fit_model(X_train, y_train, **config_kwargs)

                # Evaluate
                metrics = self.evaluate_model(model, X_test, y_test, true_params)
                metrics['runtime'] = runtime
                metrics['repeat'] = i + 1
                metrics['method'] = config_name

                repeat_results[config_name] = metrics

            self.results.append(repeat_results)
            print("", flush=True)

        print("Experiment completed.")

    @abstractmethod
    def get_model_configurations(self) -> Dict[str, dict]:
        """
        Return the dictionary of model configurations to compare.

        Returns
        -------
        configs : Dict[str, dict]
            {config_name: config_kwargs}
        """
        raise NotImplementedError

    def aggregate_results(self) -> pd.DataFrame:
        """
        Aggregate results across all repetitions.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with mean and std for each metric-method combination.
        """
        # Flatten results
        all_rows = []
        for repeat_result in self.results:
            for method_name, metrics in repeat_result.items():
                row = {'method': method_name}
                row.update(metrics)
                all_rows.append(row)

        df = pd.DataFrame(all_rows)

        # Compute mean and std across repetitions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['repeat']]

        agg_df = df.groupby('method')[numeric_cols].agg(['mean', 'std'])
        agg_df = agg_df.round(4)

        self.aggregated_results = agg_df
        return agg_df

    def save_results(self, filepath: str):
        """
        Save aggregated results to CSV.

        Parameters
        ----------
        filepath : str
            Path to output CSV file.
        """
        if self.aggregated_results is None:
            self.aggregate_results()

        self.aggregated_results.to_csv(filepath)
        print(f"Results saved to {filepath}")

    def get_raw_results(self) -> pd.DataFrame:
        """
        Get raw results from all repetitions as a DataFrame.

        Returns
        -------
        df : pd.DataFrame
            Raw results with one row per method per repetition.
        """
        all_rows = []
        for repeat_result in self.results:
            for method_name, metrics in repeat_result.items():
                row = {'method': method_name}
                row.update(metrics)
                all_rows.append(row)

        return pd.DataFrame(all_rows)

    def plot_results(self, output_dir: str):
        """
        Generate and save plots of results.

        Parameters
        ----------
        output_dir : str
            Directory to save plots.
        """
        # Default implementation does nothing
        # Override in subclass or use visualization.py
        pass


def compute_variable_selection_metrics(
    selected: np.ndarray,
    true_nonzero: np.ndarray
) -> Dict[str, float]:
    """
    Compute standard variable selection metrics.

    Parameters
    ----------
    selected : np.ndarray
        Boolean array of selected features.
    true_nonzero : np.ndarray
        Boolean array of true nonzero features.

    Returns
    -------
    metrics : Dict[str, float]
        Contains:
        - tp: True positives
        - fp: False positives
        - tn: True negatives
        - fn: False negatives
        - tpr: True positive rate (recall)
        - fpr: False positive rate
        - precision: Precision
        - f1: F1 score
        - accuracy: Overall accuracy
    """
    # Convert to boolean
    selected = np.asarray(selected).astype(bool)
    true_nonzero = np.asarray(true_nonzero).astype(bool)

    tp = np.sum(selected & true_nonzero)
    fp = np.sum(selected & ~true_nonzero)
    tn = np.sum(~selected & ~true_nonzero)
    fn = np.sum(~selected & true_nonzero)

    n_true = np.sum(true_nonzero)
    n_false = np.sum(~true_nonzero)

    tpr = tp / n_true if n_true > 0 else 0.0
    fpr = fp / n_false if n_false > 0 else 0.0

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tpr + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * (tpr * precision) / (tpr + precision)

    accuracy = (tp + tn) / (n_true + n_false)

    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'tpr': tpr,
        'fpr': fpr,
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy
    }


def compute_group_accuracy(
    selected: np.ndarray,
    groups: Dict[int, List[int]],
    true_nonzero: np.ndarray
) -> float:
    """
    Compute group accuracy for grouped selection.

    For each group, check that all true members are either all selected or all not selected.

    Parameters
    ----------
    selected : np.ndarray
        Selected features (boolean).
    groups : Dict[int, List[int]]
        Group mapping: group_id -> list of feature indices.
    true_nonzero : np.ndarray
        True nonzero features.

    Returns
    -------
    accuracy : float
        Fraction of groups with correct grouping behavior.
    """
    correct = 0
    total = 0

    for group_id, feature_indices in groups.items():
        feature_indices = np.array(feature_indices)
        # Does this group contain any true nonzero features?
        has_true = np.any(true_nonzero[feature_indices])

        # Are all features consistently selected?
        all_selected = np.all(selected[feature_indices])
        all_not_selected = np.all(~selected[feature_indices])

        # If group has true features, should all be selected
        # If group has no true features, should all be not selected
        if (has_true and all_selected) or (not has_true and all_not_selected):
            correct += 1

        total += 1

    return correct / total if total > 0 else 0.0
