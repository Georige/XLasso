"""
Base Module - Common Utilities and Abstract Base Classes
========================================================
Provides:
- BaseSparseSelector: Abstract base class for all sparse selectors
- DataGenerator: Unified interface for synthetic data generation
- MetricCalculator: Unified metrics computation (F1, TPR, FDR, R2, MSE, etc.)
- CrossValidator: K-fold cross-validation wrapper
- LOOHelper: Leave-One-Out related utilities (Sherman-Morrison, etc.)
"""

import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from sklearn.model_selection import KFold
import warnings


# =============================================================================
# Abstract Base Class
# =============================================================================

class BaseSparseSelector(ABC):
    """
    Abstract base class for sparse feature selectors.

    All sparse selector algorithms should inherit from this class and implement:
    - fit(): Fit the model to training data
    - predict(): Make predictions on new data
    - get_selected_features(): Return indices of selected features
    - get_coefficients(): Return the fitted coefficients
    """

    def __init__(self, standardize: bool = True, fit_intercept: bool = True):
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.n_features_in_: Optional[int] = None
        self.is_fitted_: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseSparseSelector":
        """
        Fit the sparse selector to training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix
        y : np.ndarray of shape (n_samples,)
            Target vector

        Returns
        -------
        self : BaseSparseSelector
            Fitted estimator
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predictions
        """
        pass

    def get_selected_features(self, threshold: float = 1e-6) -> np.ndarray:
        """
        Get indices of selected features.

        Parameters
        ----------
        threshold : float
            Coefficient threshold below which features are considered unselected

        Returns
        -------
        selected : np.ndarray
            Indices of selected features
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return np.where(np.abs(self.coef_) > threshold)[0]

    def get_coefficients(self) -> np.ndarray:
        """
        Get fitted coefficients.

        Returns
        -------
        coef : np.ndarray
            Fitted coefficients
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R-squared score.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True target values

        Returns
        -------
        score : float
            R-squared score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# =============================================================================
# Data Generator
# =============================================================================

@dataclass
class DataGenerator:
    """
    Unified data generator for sparse regression experiments.

    Supports multiple correlation structures:
    - pairwise: Equicorrelated (constant correlation between all pairs)
    - ar1: AR(1) autocorrelation structure
    - twin: Twin variables (pairs with opposite signs)
    - block: Block diagonal correlation structure
    """

    random_state: int = 42

    def __post_init__(self):
        self.rng = np.random.RandomState(self.random_state)

    def generate(
        self,
        n_samples: int = 300,
        n_features: int = 500,
        n_nonzero: int = 20,
        sigma: float = 1.0,
        correlation_type: str = "pairwise",
        rho: float = 0.5,
        family: str = "gaussian",
        block_size: int = 10,
        n_blocks: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic sparse regression data.
        """
        if correlation_type == "pairwise":
            return self._generate_pairwise(n_samples, n_features, n_nonzero, sigma, family, rho)
        elif correlation_type == "ar1":
            return self._generate_ar1(n_samples, n_features, n_nonzero, sigma, family, rho)
        elif correlation_type == "twin":
            return self._generate_twin(n_samples, n_features, n_nonzero, sigma, family, rho)
        elif correlation_type == "block":
            return self._generate_block(n_samples, n_features, n_nonzero, sigma, family, block_size, n_blocks)
        else:
            raise ValueError(f"Unknown correlation_type: {correlation_type}")

    def _generate_pairwise(self, n, p, k, sigma, family, rho) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate equicorrelated data."""
        cov = np.full((p, p), rho)
        np.fill_diagonal(cov, 1.0)
        X = self.rng.multivariate_normal(np.zeros(p), cov, size=n)
        beta_true = np.zeros(p)
        beta_true[:k] = 1.0
        if family == "gaussian":
            y = X @ beta_true + self.rng.randn(n) * sigma
        else:
            z = X @ beta_true + self.rng.randn(n) * sigma
            y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
        return X, y, beta_true

    def _generate_ar1(self, n, p, k, sigma, family, rho) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate AR(1) autocorrelated data."""
        cov = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                cov[i, j] = rho ** abs(i - j)
        X = self.rng.multivariate_normal(np.zeros(p), cov, size=n)
        beta_true = np.zeros(p)
        beta_true[1 : 2 * k : 2] = 1.0
        if family == "gaussian":
            y = X @ beta_true + self.rng.randn(n) * sigma
        else:
            z = X @ beta_true + self.rng.randn(n) * sigma
            y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
        return X, y, beta_true

    def _generate_twin(self, n, p, k, sigma, family, rho) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate twin variables data."""
        X = self.rng.randn(n, p)
        beta_true = np.zeros(p)
        for i in range(k):
            beta_true[2 * i] = 2.0
            beta_true[2 * i + 1] = -2.5
        if family == "gaussian":
            y = X @ beta_true + self.rng.randn(n) * sigma
        else:
            z = X @ beta_true + self.rng.randn(n) * sigma
            y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
        return X, y, beta_true

    def _generate_block(self, n, p, k, sigma, family, block_size, n_blocks) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate block diagonal correlation structure."""
        cov = np.zeros((p, p))
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, p)
            block_cov = np.full((end - start, end - start), rho)
            np.fill_diagonal(block_cov, 1.0)
            cov[start:end, start:end] = block_cov
        X = self.rng.multivariate_normal(np.zeros(p), cov, size=n)
        beta_true = np.zeros(p)
        beta_true[:k] = 1.0
        if family == "gaussian":
            y = X @ beta_true + self.rng.randn(n) * sigma
        else:
            z = X @ beta_true + self.rng.randn(n) * sigma
            y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
        return X, y, beta_true


# =============================================================================
# Metric Calculator
# =============================================================================

class MetricCalculator:
    """Unified metrics calculator for sparse regression evaluation."""

    @staticmethod
    def calculate_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        beta_true: np.ndarray,
        selected: np.ndarray,
        n_nonzero: int = None,
        threshold: float = 1e-6,
    ) -> Dict[str, float]:
        """Calculate all metrics."""
        if n_nonzero is None:
            n_nonzero = int(np.sum(np.abs(beta_true) > threshold))

        mse = MetricCalculator.mse(y_true, y_pred)
        r2 = MetricCalculator.r2(y_true, y_pred)
        true_nonzero = set(np.where(np.abs(beta_true) > threshold)[0])
        selected_set = set(selected)
        tp = len(true_nonzero & selected_set)
        fp = len(selected_set - true_nonzero)
        fn = len(true_nonzero - selected_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        tpr = recall
        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        sparsity = 1.0 - len(selected) / len(beta_true) if len(beta_true) > 0 else 0.0

        return {
            "mse": mse, "r2": r2, "f1": f1, "tpr": tpr, "fdr": fdr,
            "precision": precision, "recall": recall, "sparsity": sparsity,
            "n_selected": len(selected), "n_true_nonzero": n_nonzero,
        }

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# =============================================================================
# Cross Validator
# =============================================================================

class CrossValidator:
    """K-fold cross-validation wrapper."""

    def __init__(self, n_folds: int = 5, shuffle: bool = True, random_state: int = 42, stratify: bool = False):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify
        self.kfold = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Generate train/test splits."""
        yield from self.kfold.split(X)

    @staticmethod
    def get_fold_indices(n_samples: int, n_folds: int, fold: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get train/test indices for a specific fold."""
        rng = np.random.RandomState(seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        fold_size = n_samples // n_folds
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        return train_idx, test_idx


# =============================================================================
# LOO Helper
# =============================================================================

class LOOHelper:
    """Leave-One-Out related utilities."""

    @staticmethod
    def sherman_morrison_update(X: np.ndarray, diag_xx: np.ndarray, threshold: float = 1e-10) -> np.ndarray:
        """Compute diagonal of (X^T X)^{-1} using Sherman-Morrison formula."""
        n, p = X.shape
        inv_diag = np.zeros(p)
        M = np.eye(n)
        for j in range(p):
            x_j = X[:, j]
            u = M @ x_j
            denom = 1.0 + x_j @ u
            if np.abs(denom) < threshold:
                inv_diag[j] = 0.0
            else:
                inv_diag[j] = 1.0 / denom
        return inv_diag

    @staticmethod
    def compute_loo_predictions(X: np.ndarray, y: np.ndarray, beta: np.ndarray, intercept: float = 0.0) -> np.ndarray:
        """Compute Leave-One-Out predictions efficiently."""
        n = X.shape[0]
        loo_pred = np.zeros(n)
        for i in range(n):
            X_minus_i = np.delete(X, i, axis=0)
            y_minus_i = np.delete(y, i)
            if X_minus_i.shape[0] > X_minus_i.shape[1]:
                beta_i = np.linalg.lstsq(X_minus_i, y_minus_i, rcond=None)[0]
            else:
                lambda_reg = 0.1
                XtX = X_minus_i.T @ X_minus_i + lambda_reg * np.eye(X_minus_i.shape[1])
                Xty = X_minus_i.T @ y_minus_i
                beta_i = np.linalg.solve(XtX, Xty)
            loo_pred[i] = X[i] @ beta_i + intercept
        return loo_pred