"""
UniLasso - UniLasso-based Sparse Feature Selector
=================================================
Wrapper around the UniLasso package for sparse feature selection.

Uses the fit_unilasso and cv_unilasso functions from unilasso package.
Reference: https://arxiv.org/abs/2501.18360
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Union, List, Dict, Any
import sys
from pathlib import Path

# Import unilasso functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "unilasso"))

from unilasso import fit_unilasso, cv_unilasso

from .base import BaseSparseSelector


class UniLasso(BaseSparseSelector):
    """
    UniLasso: Univariate-Guided Sparse Regression.

    A sparse selector that uses univariate regression guides to inform
    the Lasso penalty, improving feature selection accuracy.
    """

    def __init__(
        self,
        lambda_1: float = 0.01,
        lambda_2: float = 0.01,
        group_threshold: float = 0.7,
        standardize: bool = True,
        fit_intercept: bool = True,
        family: str = "gaussian",
        n_lmdas: int = 100,
        lmda_min_ratio: float = 1e-4,
        verbose: bool = False,
    ):
        super().__init__(standardize=standardize, fit_intercept=fit_intercept)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.group_threshold = group_threshold
        self.family = family
        self.n_lmdas = n_lmdas
        self.lmda_min_ratio = lmda_min_ratio
        self.verbose = verbose
        self.lmdas_: Optional[np.ndarray] = None
        self.result_: Any = None

    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        lmdas: Optional[Union[float, List[float], np.ndarray]] = None,
        **kwargs
    ) -> "UniLasso":
        """Fit UniLasso model."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Standardize
        if self.standardize:
            self._X_mean = np.mean(X, axis=0)
            self._X_std = np.std(X, axis=0)
            self._X_std[self._X_std < 1e-10] = 1.0
            X_scaled = (X - self._X_mean) / self._X_std
        else:
            X_scaled = X
            self._X_mean = np.zeros(n_features)
            self._X_std = np.ones(n_features)

        y_centered = y - np.mean(y)

        # Compute lambda path
        if lmdas is None:
            lambda_max = np.max(np.abs(X_scaled.T @ y_centered)) / n_samples
            lmdas = np.linspace(lambda_max, lambda_max * self.lmda_min_ratio, self.n_lmdas)

        # Call unilasso fit
        self.result_ = fit_unilasso(
            X=X_scaled, y=y_centered, family=self.family, lmdas=lmdas,
            n_lmdas=self.n_lmdas, lmda_min_ratio=self.lmda_min_ratio,
            verbose=self.verbose,
        )

        # Extract coefficients
        if hasattr(self.result_, 'coefs'):
            if self.result_.coefs.ndim == 2:
                best_idx = 0 if not hasattr(self.result_, 'best_idx') else self.result_.best_idx
                if best_idx is None:
                    best_idx = 0
                self.coef_ = self.result_.coefs[best_idx]
            else:
                self.coef_ = self.result_.coefs
        else:
            self.coef_ = self.result_.get('coefs', np.zeros(n_features))

        self.intercept_ = float(getattr(self.result_, 'intercept', np.mean(y)))
        self.coef_ = self.coef_ / self._X_std
        self.intercept_ = self.intercept_ + np.sum(self._X_mean * self.coef_)

        self.is_fitted_ = True
        self.lmdas_ = lmdas if isinstance(lmdas, np.ndarray) else np.array([lmdas])

        return self

    def predict(self, X: npt.NDArray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.coef_ + self.intercept_

    def score(self, X: npt.NDArray, y: npt.NDArray) -> float:
        """Calculate R-squared score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


class UniLassoCV(BaseSparseSelector):
    """Cross-validated UniLasso with automatic lambda selection."""

    def __init__(
        self,
        lambda_1: float = 0.01,
        lambda_2: float = 0.01,
        n_folds: int = 5,
        standardize: bool = True,
        fit_intercept: bool = True,
        family: str = "gaussian",
        lmda_min_ratio: float = 1e-4,
        verbose: bool = False,
        random_state: int = 42,
    ):
        super().__init__(standardize=standardize, fit_intercept=fit_intercept)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.n_folds = n_folds
        self.family = family
        self.lmda_min_ratio = lmda_min_ratio
        self.verbose = verbose
        self.random_state = random_state
        self.best_lmda_: Optional[float] = None
        self.cv_results_: Optional[Dict] = None

    def fit(self, X: npt.NDArray, y: npt.NDArray, **kwargs) -> "UniLassoCV":
        """Fit UniLassoCV model with cross-validation."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self.standardize:
            self._X_mean = np.mean(X, axis=0)
            self._X_std = np.std(X, axis=0)
            self._X_std[self._X_std < 1e-10] = 1.0
            X_scaled = (X - self._X_mean) / self._X_std
        else:
            X_scaled = X
            self._X_mean = np.zeros(n_features)
            self._X_std = np.ones(n_features)

        y_centered = y - np.mean(y)

        self.result_ = cv_unilasso(
            X=X_scaled, y=y_centered, family=self.family, n_folds=self.n_folds,
            lmda_min_ratio=self.lmda_min_ratio, verbose=self.verbose,
            seed=self.random_state,
        )

        if hasattr(self.result_, 'coefs'):
            if self.result_.coefs.ndim == 2:
                best_idx = 0 if not hasattr(self.result_, 'best_idx') else self.result_.best_idx
                if best_idx is None:
                    best_idx = 0
                self.coef_ = self.result_.coefs[best_idx]
            else:
                self.coef_ = self.result_.coefs
        else:
            self.coef_ = np.zeros(n_features)

        self.best_lmda_ = float(self.result_.best_lmda) if hasattr(self.result_, 'best_lmda') else self.lambda_1
        self.intercept_ = float(getattr(self.result_, 'intercept', np.mean(y)))
        self.coef_ = self.coef_ / self._X_std
        self.intercept_ = self.intercept_ + np.sum(self._X_mean * self.coef_)

        self.cv_results_ = {}
        if hasattr(self.result_, 'avg_losses'):
            self.cv_results_['avg_losses'] = self.result_.avg_losses
        if hasattr(self.result_, 'lmdas'):
            self.cv_results_['lmdas'] = self.result_.lmdas

        self.is_fitted_ = True
        return self

    def predict(self, X: npt.NDArray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.coef_ + self.intercept_

    def score(self, X: npt.NDArray, y: npt.NDArray) -> float:
        """Calculate R-squared score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
