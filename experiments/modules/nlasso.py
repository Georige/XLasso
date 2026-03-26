"""
NLasso - Nonlinear Lasso with Spline/Tree Models
================================================
Wrapper around the NLasso package for sparse feature selection
with nonlinear univariate models.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Union, List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.NLasso import NLasso as NLassoImpl

from .base import BaseSparseSelector


class NLasso(BaseSparseSelector):
    """
    NLasso: Nonlinear Lasso with flexible univariate models.

    Uses spline or decision tree models for univariate feature
    transformations before applying Lasso penalty.
    """

    def __init__(
        self,
        lambda_ridge: float = 10.0,
        lambda_: float = None,
        n_lambda: int = 50,
        gamma: float = 0.3,
        s: float = 1.0,
        group_threshold: float = 0.7,
        group_min_size: int = 2,
        group_max_size: int = 10,
        group_truncation_threshold: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = True,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
    ):
        super().__init__(standardize=standardize, fit_intercept=fit_intercept)
        self.lambda_ridge = lambda_ridge
        self.lambda_ = lambda_
        self.n_lambda = n_lambda
        self.gamma = gamma
        self.s = s
        self.group_threshold = group_threshold
        self.group_min_size = group_min_size
        self.group_max_size = group_max_size
        self.group_truncation_threshold = group_truncation_threshold
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: npt.NDArray, y: npt.NDArray, **kwargs) -> "NLasso":
        """Fit NLasso model."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self._impl = NLassoImpl(
            lambda_ridge=self.lambda_ridge,
            lambda_=self.lambda_,
            lambda_path=None,
            n_lambda=self.n_lambda,
            gamma=self.gamma,
            s=self.s,
            group_threshold=self.group_threshold,
            group_min_size=self.group_min_size,
            group_max_size=self.group_max_size,
            group_truncation_threshold=self.group_truncation_threshold,
            max_iter=self.max_iter,
            tol=self.tol,
            standardize=self.standardize,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self._impl.fit(X, y)

        if hasattr(self._impl, 'coef_'):
            self.coef_ = self._impl.coef_.flatten()
        else:
            self.coef_ = np.zeros(n_features)

        if hasattr(self._impl, 'intercept_'):
            self.intercept_ = float(self._impl.intercept_)
        else:
            self.intercept_ = 0.0

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


class NLassoCV(BaseSparseSelector):
    """Cross-validated NLasso with automatic lambda selection."""

    def __init__(
        self,
        lambda_ridge: float = 10.0,
        n_lambda: int = 50,
        n_folds: int = 5,
        gamma: float = 0.3,
        s: float = 1.0,
        group_threshold: float = 0.7,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
    ):
        super().__init__(standardize=standardize)
        self.lambda_ridge = lambda_ridge
        self.n_lambda = n_lambda
        self.n_folds = n_folds
        self.gamma = gamma
        self.s = s
        self.group_threshold = group_threshold
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: npt.NDArray, y: npt.NDArray, **kwargs) -> "NLassoCV":
        """Fit NLassoCV with cross-validation."""
        from modules.NLasso import NLassoCV as NLassoCVImpl

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self._impl = NLassoCVImpl(
            lambda_ridge=self.lambda_ridge,
            n_lambda=self.n_lambda,
            gamma=self.gamma,
            s=self.s,
            group_threshold=self.group_threshold,
            max_iter=self.max_iter,
            tol=self.tol,
            standardize=self.standardize,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self._impl.fit(X, y)

        if hasattr(self._impl, 'coef_'):
            self.coef_ = self._impl.coef_.flatten()
        else:
            self.coef_ = np.zeros(n_features)

        if hasattr(self._impl, 'intercept_'):
            self.intercept_ = float(self._impl.intercept_)
        else:
            self.intercept_ = 0.0

        if hasattr(self._impl, 'best_lambda_'):
            self.best_lambda_ = float(self._impl.best_lambda_)
        else:
            self.best_lambda_ = self.lambda_ridge

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