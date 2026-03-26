"""
Post-Lasso - Two-Stage Sparse Feature Selection
================================================
Post-Lasso performs initial variable selection using Lasso,
then refits the model using only the selected variables.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any
import warnings
from sklearn.linear_model import LassoCV, Lasso

from .base import BaseSparseSelector


class PostLasso(BaseSparseSelector):
    """
    Post-Lasso: Two-stage sparse selection.

    Stage 1: Fit Lasso to select features
    Stage 2: Refit OLS using only selected features
    """

    def __init__(
        self,
        lambda_: float = 0.01,
        cv_folds: int = 5,
        selection_threshold: float = 1e-6,
        standardize: bool = True,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        super().__init__(standardize=standardize, fit_intercept=fit_intercept)
        self.lambda_ = lambda_
        self.cv_folds = cv_folds
        self.selection_threshold = selection_threshold
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.selected_: Optional[np.ndarray] = None
        self.stage1_coef_: Optional[np.ndarray] = None

    def fit(self, X: npt.NDArray, y: npt.NDArray, **kwargs) -> "PostLasso":
        """Fit Post-Lasso model."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Stage 1: Lasso selection
        if self.cv_folds > 1:
            lasso = LassoCV(
                alphas=[self.lambda_],
                cv=self.cv_folds,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                fit_intercept=self.fit_intercept,
                normalize=not self.standardize,
            )
            lasso.fit(X, y)
            self.stage1_coef_ = lasso.coef_
            self.best_alpha_ = float(lasso.alpha_)
        else:
            lasso = Lasso(
                alpha=self.lambda_,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                fit_intercept=self.fit_intercept,
                normalize=not self.standardize,
            )
            lasso.fit(X, y)
            self.stage1_coef_ = lasso.coef_
            self.best_alpha_ = self.lambda_

        self.selected_ = np.where(np.abs(self.stage1_coef_) > self.selection_threshold)[0]

        if len(self.selected_) == 0:
            warnings.warn("No features selected by Lasso. Using zero coefficients.")
            self.coef_ = np.zeros(n_features)
            self.intercept_ = float(np.mean(y)) if self.fit_intercept else 0.0
        else:
            X_selected = X[:, self.selected_]

            if self.fit_intercept:
                X_selected_with_intercept = np.column_stack([np.ones(n_samples), X_selected])
                coeffs = np.linalg.lstsq(X_selected_with_intercept, y, rcond=None)[0]
                self.intercept_ = float(coeffs[0])
                self.coef_ = coeffs[1:]
            else:
                self.coef_ = np.linalg.lstsq(X_selected, y, rcond=None)[0]
                self.intercept_ = 0.0

            full_coef = np.zeros(n_features)
            full_coef[self.selected_] = self.coef_
            self.coef_ = full_coef

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

    def get_selected_features(self, threshold: float = None) -> np.ndarray:
        """Get indices of selected features."""
        if self.selected_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.selected_

    def score(self, X: npt.NDArray, y: npt.NDArray) -> float:
        """Calculate R-squared score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0