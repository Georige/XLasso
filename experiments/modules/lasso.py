"""
Lasso - sklearn Lasso wrapper
=============================
Wrapper around sklearn's Lasso and LassoCV for sparse feature selection.
Follows the BaseSparseSelector interface.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Union, List, Dict, Any

from sklearn.linear_model import Lasso as SklearnLasso, LassoCV as SklearnLassoCV

from .base import BaseSparseSelector


class Lasso(BaseSparseSelector):
    """
    Lasso regression wrapper using sklearn's Lasso implementation.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 penalty.
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    standardize : bool, default=True
        Whether to standardize features before fitting.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for optimization.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        standardize: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        super().__init__(standardize=standardize, fit_intercept=fit_intercept)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.n_features_in_: Optional[int] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> "Lasso":
        """Fit Lasso model."""
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

        # Center y for intercept calculation
        y_centered = y - np.mean(y)

        # Fit sklearn Lasso
        self._model = SklearnLasso(
            alpha=self.alpha,
            fit_intercept=False,  # We handle intercept manually via centering
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self._model.fit(X_scaled, y_centered, sample_weight=sample_weight)

        # Extract and scale coefficients
        self.coef_ = self._model.coef_ / self._X_std
        self.intercept_ = float(np.mean(y) - np.sum(self._model.coef_ * self._X_mean / self._X_std))

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


class LassoCV(BaseSparseSelector):
    """
    Cross-validated Lasso with automatic alpha selection.

    Parameters
    ----------
    alphas : array-like, optional
        List of alpha values to try. If None, a set of values is chosen automatically.
    n_folds : int, default=5
        Number of folds for cross-validation.
    fit_intercept : bool, default=True
        Whether to fit the intercept.
    standardize : bool, default=True
        Whether to standardize features before fitting.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for optimization.
    cv : int, default=5
        Number of folds for cross-validation.
    random_state : int, default=42
        Random state for reproducibility.
    use_1se : bool, default=True
        If True, use 1-SE rule to select the most regularized model
        within 1 standard error of the best score. The 1-SE rule selects
        the largest alpha whose CV MSE is within 1-SE of the minimum MSE.
    """

    def __init__(
        self,
        alphas: Optional[Union[List[float], np.ndarray]] = None,
        n_folds: int = 5,
        fit_intercept: bool = True,
        standardize: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        cv: int = 5,
        random_state: int = 42,
        use_1se: bool = True,
    ):
        super().__init__(standardize=standardize, fit_intercept=fit_intercept)
        self.alphas = alphas
        self.n_folds = n_folds
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.random_state = random_state
        self.use_1se = use_1se
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.n_features_in_: Optional[int] = None
        self.is_fitted_: bool = False
        self.best_alpha_: Optional[float] = None
        self.cv_results_: Optional[Dict] = None

    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> "LassoCV":
        """Fit LassoCV model with cross-validation."""
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

        # Center y
        y_centered = y - np.mean(y)

        # Fit sklearn LassoCV
        # We will manually apply 1-SE rule after fitting using mse_path_
        self._model = SklearnLassoCV(
            alphas=self.alphas,
            cv=self.cv,
            fit_intercept=False,  # We handle intercept manually
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            selection='random',  # Random selection is more efficient for many features
        )
        self._model.fit(X_scaled, y_centered, sample_weight=sample_weight)

        # Apply 1-SE rule manually if requested
        if self.use_1se and hasattr(self._model, 'mse_path_') and self._model.mse_path_ is not None:
            # mse_path_ shape: (n_alphas, n_folds)
            mse_path = self._model.mse_path_
            mean_mse = mse_path.mean(axis=-1)
            std_mse = mse_path.std(axis=-1)
            se_mse = std_mse / np.sqrt(self.cv)

            # Find min MSE and its SE
            min_mse_idx = np.argmin(mean_mse)
            min_mse = mean_mse[min_mse_idx]
            min_se = se_mse[min_mse_idx]

            # Threshold = min_mse + 1*SE (MSE越小越好，所以是加)
            threshold = min_mse + min_se

            # Find all alphas where MSE <= threshold (within 1-SE of best)
            candidates_mask = mean_mse <= threshold
            candidate_indices = np.where(candidates_mask)[0]
            candidate_alphas = self._model.alphas_[candidates_mask]

            if len(candidate_indices) > 0:
                # For each valid alpha, fit and count non-zero coefficients
                # Select the one with FEWEST non-zeros (most sparse)
                n_nonzero_list = []
                for idx in candidate_indices:
                    alpha = self._model.alphas_[idx]
                    model_tmp = SklearnLasso(
                        alpha=alpha,
                        fit_intercept=False,
                        max_iter=self.max_iter,
                        tol=self.tol,
                    )
                    model_tmp.fit(X_scaled, y_centered, sample_weight=sample_weight)
                    n_nonzero = np.sum(model_tmp.coef_ != 0)
                    n_nonzero_list.append(n_nonzero)

                # Pick the alpha with minimum non-zeros
                best_candidate_idx = candidate_indices[np.argmin(n_nonzero_list)]
                best_alpha = float(self._model.alphas_[best_candidate_idx])
            else:
                best_alpha = float(self._model.alphas_[min_mse_idx])
        else:
            # Use standard min-MSE alpha
            best_alpha = float(self._model.alpha_)

        self.best_alpha_ = best_alpha

        # Refit with the selected alpha using sklearn Lasso
        final_model = SklearnLasso(
            alpha=self.best_alpha_,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        final_model.fit(X_scaled, y_centered, sample_weight=sample_weight)

        # Extract and scale coefficients
        self.coef_ = final_model.coef_ / self._X_std
        self.intercept_ = float(np.mean(y) - np.sum(final_model.coef_ * self._X_mean / self._X_std))

        # Store CV results
        self.cv_results_ = {
            'alphas': self._model.alphas_ if hasattr(self._model, 'alphas_') else self.alphas,
            'mse_path': self._model.mse_path_ if hasattr(self._model, 'mse_path_') else None,
        }

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