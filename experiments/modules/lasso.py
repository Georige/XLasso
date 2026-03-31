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
from joblib import Parallel, delayed

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
        cv_splits=None,
        **kwargs
    ) -> "LassoCV":
        """
        Fit LassoCV model with cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Training data
        y : np.ndarray
            Target values
        sample_weight : np.ndarray, optional
            Sample weights
        cv_splits : list of tuples, optional
            Pre-generated CV splits (list of (train_idx, val_idx) tuples).
            If provided, uses these splits instead of creating new KFold.
        """
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

        if cv_splits is not None:
            # === 使用外部提供的 CV splits (手动 CV) ===
            self._fit_with_splits(X_scaled, y_centered, sample_weight, cv_splits)
        else:
            # === 使用 sklearn LassoCV 内部 CV ===
            self._fit_internal_cv(X_scaled, y_centered, sample_weight)

        # Final refit on all data with selected alpha
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

        self.is_fitted_ = True
        return self

    def _fit_with_splits(self, X_scaled, y_centered, sample_weight, cv_splits):
        """使用外部提供的 cv_splits 进行手动 1-SE 寻优。
        Fold 串行，fold 内部 alpha 并行 (joblib)。
        """
        n_folds = len(cv_splits)
        alphas = np.array(self.alphas) if self.alphas is not None else self._auto_alphas(X_scaled, y_centered)
        n_alphas = len(alphas)

        # mse_path shape: (n_alphas, n_folds)
        mse_path = np.zeros((n_alphas, n_folds))

        # === Fold 串行，alpha 并行 ===
        def _eval_alpha_on_fold(alpha, X_tr, y_tr, X_val, y_val, sw_tr):
            model = SklearnLasso(
                alpha=alpha,
                fit_intercept=False,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            model.fit(X_tr, y_tr, sample_weight=sw_tr)
            y_pred = model.predict(X_val)
            return float(np.mean((y_val - y_pred) ** 2))

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y_centered[train_idx], y_centered[val_idx]
            sw_tr = sample_weight[train_idx] if sample_weight is not None else None

            # 并行：对当前 fold 的所有 alpha 候选求 MSE
            mse_results = Parallel(n_jobs=-1, prefer="threads", verbose=0)(
                delayed(_eval_alpha_on_fold)(alpha, X_tr, y_tr, X_val, y_val, sw_tr)
                for alpha in alphas
            )
            for alpha_idx, mse_val in enumerate(mse_results):
                mse_path[alpha_idx, fold_idx] = mse_val

        # Store for reference
        self._mse_path = mse_path
        self._alphas = alphas

        # 1-SE rule
        mean_mse = mse_path.mean(axis=1)
        std_mse = mse_path.std(axis=1)
        se_mse = std_mse / np.sqrt(n_folds)

        min_mse_idx = np.argmin(mean_mse)
        min_mse = mean_mse[min_mse_idx]
        min_se = se_mse[min_mse_idx]
        threshold = min_mse + min_se

        candidates_mask = mean_mse <= threshold
        candidate_indices = np.where(candidates_mask)[0]

        if len(candidate_indices) > 0:
            n_nonzero_list = []
            for idx in candidate_indices:
                model_tmp = SklearnLasso(
                    alpha=alphas[idx],
                    fit_intercept=False,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                )
                model_tmp.fit(X_scaled, y_centered, sample_weight=sample_weight)
                n_nonzero_list.append(np.sum(model_tmp.coef_ != 0))

            best_candidate_idx = candidate_indices[np.argmin(n_nonzero_list)]
            best_alpha = float(alphas[best_candidate_idx])
        else:
            best_alpha = float(alphas[min_mse_idx])

        self.best_alpha_ = best_alpha
        self.cv_results_ = {
            'alphas': alphas,
            'mse_path': mse_path,
        }

    def _fit_internal_cv(self, X_scaled, y_centered, sample_weight):
        """使用 sklearn LassoCV 内部 CV (原始行为)。"""
        self._model = SklearnLassoCV(
            alphas=self.alphas,
            cv=self.cv,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            selection='random',
        )
        self._model.fit(X_scaled, y_centered, sample_weight=sample_weight)

        # Apply 1-SE rule manually if requested
        if self.use_1se and hasattr(self._model, 'mse_path_') and self._model.mse_path_ is not None:
            mse_path = self._model.mse_path_
            mean_mse = mse_path.mean(axis=1)
            std_mse = mse_path.std(axis=1)
            se_mse = std_mse / np.sqrt(self.cv)

            min_mse_idx = np.argmin(mean_mse)
            min_mse = mean_mse[min_mse_idx]
            min_se = se_mse[min_mse_idx]
            threshold = min_mse + min_se

            candidates_mask = mean_mse <= threshold
            candidate_indices = np.where(candidates_mask)[0]

            if len(candidate_indices) > 0:
                n_nonzero_list = []
                for idx in candidate_indices:
                    model_tmp = SklearnLasso(
                        alpha=self._model.alphas_[idx],
                        fit_intercept=False,
                        max_iter=self.max_iter,
                        tol=self.tol,
                    )
                    model_tmp.fit(X_scaled, y_centered, sample_weight=sample_weight)
                    n_nonzero_list.append(np.sum(model_tmp.coef_ != 0))

                best_candidate_idx = candidate_indices[np.argmin(n_nonzero_list)]
                best_alpha = float(self._model.alphas_[best_candidate_idx])
            else:
                best_alpha = float(self._model.alphas_[min_mse_idx])
        else:
            best_alpha = float(self._model.alpha_)

        self.best_alpha_ = best_alpha
        self.cv_results_ = {
            'alphas': self._model.alphas_ if hasattr(self._model, 'alphas_') else self.alphas,
            'mse_path': self._model.mse_path_ if hasattr(self._model, 'mse_path_') else None,
        }

    def _auto_alphas(self, X, y):
        """Generate alpha grid automatically when alphas is None."""
        lasso_tmp = SklearnLasso(alpha=0.001, max_iter=self.max_iter, random_state=self.random_state)
        lasso_tmp.fit(X, y)
        alpha_max = np.abs(lasso_tmp.coef_).max() if np.abs(lasso_tmp.coef_).max() > 0 else 1.0
        alpha_min = alpha_max * 0.0001
        return np.linspace(alpha_max, alpha_min, 30)

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