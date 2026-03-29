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

        intercept_val = getattr(self.result_, 'intercept', None)
        if intercept_val is None:
            intercept_val = np.mean(y)
        self.intercept_ = float(np.asarray(intercept_val).ravel()[0])
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
        use_1se: bool = True,
    ):
        """
        Parameters:
            use_1se: If True, use 1-SE rule to select the most sparse model
                    within 1 standard error of the best score (default True).
        """
        super().__init__(standardize=standardize, fit_intercept=fit_intercept)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.n_folds = n_folds
        self.family = family
        self.lmda_min_ratio = lmda_min_ratio
        self.verbose = verbose
        self.random_state = random_state
        self.use_1se = use_1se
        self.best_lmda_: Optional[float] = None
        self.cv_results_: Optional[Dict] = None

    def fit(self, X: npt.NDArray, y: npt.NDArray, cv_splits=None, **kwargs) -> "UniLassoCV":
        """
        Fit UniLassoCV model with cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Training data
        y : np.ndarray
            Target values
        cv_splits : list of tuples, optional
            Pre-generated CV splits (list of (train_idx, val_idx) tuples).
            If provided, uses these splits instead of creating new KFold internally.
        """
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

        # 保存用于后续 1-SE 计算
        self._X_for_cv = X_scaled
        self._y_for_cv = y_centered

        if cv_splits is not None:
            # Use external CV splits for benchmark compatibility
            from unilasso.uni_lasso import cv_unilasso_with_splits
            self.result_ = cv_unilasso_with_splits(
                X=X_scaled, y=y_centered, cv_splits=cv_splits,
                family=self.family, lmda_min_ratio=self.lmda_min_ratio,
                verbose=self.verbose, seed=self.random_state,
            )
        else:
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
        intercept_val = getattr(self.result_, 'intercept', None)
        if intercept_val is None:
            intercept_val = np.mean(y)
        self.intercept_ = float(np.asarray(intercept_val).ravel()[0])
        self.coef_ = self.coef_ / self._X_std
        self.intercept_ = self.intercept_ + np.sum(self._X_mean * self.coef_)

        self.cv_results_ = {}
        if hasattr(self.result_, 'avg_losses'):
            self.cv_results_['avg_losses'] = self.result_.avg_losses
        if hasattr(self.result_, 'lmdas'):
            self.cv_results_['lmdas'] = self.result_.lmdas
        if hasattr(self.result_, 'std_losses'):
            self.cv_results_['std_losses'] = self.result_.std_losses

        # 1-SE rule: select the most sparse model within 1 std of best score
        if self.use_1se:
            best_lmda_1se = self._select_1se_lmda()
            if self.verbose and best_lmda_1se != self.best_lmda_:
                print(f"[UniLassoCV] Best lmda (CV): {self.best_lmda_:.6f}")
                print(f"[UniLassoCV] 1-SE lmda:      {best_lmda_1se:.6f}")
            self.best_lmda_ = best_lmda_1se
            # Re-fit with selected lambda to get correct coefficients
            self._fit_with_selected_lmda(X_scaled, y_centered, n_features)

        self.is_fitted_ = True
        return self

    def _select_1se_lmda(self) -> float:
        """
        1-SE 规则：选择最稀疏的模型，其分数在 best - 1*SE 范围内

        在候选中选择非零系数数量最少的（最稀疏的）
        """
        if 'avg_losses' not in self.cv_results_ or 'lmdas' not in self.cv_results_:
            return self.best_lmda_

        avg_losses = np.asarray(self.cv_results_['avg_losses'])
        lmdas = np.asarray(self.cv_results_['lmdas'])

        # 获取 std_losses，如果可用的话
        std_losses = None
        if 'std_losses' in self.cv_results_:
            std_losses = np.asarray(self.cv_results_['std_losses'])

        # 损失越低越好
        best_loss = np.min(avg_losses)

        if std_losses is not None:
            # 严格使用标准误 SE = std / sqrt(K)
            K = self.n_folds
            se_losses = std_losses / np.sqrt(K)
            # 阈值：best_loss + 1*SE (对于损失，越低越好，所以用加号)
            threshold = best_loss + se_losses
            # 找出所有通过 1-SE 检验的候选 lambda
            candidates_mask = avg_losses <= threshold
        else:
            # 如果没有 std，只能用 best
            candidates_mask = avg_losses == best_loss

        candidate_indices = np.where(candidates_mask)[0]
        if len(candidate_indices) == 0:
            return self.best_lmda_

        # 计算每个候选的非零系数数量，选择最稀疏的
        n_nonzero_list = []
        for idx in candidate_indices:
            lmda = lmdas[idx]
            # 临时创建模型获取系数
            lmdas_single = np.asarray([lmda])
            result = fit_unilasso(
                X=self._X_for_cv, y=self._y_for_cv, family=self.family,
                lmdas=lmdas_single, n_lmdas=1, lmda_min_ratio=1e-6, verbose=False,
            )
            if hasattr(result, 'coefs'):
                coef = result.coefs[0] if result.coefs.ndim > 1 else result.coefs
            else:
                coef = result.get('coefs', np.zeros(self.n_features_in_))
            n_nonzero = np.sum(np.abs(coef) > 1e-6)
            n_nonzero_list.append(n_nonzero)

        # 选择非零系数最少的（最稀疏的）
        best_candidate_idx = candidate_indices[np.argmin(n_nonzero_list)]
        return float(lmdas[best_candidate_idx])

    def _fit_with_selected_lmda(self, X_scaled: npt.NDArray, y_centered: npt.NDArray, n_features: int) -> None:
        """使用选定的 lambda 重新拟合模型"""
        lmdas = np.asarray([self.best_lmda_])
        result = fit_unilasso(
            X=X_scaled, y=y_centered, family=self.family, lmdas=lmdas,
            n_lmdas=1, lmda_min_ratio=1e-6, verbose=False,
        )

        if hasattr(result, 'coefs'):
            if result.coefs.ndim == 2:
                self.coef_ = result.coefs[0]
            else:
                self.coef_ = result.coefs
        else:
            self.coef_ = result.get('coefs', np.zeros(n_features))

        intercept_val = getattr(result, 'intercept', None)
        if intercept_val is None:
            intercept_val = np.mean(y_centered)
        self.intercept_ = float(np.asarray(intercept_val).ravel()[0])
        self.coef_ = self.coef_ / self._X_std
        self.intercept_ = self.intercept_ + np.sum(self._X_mean * self.coef_)

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
