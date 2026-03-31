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
from unilasso.uni_lasso import cv_unilasso_with_splits

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
        self.intercept_ = self.intercept_ - np.sum(self._X_mean * self.coef_)

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
        n_jobs: int = 1,
    ):
        """
        Parameters:
            use_1se: If True, use 1-SE rule to select the most sparse model
                    within 1 standard error of the best score (default True).
            n_jobs: Number of parallel jobs for fold-level parallelism (-1 = all cores).
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
        self.n_jobs = n_jobs
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
                n_jobs=self.n_jobs,
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
        self.intercept_ = self.intercept_ - np.sum(self._X_mean * self.coef_)

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
        self.intercept_ = self.intercept_ - np.sum(self._X_mean * self.coef_)

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


# =============================================================================
# Wrapper classes for fit_unilasso and cv_unilasso to work with sweep.py config
# =============================================================================

class _UniLassoFunctionWrapper:
    """Base wrapper for unilasso functions to work with sklearn-style API."""

    func_name = None  # Override in subclass

    def __init__(self, **params):
        self._params = params
        self.__dict__.update(params)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.is_fitted_: bool = False
        self.n_features_in_: Optional[int] = None

    def get_params(self, deep: bool = True) -> dict:
        return self._params.copy()

    def set_params(self, **params) -> "_UniLassoFunctionWrapper":
        self._params.update(params)
        self.__dict__.update(params)
        return self

    def fit(self, X: npt.NDArray, y: npt.NDArray, **kwargs) -> "_UniLassoFunctionWrapper":
        raise NotImplementedError("Subclass must implement fit")

    def predict(self, X: npt.NDArray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.coef_ + self.intercept_


class UniLassoFit(_UniLassoFunctionWrapper):
    """
    Wrapper for fit_unilasso function.
    Works with sweep.py config: algo: fit_unilasso
    """
    func_name = "fit_unilasso"

    def fit(self, X: npt.NDArray, y: npt.NDArray, **kwargs) -> "UniLassoFit":
        """Fit using fit_unilasso."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Standardize X
        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)
        self._X_std[self._X_std < 1e-10] = 1.0
        X_scaled = (X - self._X_mean) / self._X_std

        y_centered = y - np.mean(y)

        # Build params for fit_unilasso
        params = {k: v for k, v in self._params.items() if v is not None}
        result = fit_unilasso(X=X_scaled, y=y_centered, **params)

        # Extract coefficients
        if hasattr(result, 'coefs'):
            self.coef_ = result.coefs
            if self.coef_.ndim > 1:
                self.coef_ = self.coef_.ravel()
        else:
            self.coef_ = result.get('coefs', np.zeros(self.n_features_in_))

        # Denormalize
        self.coef_ = self.coef_ / self._X_std

        # Intercept
        intercept_val = getattr(result, 'intercept', None)
        if intercept_val is None:
            intercept_val = np.mean(y)
        self.intercept_ = float(np.asarray(intercept_val).ravel()[0]) - np.sum(self._X_mean * self.coef_)

        # Store prior signs from univariate regression (beta attribute)
        self.prior_signs_: Optional[np.ndarray] = None
        self.sign_accuracy_: Optional[float] = None
        if hasattr(result, 'beta') and result.beta is not None:
            self.prior_signs_ = np.sign(result.beta)
            self.prior_signs_[self.prior_signs_ == 0] = 1.0

        self.is_fitted_ = True
        return self

    def ridge_sign_accuracy(self, beta_true: np.ndarray) -> dict:
        """
        Compute sign accuracy: fraction of true signals where prior sign matches true sign.
        For UniLasso, prior signs come from univariate regression (beta).
        """
        if self.prior_signs_ is None:
            return {"sign_accuracy": None, "correct": 0, "total": 0}

        true_signals_idx = np.where(beta_true != 0)[0]
        if len(true_signals_idx) == 0:
            return {"sign_accuracy": None, "correct": 0, "total": 0}

        correct = np.sum(self.prior_signs_[true_signals_idx] == np.sign(beta_true[true_signals_idx]))
        acc = correct / len(true_signals_idx)
        self.sign_accuracy_ = acc
        return {"sign_accuracy": acc, "correct": int(correct), "total": len(true_signals_idx)}


class UniLassoCVFit(_UniLassoFunctionWrapper):
    """
    Wrapper for cv_unilasso function with 1-SE rule.
    Works with sweep.py config: algo: cv_unilasso
    """
    func_name = "cv_unilasso"

    def __init__(self, **params):
        super().__init__(**params)
        self.use_1se_ = params.get('use_1se', True)
        self.n_folds_ = params.get('n_folds', 5)
        self.lmda_min_ratio_ = params.get('lmda_min_ratio', 1e-4)
        self.family_ = params.get('family', 'gaussian')
        self.seed_ = params.get('random_state', 42)

    def fit(self, X: npt.NDArray, y: npt.NDArray, cv_splits=None, **kwargs) -> "UniLassoCVFit":
        """Fit using cv_unilasso with optional 1-SE rule."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Standardize X
        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)
        self._X_std[self._X_std < 1e-10] = 1.0
        X_scaled = (X - self._X_mean) / self._X_std

        y_centered = y - np.mean(y)

        # Use external cv_splits if provided, otherwise let cv_unilasso generate internally
        if cv_splits is not None:
            cv_params = {
                'cv_splits': cv_splits,
                'family': self.family_,
                'lmda_min_ratio': self.lmda_min_ratio_,
                'seed': self.seed_,
            }
            cv_result = cv_unilasso_with_splits(X=X_scaled, y=y_centered, **cv_params)
        else:
            cv_params = {
                'family': self.family_,
                'n_folds': self.n_folds_,
                'lmda_min_ratio': self.lmda_min_ratio_,
                'seed': self.seed_,
            }
            cv_result = cv_unilasso(X=X_scaled, y=y_centered, **cv_params)

        # Extract best lambda (1-SE or min loss)
        if self.use_1se_ and hasattr(cv_result, 'cv_results_') and cv_result.cv_results_:
            cv_res = cv_result.cv_results_
            if 'lmdas' in cv_res and 'avg_losses' in cv_res:
                lmdas = np.asarray(cv_res['lmdas'])
                avg_losses = np.asarray(cv_res['avg_losses'])
                std_losses = np.asarray(cv_res.get('std_losses', np.zeros_like(avg_losses)))
                K = self.n_folds_
                se_losses = std_losses / np.sqrt(K)

                best_idx = np.argmin(avg_losses)
                threshold = avg_losses[best_idx] + se_losses[best_idx]
                candidates = np.where(avg_losses <= threshold)[0]
                best_lmda = float(lmdas[best_idx])  # fallback to min loss
        else:
            best_lmda = getattr(cv_result, 'best_lmda_', cv_result.lmda_ if hasattr(cv_result, 'lmda_') else 0.01)

        # Refit with best lambda
        fit_result = fit_unilasso(
            X=X_scaled, y=y_centered,
            family=self.family_,
            lmdas=[best_lmda],
            n_lmdas=1,
            lmda_min_ratio=1e-6,
            verbose=False,
        )

        # Extract coefficients
        if hasattr(fit_result, 'coefs'):
            self.coef_ = fit_result.coefs
            if self.coef_.ndim > 1:
                self.coef_ = self.coef_.ravel()
        else:
            self.coef_ = fit_result.get('coefs', np.zeros(self.n_features_in_))

        # Denormalize
        self.coef_ = self.coef_ / self._X_std

        # Intercept
        intercept_val = getattr(fit_result, 'intercept', None)
        if intercept_val is None:
            intercept_val = np.mean(y)
        self.intercept_ = float(np.asarray(intercept_val).ravel()[0]) - np.sum(self._X_mean * self.coef_)

        # Store prior signs from univariate regression (via get_beta())
        self.prior_signs_: Optional[np.ndarray] = None
        self.sign_accuracy_: Optional[float] = None
        try:
            beta_prior = cv_result.get_beta()
            if beta_prior is not None:
                self.prior_signs_ = np.sign(beta_prior)
                self.prior_signs_[self.prior_signs_ == 0] = 1.0
        except Exception:
            pass

        self.best_lmda_ = best_lmda
        self.is_fitted_ = True
        return self

    def score(self, X: npt.NDArray, y: npt.NDArray) -> float:
        """Calculate R-squared score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def ridge_sign_accuracy(self, beta_true: np.ndarray) -> dict:
        """
        Compute sign accuracy: fraction of true signals where prior sign matches true sign.
        For UniLasso, prior signs come from univariate regression (beta).
        """
        if self.prior_signs_ is None:
            return {"sign_accuracy": None, "correct": 0, "total": 0}

        true_signals_idx = np.where(beta_true != 0)[0]
        if len(true_signals_idx) == 0:
            return {"sign_accuracy": None, "correct": 0, "total": 0}

        correct = np.sum(self.prior_signs_[true_signals_idx] == np.sign(beta_true[true_signals_idx]))
        acc = correct / len(true_signals_idx)
        self.sign_accuracy_ = acc
        return {"sign_accuracy": acc, "correct": int(correct), "total": len(true_signals_idx)}
