"""
AdaptiveFlippedLasso 基类定义
遵循 scikit-learn Estimator 接口规范
"""
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# 性能优化常量
_COPY_WHEN_POSSIBLE = False
_DTYPE = np.float64


class BaseAdaptiveFlippedLasso(BaseEstimator, ABC):
    """
    AdaptiveFlippedLasso 基类

    算法流程：
    1. 第一阶段：强 Ridge 回归得到 beta_ridge
    2. 基于 |beta_ridge| 计算归一化权重
    3. 翻转特征方向使与 beta_ridge 同向
    4. 特征缩放：X_flipped / weights
    5. 非负 Lasso 拟合
    6. 逆重构：final_coef = (coef / weights) * signs
    """

    def __init__(
        self,
        lambda_ridge: float = 10.0,
        lambda_: float = 0.01,
        gamma: float = 1.0,
        alpha_min_ratio: float = 1e-4,
        n_alpha: int = 50,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = True,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
    ):
        # 超参数
        self.lambda_ridge = lambda_ridge
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose

        # 初始化拟合后属性
        self._init_fitted_attributes()

    def get_params(self, deep: bool = True) -> dict:
        params = {
            'lambda_ridge': self.lambda_ridge,
            'lambda_': self.lambda_,
            'gamma': self.gamma,
            'alpha_min_ratio': self.alpha_min_ratio,
            'n_alpha': self.n_alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'standardize': self.standardize,
            'fit_intercept': self.fit_intercept,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    def _init_fitted_attributes(self):
        """初始化拟合后属性"""
        self.coef_ = None
        self.intercept_ = 0.0
        self.scaler_ = None
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.task_type_ = None

    def _compute_first_stage(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        第一阶段：强 Ridge 回归 + 权重计算 + 特征翻转
        Returns:
            signs: 特征方向符号 (p,)
            weights: 归一化权重 (p,)
            X_flipped: 翻转后的特征矩阵 (n, p)
            beta_ridge: Ridge 系数 (p,)
        """
        from sklearn.linear_model import Ridge

        # 强 Ridge 回归
        ridge = Ridge(alpha=self.lambda_ridge, fit_intercept=True, random_state=self.random_state)
        ridge.fit(X, y)
        beta_ridge = ridge.coef_

        # 提取符号
        signs = np.sign(beta_ridge)
        signs[signs == 0] = 1.0  # 处理零值

        # 计算归一化权重（不裁剪，允许噪声特征的大权重无界增长）
        eps = 1e-5
        raw_weights = 1.0 / (np.abs(beta_ridge) + eps) ** self.gamma
        weights = raw_weights / np.mean(raw_weights)
        # 注意：不再裁剪到 [0,1]，因为噪声权重大于1是算法的关键

        # 翻转特征方向
        X_flipped = X * signs

        if self.verbose:
            print(f"[Stage 1] Ridge done: beta_ridge range [{np.min(beta_ridge):.4f}, {np.max(beta_ridge):.4f}]")
            print(f"[Stage 1] weights range [{np.min(weights):.4f}, {np.max(weights):.4f}]")

        return signs, weights, X_flipped, beta_ridge

    def _fit_second_stage(
        self,
        X_flipped: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        signs: np.ndarray,
    ) -> np.ndarray:
        """
        第二阶段：特征缩放 + 非负 Lasso 拟合 + 逆重构
        """
        # 特征缩放
        X_adaptive = X_flipped / weights

        # 计算 alpha 路径
        if self.lambda_ is not None:
            alphas = [self.lambda_]
        else:
            alpha_max = np.max(np.abs(X_adaptive.T @ y)) / len(y)
            alpha_min = alpha_max * self.alpha_min_ratio
            alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

        # 求解最优 alpha
        best_score = -np.inf
        best_coef = None
        best_alpha = None

        for alpha in alphas:
            model = Lasso(
                alpha=alpha,
                positive=True,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            model.fit(X_adaptive, y)
            score = model.score(X_adaptive, y)

            if score > best_score:
                best_score = score
                best_coef = model.coef_.copy()
                best_alpha = alpha

        if best_coef is None:
            best_coef = np.zeros(X_adaptive.shape[1])
            best_alpha = alphas[0]

        self.lambda_ = best_alpha

        if self.verbose:
            print(f"[Stage 2] Best alpha={best_alpha:.6f}, non-zero={np.sum(best_coef > 0)}/{len(best_coef)}")

        # 逆重构：还原到原始特征空间
        # final_coef = (model.coef_ / weights) * signs
        final_coef = (best_coef / weights) * signs

        return final_coef

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        拟合 AdaptiveFlippedLasso 模型
        """
        X, y = check_X_y(
            X, y,
            accept_sparse=['csr', 'csc'],
            dtype=_DTYPE,
            copy=_COPY_WHEN_POSSIBLE,
            ensure_2d=True,
            ensure_min_samples=2,
            ensure_min_features=2
        )
        self.n_features_in_ = X.shape[1]

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=_DTYPE)

        # 标准化
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X = self.scaler_.fit_transform(X)

        # 第一阶段
        signs, weights, X_flipped, beta_ridge = self._compute_first_stage(X, y)
        self.signs_ = signs
        self.weights_ = weights
        self.beta_ridge_ = beta_ridge

        # 第二阶段
        coef_standardized = self._fit_second_stage(X_flipped, y, weights, signs)

        # 逆标准化
        if self.standardize:
            self.coef_ = coef_standardized / self.scaler_.scale_
            if self.fit_intercept:
                self.intercept_ = self.scaler_.mean_[0] - np.sum(self.coef_ * self.scaler_.mean_)
        else:
            self.coef_ = coef_standardized
            if self.fit_intercept:
                self.intercept_ = np.mean(y) - np.mean(X @ coef_standardized)

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测

        注意：coef_ 在 fit() 时已经通过除以 scaler_.scale_ 转换到原始空间，
        因此预测时必须使用原始 X（未标准化）。
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(
            X,
            accept_sparse=['csr', 'csc'],
            dtype=_DTYPE,
            copy=_COPY_WHEN_POSSIBLE,
            ensure_2d=True
        )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")

        # 不再标准化 X，因为 coef_ 已经在原始空间
        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """默认 R² 评分"""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        check_is_fitted(self, 'is_fitted_')
        return np.abs(self.coef_)

    def _more_tags(self):
        return {
            'requires_y': True,
            'allow_nan': False,
            'X_types': ['2darray', 'sparse'],
            'output_types': ['continuous', 'binary'],
        }


class AdaptiveFlippedLassoRegressor(BaseAdaptiveFlippedLasso, RegressorMixin):
    """AdaptiveFlippedLasso 回归器"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_type_ = 'regression'


class AdaptiveFlippedLassoClassifier(BaseAdaptiveFlippedLasso, ClassifierMixin):
    """AdaptiveFlippedLasso 分类器"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_type_ = 'classification'
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("AdaptiveFlippedLassoClassifier only supports binary classification")
        y_continuous = (y == self.classes_[1]).astype(_DTYPE)
        return super().fit(X, y_continuous, sample_weight)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        z = super().predict(X)
        proba_1 = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - proba_1, proba_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)
