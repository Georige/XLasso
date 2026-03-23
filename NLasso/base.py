"""
NLasso 基类定义
遵循 scikit-learn Estimator 接口规范，性能优先设计
"""
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler

# 性能优化常量
_COPY_WHEN_POSSIBLE = False  # 优先使用视图而非拷贝
_DTYPE = np.float64  # 统一数值精度，平衡速度与精度


class BaseNLasso(BaseEstimator, ABC):
    """NLasso 基类，包含所有变体共享的参数与逻辑"""

    def __init__(
        self,
        lambda_ridge: float = 10.0,
        lambda_: float = None,
        lambda_path: list = None,
        n_lambda: int = 50,
        gamma: float = 0.3,
        s: float = 1.0,
        group_threshold: float = 0.7,
        group_min_size: int = 2,
        group_max_size: int = 10,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = True,
        fit_intercept: bool = True,
        n_jobs: int = -1,
        random_state: int = 2026,
        verbose: bool = False,
    ):
        # 超参数（注意：所有参数必须显式保存为属性，且名称与__init__参数完全一致）
        self.lambda_ridge = lambda_ridge
        self.lambda_ = lambda_
        self.lambda_path = lambda_path
        self.n_lambda = n_lambda
        self.gamma = gamma
        self.s = s
        self.group_threshold = group_threshold
        self.group_min_size = group_min_size
        self.group_max_size = group_max_size
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # 初始化拟合后属性
        self._init_fitted_attributes()

    def get_params(self, deep: bool = True) -> dict:
        """
        重写get_params以确保sklearn能正确识别所有参数
        解决ABC多重继承导致的参数识别问题
        """
        params = {
            'lambda_ridge': self.lambda_ridge,
            'lambda_': self.lambda_,
            'lambda_path': self.lambda_path,
            'n_lambda': self.n_lambda,
            'gamma': self.gamma,
            's': self.s,
            'group_threshold': self.group_threshold,
            'group_min_size': self.group_min_size,
            'group_max_size': self.group_max_size,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'standardize': self.standardize,
            'fit_intercept': self.fit_intercept,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }
        return params

    def set_params(self, **params):
        """
        重写set_params以确保sklearn能正确设置所有参数
        """
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    def _init_fitted_attributes(self):
        """初始化拟合后属性"""
        self.coef_ = None  # 原始特征空间系数 (p,)
        self.intercept_ = 0.0  # 截距项
        self.scaler_ = None  # 特征标准化器
        self.is_fitted_ = False
        self.n_features_in_ = None  # 输入特征数
        self.task_type_ = None  # 'regression' / 'classification'

    @abstractmethod
    def _fit_first_stage(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        第一阶段：强Ridge回归与LOO矩阵构造
        返回: (beta_ridge, X_loo, weights)
            beta_ridge: 强Ridge回归系数 (p,)
            X_loo: 留一引导矩阵 (n, p)
            weights: 非对称惩罚权重 (p, 2) -> [w_plus, w_minus]
        """
        pass

    @abstractmethod
    def _fit_group_module(self, X: np.ndarray, X_loo: np.ndarray) -> tuple:
        """
        组处理模块：高相关变量分组与正交分解
        返回: (X_loo_transformed, transform_info)
            X_loo_transformed: 变换后的引导矩阵 (n, p_new)
            transform_info: 系数还原所需的变换信息
        """
        pass

    @abstractmethod
    def _fit_second_stage(self, X_loo_transformed: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        第二阶段：非对称Lasso优化求解
        返回: theta: 变换空间系数 (p_new,)
        """
        pass

    @abstractmethod
    def _reconstruct_coefficients(self, theta: np.ndarray, transform_info: dict) -> np.ndarray:
        """
        系数还原：从变换空间映射回原始特征空间
        返回: coef: 原始特征系数 (p,)
        """
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        拟合NLasso模型
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 响应变量 (n_samples,)
            sample_weight: 样本权重 (n_samples,) 可选
        Returns:
            self: 拟合后的模型
        """
        # 性能优化：提前验证输入，避免后续重复检查
        X, y = check_X_y(
            X, y,
            accept_sparse=['csr', 'csc'],  # 支持稀疏矩阵
            dtype=_DTYPE,
            copy=_COPY_WHEN_POSSIBLE,
            ensure_2d=True,
            ensure_min_samples=2,
            ensure_min_features=2
        )
        self.n_features_in_ = X.shape[1]

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=_DTYPE)
            if sample_weight.shape != y.shape:
                raise ValueError(f"sample_weight shape {sample_weight.shape} must match y shape {y.shape}")

        # 1. 数据预处理（无拷贝优化）
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X = self.scaler_.fit_transform(X)

        # 2. 第一阶段：强Ridge + LOO矩阵 + 权重计算
        beta_ridge, X_loo, weights = self._fit_first_stage(X, y)
        if self.verbose:
            print(f"[Stage 1] Completed: Ridge max magnitude={np.max(np.abs(beta_ridge)):.4f}")

        # 3. 组处理模块：分组与正交分解
        X_loo_transformed, transform_info = self._fit_group_module(X, X_loo)
        if self.verbose:
            print(f"[Stage 2] Completed: {transform_info.get('n_groups', 0)} groups detected")

        # 4. 第二阶段：非对称Lasso求解
        theta = self._fit_second_stage(X_loo_transformed, y, weights)
        if self.verbose:
            print(f"[Stage 3] Completed: non-zero coefficients={np.sum(theta != 0):d}/{len(theta):d}")

        # 5. 系数还原到原始特征空间
        coef_standardized = self._reconstruct_coefficients(theta, transform_info)

        # 6. 逆标准化得到原始尺度系数
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
        """
        预测
        Args:
            X: 特征矩阵 (n_samples, n_features)
        Returns:
            y_pred: 预测值 (n_samples,)
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

        # 预测阶段性能优化：最小化计算量
        if self.standardize:
            X = self.scaler_.transform(X)

        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        模型评分（默认R²，分类任务可重写）
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def get_feature_importance(self) -> np.ndarray:
        """
        获取特征重要性（系数绝对值）
        """
        check_is_fitted(self, 'is_fitted_')
        return np.abs(self.coef_)

    def _more_tags(self):
        """
        sklearn兼容性标签
        """
        return {
            'requires_y': True,
            'allow_nan': False,
            'X_types': ['2darray', 'sparse'],
            'output_types': ['continuous', 'binary'],
        }


class NLassoRegressor(BaseNLasso, RegressorMixin):
    """NLasso 回归器"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_type_ = 'regression'


class NLassoClassifier(BaseNLasso, ClassifierMixin):
    """NLasso 分类器"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_type_ = 'classification'
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        # 分类任务额外处理类别
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("NLassoClassifier currently only supports binary classification")
        # 将y转为0/1编码
        y = (y == self.classes_[1]).astype(_DTYPE)
        return super().fit(X, y, sample_weight)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率（二分类）
        Returns: (n_samples, 2) -> [P(0), P(1)]
        """
        check_is_fitted(self, 'is_fitted_')
        z = super().predict(X)
        proba_1 = 1 / (1 + np.exp(-np.clip(z, -30, 30)))  # 数值稳定sigmoid
        return np.column_stack([1 - proba_1, proba_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        分类任务默认用准确率评分
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)
