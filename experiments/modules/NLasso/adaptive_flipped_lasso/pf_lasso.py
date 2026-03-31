"""
PFL: Pure Flipped Lasso (纯符号约束 Lasso)

一种最简化的符号约束回归方法，仅利用 Ridge 回归的符号先验信息，
不做任何幅度加权，直接通过非负 Lasso 实现稀疏特征选择。

算法流程：
    Step 1: 全局数据标准化 (Global Standardization)
    Step 2: 纯净符号先验提取 (Pure Sign Extraction via RidgeCV)
    Step 3: 绝对象限映射 (Absolute Quadrant Mapping via Sign-Flipping)
    Step 4: 降维正交搜索与终极拟合 (Non-negative Constrained LassoCV)
    Step 5: 时空逆转与还原 (Space Restoration)

特性：
    - 零幅度加权：只利用符号信息，完全丢弃 Ridge 系数的大小
    - 硬符号约束：通过 sign-flipping 将所有特征映射到第一象限
    - 纯非负优化：标准 Lasso(positive=True)，无自适应权重
    - 极简主义：算法最简单，参数最少（仅 CV 折数 K）
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, lasso_path
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import warnings

# 性能优化常量
_COPY_WHEN_POSSIBLE = False
_DTYPE = np.float64


class PFLRegressor(BaseEstimator, RegressorMixin):
    """
    PFL 回归器：Pure Flipped Lasso (纯符号约束 Lasso)

    通过纯净符号先验引导，直接在第一象限进行非负 Lasso 优化。

    参数
    ----
    cv : int, default=5
        交叉验证折数，用于选择最优 Lasso alpha。
    lambda_ridge : float, default=1.0
        Ridge 回归的正则化强度（仅用于符号提取）。
    alpha_min_ratio : float, default=1e-4
        Lasso alpha 路径的最小值比例。
    n_alpha : int, default=100
        Lasso alpha 候选数量。
    max_iter : int, default=1000
        Lasso 最大迭代次数。
    tol : float, default=1e-4
        Lasso 收敛容忍度。
    standardize : bool, default=True
        是否在全局标准化特征（推荐 True）。
    fit_intercept : bool, default=True
        是否拟合截距。
    random_state : int, default=2026
        随机种子。
    verbose : bool, default=False
        是否输出详细信息。

    示例
    ----
    >>> from pf_lasso import PFLRegressor
    >>> model = PFLRegressor(cv=5, lambda_ridge=1.0)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        cv: int = 5,
        lambda_ridge: float = 1.0,
        alpha_min_ratio: float = 1e-4,
        n_alpha: int = 100,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = True,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
    ):
        self.cv = cv
        self.lambda_ridge = lambda_ridge
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose

        # 初始化属性
        self.coef_ = None
        self.intercept_ = 0.0
        self.scaler_ = None
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.signs_ = None  # Ridge 先验符号向量（sign-flipping 方向）

    def get_params(self, deep: bool = True) -> dict:
        return {
            'cv': self.cv,
            'lambda_ridge': self.lambda_ridge,
            'alpha_min_ratio': self.alpha_min_ratio,
            'n_alpha': self.n_alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'standardize': self.standardize,
            'fit_intercept': self.fit_intercept,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        拟合 PFL 模型

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)
        y : np.ndarray
            目标向量 (n_samples,)
        sample_weight : np.ndarray, optional
            样本权重 (n_samples,)

        Returns
        -------
        self : PFLRegressor
            拟合后的模型
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

        # ================================================================
        # Step 1: 全局数据标准化 (Global Standardization)
        # ================================================================
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X_std = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_std = X.copy()

        # ================================================================
        # Step 2: 纯净符号先验提取 (Pure Sign Extraction)
        # ================================================================
        # RidgeCV 自动选择最优 lambda_ridge
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ridge_cv = RidgeCV(
                alphas=np.logspace(-4, 4, 50),
                cv=self.cv,
                scoring='neg_mean_squared_error'
            )
            ridge_cv.fit(X_std, y)
        beta_ridge = ridge_cv.coef_
        self.best_lambda_ridge_ = ridge_cv.alpha_

        # 符号离散化：严格包含零值保护
        signs = np.sign(beta_ridge)
        signs[signs == 0] = 1.0  # 零值默认为正，避免特征被物理抹除

        if self.verbose:
            print(f"[Step 2] RidgeCV done: best_lambda_ridge={self.best_lambda_ridge_:.4f}")
            print(f"[Step 2] signs distribution: +1={np.sum(signs > 0)}, -1={np.sum(signs < 0)}")

        # ================================================================
        # Step 3: 绝对象限映射 (Absolute Quadrant Mapping)
        # ================================================================
        # 逐列符号翻转，统一映射至第一象限（非负象限）
        X_flipped = X_std * signs

        if self.verbose:
            print(f"[Step 3] X_flipped range: [{np.min(X_flipped):.4f}, {np.max(X_flipped):.4f}]")

        # ================================================================
        # Step 4: 降维正交搜索与终极拟合 (Non-negative LassoCV)
        # ================================================================
        # 计算 alpha 路径
        alpha_max = np.max(np.abs(X_flipped.T @ y)) / len(y)
        alpha_min = alpha_max * self.alpha_min_ratio
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

        # LassoCV 交叉验证选择最优 alpha
        lasso_cv = LassoCV(
            alphas=alphas,
            cv=self.cv,
            positive=True,  # 强制非负约束
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            n_jobs=-1,
        )
        lasso_cv.fit(X_flipped, y, sample_weight=sample_weight)

        # 最优 alpha 和非负系数
        best_alpha = lasso_cv.alpha_
        theta = lasso_cv.coef_

        if self.verbose:
            print(f"[Step 4] LassoCV done: best_alpha={best_alpha:.6f}, non-zero={np.sum(theta > 0)}/{len(theta)}")

        # ================================================================
        # Step 5: 时空逆转与还原 (Space Restoration)
        # ================================================================
        # 解除符号映射：还原到标准化空间
        # beta_std = theta * signs (因为 X_flipped = X_std * signs)
        beta_std = theta * signs

        # 解除标准化：还原到原始物理空间
        if self.standardize:
            self.coef_ = beta_std / self.scaler_.scale_
            if self.fit_intercept:
                self.intercept_ = np.mean(y) - np.sum(self.coef_ * self.scaler_.mean_)
        else:
            self.coef_ = beta_std
            if self.fit_intercept:
                self.intercept_ = np.mean(y) - np.mean(X @ beta_std)

        # 存储 Ridge 先验符号向量（供外部实验代码计算真正的符号准确率）
        self.signs_ = signs

        if self.verbose:
            print(f"[Step 5] Restoration done: coef range [{np.min(self.coef_):.6f}, {np.max(self.coef_):.6f}]")

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray
            预测值 (n_samples,)
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

        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """默认 R² 评分"""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性（系数绝对值）"""
        check_is_fitted(self, 'is_fitted_')
        return np.abs(self.coef_)

    def _more_tags(self):
        return {
            'requires_y': True,
            'allow_nan': False,
            'X_types': ['2darray', 'sparse'],
            'output_types': ['continuous'],
        }


class PFLClassifier(BaseEstimator, ClassifierMixin):
    """
    PFL 分类器：Pure Flipped Lasso 二分类版本

    参数同上。
    """

    def __init__(
        self,
        cv: int = 5,
        lambda_ridge: float = 1.0,
        alpha_min_ratio: float = 1e-4,
        n_alpha: int = 100,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = True,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
    ):
        self.cv = cv
        self.lambda_ridge = lambda_ridge
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose

        # 初始化属性
        self.coef_ = None
        self.intercept_ = 0.0
        self.scaler_ = None
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.classes_ = None
        self.signs_ = None

    def get_params(self, deep: bool = True) -> dict:
        return {
            'cv': self.cv,
            'lambda_ridge': self.lambda_ridge,
            'alpha_min_ratio': self.alpha_min_ratio,
            'n_alpha': self.n_alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'standardize': self.standardize,
            'fit_intercept': self.fit_intercept,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        拟合 PFL 分类模型（二分类）

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)
        y : np.ndarray
            目标向量 (n_samples,)，二分类标签
        sample_weight : np.ndarray, optional
            样本权重 (n_samples,)

        Returns
        -------
        self : PFLClassifier
            拟合后的模型
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

        # 严格二值化（防止高 sigma 时浮点噪声导致 unique > 2）
        y = np.asarray(y).ravel()
        y_binary = np.where(y > 0.5, 1, 0)
        self.classes_ = np.array([0, 1])

        if len(np.unique(y_binary)) != 2:
            raise ValueError(f"PFLClassifier only supports binary classification, got labels: {np.unique(y_binary)}")

        # 将 y 转为连续值（0/1 概率形式用于回归拟合）
        y_continuous = y_binary.astype(_DTYPE)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=_DTYPE)

        # ================================================================
        # Step 1: 全局数据标准化
        # ================================================================
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X_std = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_std = X.copy()

        # ================================================================
        # Step 2: 纯净符号先验提取
        # ================================================================
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ridge_cv = RidgeCV(
                alphas=np.logspace(-4, 4, 50),
                cv=self.cv,
                scoring='neg_mean_squared_error'
            )
            ridge_cv.fit(X_std, y_continuous)
        beta_ridge = ridge_cv.coef_
        self.best_lambda_ridge_ = ridge_cv.alpha_

        signs = np.sign(beta_ridge)
        signs[signs == 0] = 1.0

        # ================================================================
        # Step 3: 绝对象限映射
        # ================================================================
        X_flipped = X_std * signs

        # ================================================================
        # Step 4: 非负 LassoCV
        # ================================================================
        alpha_max = np.max(np.abs(X_flipped.T @ y_continuous)) / len(y_continuous)
        alpha_min = alpha_max * self.alpha_min_ratio
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

        lasso_cv = LassoCV(
            alphas=alphas,
            cv=self.cv,
            positive=True,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            n_jobs=-1,
        )
        lasso_cv.fit(X_flipped, y_continuous, sample_weight=sample_weight)

        best_alpha = lasso_cv.alpha_
        theta = lasso_cv.coef_

        # ================================================================
        # Step 5: 时空逆转与还原
        # ================================================================
        beta_std = theta * signs

        if self.standardize:
            self.coef_ = beta_std / self.scaler_.scale_
            if self.fit_intercept:
                self.intercept_ = np.mean(y_continuous) - np.sum(self.coef_ * self.scaler_.mean_)
        else:
            self.coef_ = beta_std
            if self.fit_intercept:
                self.intercept_ = np.mean(y_continuous) - np.mean(X @ beta_std)

        self.signs_ = signs

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率（二分类）

        Returns: (n_samples, 2) -> [P(0), P(1)]
        """
        check_is_fitted(self, 'is_fitted_')
        z = X @ self.coef_ + self.intercept_
        proba_1 = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - proba_1, proba_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        """
        check_is_fitted(self, 'is_fitted_')
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """默认准确率评分"""
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def _more_tags(self):
        return {
            'requires_y': True,
            'allow_nan': False,
            'X_types': ['2darray', 'sparse'],
            'output_types': ['binary'],
        }


class PFLRegressorCV(BaseEstimator, RegressorMixin):
    """
    PFL 回归器 - 全量 CV 版本（与 AdaptiveFlippedLassoCV 对齐）

    在每个 CV 折内独立进行符号提取和 LassoCV，选择全局最优 alpha。

    注意：与 PFLRegressor 的区别是使用 K-Fold 严格隔离来选择 alpha，
    而 PFLRegressor 在全量数据上做一次 LassoCV。
    """

    def __init__(
        self,
        cv: int = 5,
        lambda_ridge_list: tuple = (0.1, 1.0, 10.0, 100.0),
        gamma: float = 1.0,
        weight_cap: float = None,
        alpha_min_ratio: float = 1e-4,
        n_alpha: int = 100,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = False,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
        n_jobs: int = -1,
    ):
        self.cv = cv
        self.lambda_ridge_list = lambda_ridge_list
        self.gamma = gamma
        self.weight_cap = weight_cap
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        # 初始化属性
        self.coef_ = None
        self.intercept_ = 0.0
        self.scaler_ = None
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.best_alpha_ = None
        self.best_lambda_ridge_ = None
        self.signs_ = None

    def get_params(self, deep: bool = True) -> dict:
        return {
            'cv': self.cv,
            'lambda_ridge_list': self.lambda_ridge_list,
            'gamma': self.gamma,
            'weight_cap': self.weight_cap,
            'alpha_min_ratio': self.alpha_min_ratio,
            'n_alpha': self.n_alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'standardize': self.standardize,
            'fit_intercept': self.fit_intercept,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'n_jobs': self.n_jobs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, cv_splits=None):
        """
        拟合 PFLRegressorCV 模型

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (n_samples, n_features)
        y : np.ndarray
            目标向量 (n_samples,)
        sample_weight : np.ndarray, optional
            样本权重 (n_samples,)
        cv_splits : list of tuples, optional
            预先生成的 CV splits (list of (train_idx, val_idx) tuples)。
            如果提供，使用这些 splits 而非新建 KFold，确保所有算法使用相同的 CV splits 进行公平比较。
        """
        from sklearn.model_selection import KFold

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

        n = X.shape[0]

        # 注意：不在此处全局标准化！标准化在每个折内部独立进行（严格隔离）

        # 使用提供的 cv_splits 或创建新的
        if cv_splits is not None:
            n_folds = len(cv_splits)
            splits = cv_splits
        else:
            n_folds = self.cv
            kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            splits = list(kfold.split(X))

        # 错误矩阵：(n_alphas, n_folds)
        error_matrix = np.full((self.n_alpha, n_folds), np.inf)
        nselected_matrix = np.zeros((self.n_alpha, n_folds), dtype=int)

        # ================================================================
        # 阶段零：生成全局统一 alpha 网格（所有 fold 共用同一个序列）
        # 注意：必须在循环前生成，确保 Stage 2 的 mean(axis=1) 有意义
        # ================================================================
        if self.standardize:
            scaler_global = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X_global = scaler_global.fit_transform(X)
        else:
            X_global = X

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ridge_global = RidgeCV(
                alphas=self.lambda_ridge_list,
                cv=3,
                scoring='neg_mean_squared_error'
            )
            ridge_global.fit(X_global, y)
        beta_ridge_global = ridge_global.coef_

        signs_global = np.sign(beta_ridge_global)
        signs_global[signs_global == 0] = 1.0

        raw_weights_global = 1.0 / (np.abs(beta_ridge_global) + 1e-10) ** self.gamma
        weights_global = raw_weights_global / np.min(raw_weights_global)
        if self.weight_cap is not None:
            weights_global = np.clip(weights_global, 1.0, self.weight_cap)

        X_adaptive_global = (X_global * signs_global) / weights_global

        alpha_max = np.max(np.abs(X_adaptive_global.T @ y)) / len(y)
        alpha_min = alpha_max * self.alpha_min_ratio
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

        if self.verbose:
            print(f"[Stage 0] Global alpha grid: len={len(alphas)}, "
                  f"range=[{alphas[-1]:.2e}, {alphas[0]:.2e}]")

        # ================================================================
        # 阶段一：K 折严格内部寻优（joblib 并行 + lasso_path 向量化）
        # ================================================================
        eps = 1e-10  # 防止除零

        def _compute_single_fold_errors(fold_idx, train_idx, val_idx):
            """单 fold 并行计算：标准化 → 符号先验 → 自适应加权 → lasso_path → MSE 路径"""
            X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # Step 1: 每折内部独立标准化（严格隔离，禁止数据泄露）
            if self.standardize:
                scaler_fold = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
                X_tr = scaler_fold.fit_transform(X_tr_raw)
                X_va = scaler_fold.transform(X_va_raw)
            else:
                X_tr, X_va = X_tr_raw, X_va_raw

            # Step 2: 纯净符号先验提取（仅在训练折上）
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ridge_cv = RidgeCV(
                    alphas=self.lambda_ridge_list,
                    cv=3,
                    scoring='neg_mean_squared_error'
                )
                ridge_cv.fit(X_tr, y_tr)
            beta_ridge_fold = ridge_cv.coef_

            signs_fold = np.sign(beta_ridge_fold)
            signs_fold[signs_fold == 0] = 1.0

            # Step 3: 自适应加权 + 绝对象限映射（Min-Anchored 归一化）
            raw_weights = 1.0 / (np.abs(beta_ridge_fold) + eps) ** self.gamma
            weights_fold = raw_weights / np.min(raw_weights)
            if self.weight_cap is not None:
                weights_fold = np.clip(weights_fold, 1.0, self.weight_cap)

            X_adaptive_tr = (X_tr * signs_fold) / weights_fold
            X_adaptive_va = (X_va * signs_fold) / weights_fold

            # Step 4: lasso_path 向量化（使用全局统一的 alphas 网格，y 须中心化）
            y_tr_mean = np.mean(y_tr)
            y_tr_centered = y_tr - y_tr_mean

            _, coefs_path, _ = lasso_path(
                X_adaptive_tr, y_tr_centered,
                alphas=alphas,
                positive=True,
                max_iter=self.max_iter,
                tol=self.tol,
            )

            # 非零系数数量路径
            nselected_path = np.sum(coefs_path != 0, axis=0)

            # 验证集 MSE 路径（向量化，截距由 y 均值近似）
            preds_va = X_adaptive_va @ coefs_path
            if self.fit_intercept:
                preds_va += y_tr_mean

            mse_path = np.mean((y_va[:, np.newaxis] - preds_va) ** 2, axis=0)
            return fold_idx, mse_path, nselected_path

        # joblib 并行执行所有 fold
        n_jobs = self.n_jobs if hasattr(self, 'n_jobs') and self.n_jobs is not None else -1
        parallel_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_compute_single_fold_errors)(fold_idx, train_idx, val_idx)
            for fold_idx, (train_idx, val_idx) in enumerate(splits)
        )

        # 收集结果（所有 fold 共用全局 alphas，无须再追踪）
        for fold_idx, mse_path, nselected_path in parallel_results:
            error_matrix[:, fold_idx] = mse_path
            nselected_matrix[:, fold_idx] = nselected_path
            if self.verbose:
                print(f"[Fold {fold_idx + 1}/{n_folds}] completed (parallel)")

        if self.verbose and n_folds > 0:
            print(f"[Stage 1] Parallel fold computation done, n_folds={n_folds}")

        # ================================================================
        # 阶段二：1-SE 法则选拔最优 alpha
        # 在 min_MSE + 1*SE 范围内，选择非零系数最少的模型（最简模型）
        # ================================================================
        mean_error = np.mean(error_matrix, axis=1)
        std_error = np.std(error_matrix, axis=1) / np.sqrt(n_folds)

        min_mse = np.min(mean_error)
        min_mse_idx = np.argmin(mean_error)
        se_threshold = min_mse + std_error[min_mse_idx]

        # 1-SE 范围内的候选 alpha indices
        within_se = np.where(mean_error <= se_threshold)[0]
        # 这些候选中取非零系数均值最小的
        mean_nselected = np.mean(nselected_matrix, axis=1)
        best_alpha_idx = int(within_se[np.argmin(mean_nselected[within_se])])

        # 如果 1-SE 选中的模型非零系数为 0，退回 min MSE 模型
        if mean_nselected[best_alpha_idx] == 0:
            best_alpha_idx = min_mse_idx
            if self.verbose:
                print(f"[Stage 2] WARNING: 1-SE model has 0 non-zeros, falling back to min_MSE alpha")
        self.best_alpha_ = alphas[best_alpha_idx]

        if self.verbose:
            print(f"\n[Stage 2] 1-SE rule: min_MSE={min_mse:.6f}, threshold={se_threshold:.6f}, "
                  f"candidates={len(within_se)}, best_alpha={self.best_alpha_:.6f}, "
                  f"n_selected={int(mean_nselected[best_alpha_idx])}")

        # ================================================================
        # 阶段三：全量数据终极拟合
        # ================================================================
        # Step 1: 全量数据标准化（用于最终模型）
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X_std = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_std = X

        # RidgeCV on 全量数据 → 最终符号
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ridge_final = RidgeCV(
                alphas=self.lambda_ridge_list,
                cv=self.cv,
                scoring='neg_mean_squared_error'
            )
            ridge_final.fit(X_std, y)
        beta_ridge_final = ridge_final.coef_
        self.best_lambda_ridge_ = ridge_final.alpha_

        signs_final = np.sign(beta_ridge_final)
        signs_final[signs_final == 0] = 1.0

        # 自适应加权（带硬上限 cap）+ Min-Anchored 归一化（与 Stage 1 一致）
        raw_weights_final = 1.0 / (np.abs(beta_ridge_final) + eps) ** self.gamma
        raw_weights_final = raw_weights_final / np.min(raw_weights_final)
        if self.weight_cap is not None:
            raw_weights_final = np.clip(raw_weights_final, 1.0, self.weight_cap)
        self.weights_ = raw_weights_final

        # 最终自适应空间变换：X_signflip / w_j
        X_adaptive_final = (X_std * signs_final) / raw_weights_final

        # 最终 Lasso 拟合
        lasso_final = Lasso(
            alpha=self.best_alpha_,
            positive=True,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        lasso_final.fit(X_adaptive_final, y, sample_weight=sample_weight)
        theta_final = lasso_final.coef_

        # 逆重构：β_std = θ * signs / w_j
        beta_std = theta_final * signs_final / raw_weights_final

        # 逆标准化
        if self.standardize:
            self.coef_ = beta_std / self.scaler_.scale_
        else:
            self.coef_ = beta_std

        # 截距还原：lasso_final.intercept_ 建立在 X_adaptive_final (非零均值) 之上
        # 正确公式：intercept_physical = np.mean(y) - X_adaptive_final_mean @ theta_final
        if self.fit_intercept:
            X_adaptive_mean = np.mean(X_adaptive_final, axis=0)
            self.intercept_ = float(np.mean(y) - np.sum(X_adaptive_mean * theta_final))
        else:
            self.intercept_ = float(np.mean(y))

        # Stage 3 兜底：全量拟合后若非零系数为 0，退回 min_MSE alpha 重拟合
        if np.sum(self.coef_ != 0) == 0:
            min_mse_alpha = alphas[min_mse_idx]
            if self.verbose:
                print(f"[Stage 3] WARNING: alpha={self.best_alpha_:.6e} yields 0 non-zeros, "
                      f"falling back to min_MSE alpha={min_mse_alpha:.6e}")
            lasso_fallback = Lasso(
                alpha=min_mse_alpha,
                positive=True,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            lasso_fallback.fit(X_adaptive_final, y, sample_weight=sample_weight)
            theta_fallback = lasso_fallback.coef_
            beta_std_fallback = theta_fallback * signs_final / raw_weights_final
            if self.standardize:
                self.coef_ = beta_std_fallback / self.scaler_.scale_
            else:
                self.coef_ = beta_std_fallback
            if self.fit_intercept:
                X_adaptive_mean_fb = np.mean(X_adaptive_final, axis=0)
                self.intercept_ = float(np.mean(y) - np.sum(X_adaptive_mean_fb * theta_fallback))
            else:
                self.intercept_ = float(np.mean(y))
            self.best_alpha_ = min_mse_alpha

        # 存储 Ridge 先验符号向量（供外部 god's eye 评估）
        self.signs_ = signs_final

        if self.verbose:
            print(f"[Stage 3] Final: lambda_ridge={self.best_lambda_ridge_}, alpha={self.best_alpha_:.6f}, "
                  f"non_zero={np.sum(self.coef_ != 0)}/{len(self.coef_)}")

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
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
            'output_types': ['continuous'],
        }


class PFLClassifierCV(PFLClassifier):
    """
    PFL 分类器 - CV 版本
    """

    def __init__(
        self,
        cv: int = 5,
        lambda_ridge_list: tuple = (0.1, 1.0, 10.0, 100.0),
        gamma: float = 1.0,
        weight_cap: float = None,
        alpha_min_ratio: float = 1e-4,
        n_alpha: int = 100,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = False,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
        n_jobs: int = -1,
    ):
        self.cv = cv
        self.lambda_ridge_list = lambda_ridge_list
        self.gamma = gamma
        self.weight_cap = weight_cap
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        # 初始化属性（供父类方法使用）
        self.coef_ = None
        self.intercept_ = 0.0
        self.scaler_ = None
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.classes_ = None
        self.signs_ = None
        self.best_alpha_ = None
        self.best_lambda_ridge_ = None

    def get_params(self, deep: bool = True) -> dict:
        return {
            'cv': self.cv,
            'lambda_ridge_list': self.lambda_ridge_list,
            'gamma': self.gamma,
            'weight_cap': self.weight_cap,
            'alpha_min_ratio': self.alpha_min_ratio,
            'n_alpha': self.n_alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'standardize': self.standardize,
            'fit_intercept': self.fit_intercept,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'n_jobs': self.n_jobs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, cv_splits=None):
        from sklearn.model_selection import KFold

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

        # 严格二值化（防止高 sigma 时浮点噪声导致 unique > 2）
        y = np.asarray(y).ravel()
        y_binary = np.where(y > 0.5, 1, 0)
        self.classes_ = np.array([0, 1])

        if len(np.unique(y_binary)) != 2:
            raise ValueError(f"PFLClassifierCV only supports binary classification, got labels: {np.unique(y_binary)}")

        # 将 y 转为连续值（0/1 概率形式用于回归拟合）
        y_continuous = y_binary.astype(_DTYPE)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=_DTYPE)

        # 注意：不在此处全局标准化！标准化在每个折内部独立进行（严格隔离）

        # 使用提供的 cv_splits 或创建新的
        if cv_splits is not None:
            n_folds = len(cv_splits)
            splits = cv_splits
        else:
            n_folds = self.cv
            kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            splits = list(kfold.split(X))

        # 错误矩阵
        error_matrix = np.full((self.n_alpha, n_folds), np.inf)
        nselected_matrix = np.zeros((self.n_alpha, n_folds), dtype=int)

        # ================================================================
        # 阶段零：生成全局统一 alpha 网格（所有 fold 共用同一个序列）
        # ================================================================
        if self.standardize:
            scaler_global = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X_global = scaler_global.fit_transform(X)
        else:
            X_global = X

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ridge_global = RidgeCV(
                alphas=self.lambda_ridge_list,
                cv=3,
                scoring='neg_mean_squared_error'
            )
            ridge_global.fit(X_global, y_continuous)
        beta_ridge_global = ridge_global.coef_

        signs_global = np.sign(beta_ridge_global)
        signs_global[signs_global == 0] = 1.0

        raw_weights_global = 1.0 / (np.abs(beta_ridge_global) + 1e-10) ** self.gamma
        weights_global = raw_weights_global / np.min(raw_weights_global)
        if self.weight_cap is not None:
            weights_global = np.clip(weights_global, 1.0, self.weight_cap)

        X_adaptive_global = (X_global * signs_global) / weights_global

        alpha_max = np.max(np.abs(X_adaptive_global.T @ y_continuous)) / len(y_continuous)
        alpha_min = alpha_max * self.alpha_min_ratio
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

        # ================================================================
        # 阶段一：K 折严格内部寻优（joblib 并行 + lasso_path 向量化）
        # ================================================================
        eps = 1e-10

        def _compute_single_fold_errors(fold_idx, train_idx, val_idx):
            """单 fold 并行计算：标准化 → 符号先验 → 自适应加权 → lasso_path → MSE 路径"""
            X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
            y_tr, y_va = y_continuous[train_idx], y_continuous[val_idx]

            # Step 1: 每折内部独立标准化
            if self.standardize:
                scaler_fold = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
                X_tr = scaler_fold.fit_transform(X_tr_raw)
                X_va = scaler_fold.transform(X_va_raw)
            else:
                X_tr, X_va = X_tr_raw, X_va_raw

            # Step 2: 纯净符号先验提取
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ridge_cv = RidgeCV(
                    alphas=self.lambda_ridge_list,
                    cv=3,
                    scoring='neg_mean_squared_error'
                )
                ridge_cv.fit(X_tr, y_tr)
            beta_ridge_fold = ridge_cv.coef_

            signs_fold = np.sign(beta_ridge_fold)
            signs_fold[signs_fold == 0] = 1.0

            # Step 3: 自适应加权 + 绝对象限映射（Min-Anchored 归一化）
            raw_weights = 1.0 / (np.abs(beta_ridge_fold) + eps) ** self.gamma
            weights_fold = raw_weights / np.min(raw_weights)
            if self.weight_cap is not None:
                weights_fold = np.clip(weights_fold, 1.0, self.weight_cap)

            X_adaptive_tr = (X_tr * signs_fold) / weights_fold
            X_adaptive_va = (X_va * signs_fold) / weights_fold

            # Step 4: lasso_path 向量化（使用全局统一的 alphas 网格，y 须中心化）
            y_tr_mean = np.mean(y_tr)
            y_tr_centered = y_tr - y_tr_mean

            _, coefs_path, _ = lasso_path(
                X_adaptive_tr, y_tr_centered,
                alphas=alphas,
                positive=True,
                max_iter=self.max_iter,
                tol=self.tol,
            )

            # 非零系数数量路径
            nselected_path = np.sum(coefs_path != 0, axis=0)

            # 验证集 MSE 路径（向量化，截距由 y 均值近似）
            preds_va = X_adaptive_va @ coefs_path
            if self.fit_intercept:
                preds_va += y_tr_mean

            mse_path = np.mean((y_va[:, np.newaxis] - preds_va) ** 2, axis=0)
            return fold_idx, mse_path, nselected_path

        # joblib 并行执行所有 fold
        n_jobs = self.n_jobs if hasattr(self, 'n_jobs') and self.n_jobs is not None else -1
        parallel_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_compute_single_fold_errors)(fold_idx, train_idx, val_idx)
            for fold_idx, (train_idx, val_idx) in enumerate(splits)
        )

        # 收集结果（所有 fold 共用全局 alphas，无须再追踪）
        for fold_idx, mse_path, nselected_path in parallel_results:
            error_matrix[:, fold_idx] = mse_path
            nselected_matrix[:, fold_idx] = nselected_path
            if self.verbose:
                print(f"[Fold {fold_idx + 1}/{n_folds}] completed (parallel)")

        # ================================================================
        # 阶段二：1-SE 法则选拔最优 alpha
        # 在 min_MSE + 1*SE 范围内，选择非零系数最少的模型（最简模型）
        # ================================================================
        mean_error = np.mean(error_matrix, axis=1)
        std_error = np.std(error_matrix, axis=1) / np.sqrt(n_folds)

        min_mse = np.min(mean_error)
        min_mse_idx = np.argmin(mean_error)
        se_threshold = min_mse + std_error[min_mse_idx]

        within_se = np.where(mean_error <= se_threshold)[0]
        mean_nselected = np.mean(nselected_matrix, axis=1)
        best_alpha_idx = int(within_se[np.argmin(mean_nselected[within_se])])
        self.best_alpha_ = alphas[best_alpha_idx]

        if self.verbose:
            print(f"\n[Stage 2] 1-SE rule: min_MSE={min_mse:.6f}, threshold={se_threshold:.6f}, "
                  f"candidates={len(within_se)}, best_alpha={self.best_alpha_:.6f}, "
                  f"n_selected={int(mean_nselected[best_alpha_idx])}")

        # ================================================================
        # 阶段三：全量数据终极拟合
        # ================================================================
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X_std = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_std = X

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ridge_final = RidgeCV(
                alphas=self.lambda_ridge_list,
                cv=self.cv,
                scoring='neg_mean_squared_error'
            )
            ridge_final.fit(X_std, y_continuous)
        beta_ridge_final = ridge_final.coef_
        self.best_lambda_ridge_ = ridge_final.alpha_

        signs_final = np.sign(beta_ridge_final)
        signs_final[signs_final == 0] = 1.0

        # 自适应加权（带硬上限 cap）+ Min-Anchored 归一化（与 Stage 1 一致）
        raw_weights_final = 1.0 / (np.abs(beta_ridge_final) + eps) ** self.gamma
        raw_weights_final = raw_weights_final / np.min(raw_weights_final)
        if self.weight_cap is not None:
            raw_weights_final = np.clip(raw_weights_final, 1.0, self.weight_cap)
        self.weights_ = raw_weights_final

        X_adaptive_final = (X_std * signs_final) / raw_weights_final

        lasso_final = Lasso(
            alpha=self.best_alpha_,
            positive=True,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        lasso_final.fit(X_adaptive_final, y_continuous, sample_weight=sample_weight)
        theta_final = lasso_final.coef_

        beta_std = theta_final * signs_final / raw_weights_final

        if self.standardize:
            self.coef_ = beta_std / self.scaler_.scale_
        else:
            self.coef_ = beta_std

        # 截距还原：lasso_final.intercept_ 建立在 X_adaptive_final (非零均值) 之上
        # 正确公式：intercept_physical = np.mean(y) - X_adaptive_final_mean @ theta_final
        if self.fit_intercept:
            X_adaptive_mean = np.mean(X_adaptive_final, axis=0)
            self.intercept_ = float(np.mean(y_continuous) - np.sum(X_adaptive_mean * theta_final))
        else:
            self.intercept_ = float(np.mean(y_continuous))

        self.signs_ = signs_final

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        z = X @ self.coef_ + self.intercept_
        proba_1 = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - proba_1, proba_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def _more_tags(self):
        return {
            'requires_y': True,
            'allow_nan': False,
            'X_types': ['2darray', 'sparse'],
            'output_types': ['binary'],
        }