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
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, lasso_path, LogisticRegression, LogisticRegressionCV
from sklearn.linear_model._logistic import _log_reg_scoring_path
from sklearn.metrics import log_loss
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
        # Step 4: L1 正则化逻辑回归（交叉熵极小化）
        # ================================================================
        # C 网格（LogisticRegression: C = 1/alpha，范围与 Lasso 对齐）
        alpha_max = np.max(np.abs(X_flipped.T @ y)) / len(y)
        alpha_min = alpha_max * self.alpha_min_ratio
        C_max = 1.0 / alpha_min  # 最宽松（最多信号）
        C_min = 1.0 / alpha_max  # 最严格（最多稀疏）
        Cs = np.logspace(np.log10(C_min), np.log10(C_max), self.n_alpha)

        # y 转为 ±1 编码（L1 逻辑回归需要）
        y_lr = np.where(y > 0.5, 1, -1).astype(_DTYPE)

        lr_cv = LogisticRegressionCV(
            Cs=Cs,
            cv=self.cv,
            penalty='l1',
            solver='saga',
            scoring='roc_auc',
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            n_jobs=-1,
        )
        lr_cv.fit(X_flipped, y_lr, sample_weight=sample_weight)
        best_C = lr_cv.C_[0]
        theta = lr_cv.coef_.ravel()
        self.best_C_ = best_C

        if self.verbose:
            print(f"[Step 4] L1 LogisticRegression done: best_C={best_C:.6f}, non-zero={np.sum(theta != 0)}/{len(theta)}")

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
        # Step 2: 纯净符号先验提取（Logistic Regression L2 正则化）
        # ================================================================
        # y 用 +1/-1 编码，LogisticRegression 天然输出概率 in [0,1]
        y_lr = np.where(y_continuous > 0.5, 1, -1).astype(_DTYPE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr_cv = LogisticRegressionCV(
                Cs=50,
                cv=self.cv,
                scoring='roc_auc',
                solver='lbfgs',
                max_iter=1000,
                random_state=self.random_state,
            )
            lr_cv.fit(X_std, y_lr)
        beta_lr = lr_cv.coef_.ravel()
        self.best_C_ = lr_cv.C_[0]

        signs = np.sign(beta_lr)
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
        fallback_min_nonzero: int = None,
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
        self.fallback_min_nonzero = fallback_min_nonzero

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
                cv=None  # GCV (Generalized Cross-Validation)
            )
            ridge_global.fit(X_global, y)
        beta_ridge_global = ridge_global.coef_
        self.best_lambda_ridge_ = ridge_global.alpha_

        signs_global = np.sign(beta_ridge_global)
        signs_global[signs_global == 0] = 1.0
        self.signs_ = signs_global  # 存储符号供外部计算符号准确率

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
        # 阶段一：K 折严格内部寻优（折间并行，折内用 GCV 选 lambda_ridge）
        # ================================================================
        eps = 1e-10  # 防止除零

        def _compute_single_fold_errors(fold_idx, train_idx, val_idx):
            """单 fold 并行计算：折内标准化 → GCV Ridge → 符号 → lasso_path → MSE 路径"""
            X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # Step 1: 折内标准化
            if self.standardize:
                scaler_fold = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
                X_tr = scaler_fold.fit_transform(X_tr_raw)
                X_va = scaler_fold.transform(X_va_raw)
            else:
                X_tr, X_va = X_tr_raw, X_va_raw

            # Step 2: 用 GCV 自动选最优 lambda_ridge
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ridge_cv = RidgeCV(
                    alphas=self.lambda_ridge_list,
                    cv=None,  # GCV (Generalized Cross-Validation)
                    fit_intercept=True
                )
                ridge_cv.fit(X_tr, y_tr)
            beta_ridge_fold = ridge_cv.coef_
            best_lr = ridge_cv.alpha_

            signs_fold = np.sign(beta_ridge_fold)
            signs_fold[signs_fold == 0] = 1.0

            # Step 3: 自适应加权 + 绝对象限映射
            raw_weights = 1.0 / (np.abs(beta_ridge_fold) + eps) ** self.gamma
            weights_fold = raw_weights / np.min(raw_weights)
            if self.weight_cap is not None:
                weights_fold = np.clip(weights_fold, 1.0, self.weight_cap)

            X_adaptive_tr = (X_tr * signs_fold) / weights_fold
            X_adaptive_va = (X_va * signs_fold) / weights_fold

            # Step 4: lasso_path（使用全局统一的 alphas 网格）
            y_tr_mean = np.mean(y_tr)
            y_tr_centered = y_tr - y_tr_mean

            _, coefs_path, _ = lasso_path(
                X_adaptive_tr, y_tr_centered,
                alphas=alphas,
                positive=True,
                max_iter=self.max_iter,
                tol=self.tol,
            )

            nselected_path = np.sum(coefs_path != 0, axis=0)

            # 验证集 MSE 路径
            preds_va = X_adaptive_va @ coefs_path
            if self.fit_intercept:
                preds_va += y_tr_mean

            mse_path = np.mean((y_va[:, np.newaxis] - preds_va) ** 2, axis=0)
            return fold_idx, mse_path, nselected_path, best_lr

        # joblib 并行执行所有 fold
        n_jobs = self.n_jobs if hasattr(self, 'n_jobs') and self.n_jobs is not None else -1
        parallel_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_compute_single_fold_errors)(fold_idx, train_idx, val_idx)
            for fold_idx, (train_idx, val_idx) in enumerate(splits)
        )

        # 收集结果
        for fold_idx, mse_path, nselected_path, best_lr in parallel_results:
            error_matrix[:, fold_idx] = mse_path
            nselected_matrix[:, fold_idx] = nselected_path
            if self.verbose:
                print(f"[Fold {fold_idx + 1}/{n_folds}] GCV selected lambda_ridge={best_lr:.4f}, mean_MSE={np.mean(mse_path):.6f}")

        if self.verbose:
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
        # 如果设置了 fallback_min_nonzero 且选中模型稀疏度过低，也退回 min MSE 模型
        elif (self.fallback_min_nonzero is not None and
              mean_nselected[best_alpha_idx] < self.fallback_min_nonzero):
            best_alpha_idx = min_mse_idx
            if self.verbose:
                print(f"[Stage 2] WARNING: 1-SE model has {mean_nselected[best_alpha_idx]:.0f} non-zeros "
                      f"(< {self.fallback_min_nonzero}), falling back to min_MSE alpha")
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

        # RidgeCV on 全量数据 → 最终符号 (使用 GCV)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ridge_final = RidgeCV(
                alphas=self.lambda_ridge_list,
                cv=None  # GCV (Generalized Cross-Validation)
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

        # Stage 3 兜底：全量拟合后若非零系数不足，退回 min_MSE alpha 重拟合
        n_nonzero = np.sum(self.coef_ != 0)
        should_fallback = n_nonzero == 0 or (
            self.fallback_min_nonzero is not None and n_nonzero < self.fallback_min_nonzero
        )
        if should_fallback:
            min_mse_alpha = alphas[min_mse_idx]
            if self.verbose:
                print(f"[Stage 3] WARNING: alpha={self.best_alpha_:.6e} yields {n_nonzero} non-zeros, "
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
    PFL 分类器 - CV 版本（高性能重构）

    重构要点（参考 BAFLClassifierCV FISTA 引擎）：
    1. 先验提取：边缘筛选（Marginal Correlation）替代 LogisticRegressionCV，
       速度更快，且与最终 FISTA 引擎逻辑一致
    2. 路径求解：自研 FISTA 逻辑回归路径，纯矩阵实现，天然非负 L1 约束
    3. 评价坐标：Log-Loss（可导、可反映概率信心）
    4. 截距还原：FISTA 直接输出截距，无需从 intercept_ 映射
    """

    def __init__(
        self,
        cv: int = 5,
        gamma: float = 1.0,
        weight_cap: float = None,
        alpha_min_ratio: float = 1e-4,
        n_alpha: int = 100,
        max_iter: int = 2000,
        tol: float = 1e-5,
        standardize: bool = True,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
        n_jobs: int = -1,
        # 以下为向后兼容参数（已废弃，仅吸收传入值不做任何操作）
        lambda_ridge_list=None,
    ):
        self.cv = cv
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
        self.classes_ = None
        self.signs_ = None
        self.best_alpha_ = None

    def get_params(self, deep: bool = True) -> dict:
        return {
            'cv': self.cv,
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
            'lambda_ridge_list': None,  # backward compat
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    # =====================================================================
    # 🚀 自研 FISTA 路径求解器：高性能纯矩阵实现
    #    - 逻辑回归梯度 Lipschitz = ||X'X||_2 / (4N)
    #    - 非负 L1 近端算子：ReLU(soft_threshold(x, t*alpha))
    #    - Nesterov 动量加速
    #    - 热启动（warm start）：沿 alpha 路径复用上一次解
    # =====================================================================
    def _fista_logistic_path(self, X, y, alphas, warm_start_theta=None, warm_start_b=None):
        """
        Parameters
        ----------
        X : np.ndarray (N, p)
            特征矩阵（已标准化、已 sign-flip、已加权）
        y : np.ndarray (N,)
            标签 {0, 1}
        alphas : np.ndarray
            正则化参数序列（从强到弱，即从稀疏到密集）
        warm_start_theta : np.ndarray, optional
            热启动系数向量（从上一个 alpha 复用）
        warm_start_b : float, optional
            热启动截距

        Returns
        -------
        coefs_path : np.ndarray (p, n_alphas)
            每列对应一个 alpha 的系数向量
        intercepts_path : np.ndarray (n_alphas,)
            每列对应一个 alpha 的截距
        """
        from scipy.special import expit

        N, p = X.shape
        n_alphas = len(alphas)
        coefs_path = np.zeros((p, n_alphas))
        intercepts_path = np.zeros(n_alphas)

        # --- 幂迭代法估算 Lipschitz 常数 L ---
        # 逻辑回归梯度的 Lipschitz 常数上限：||X'X||_2 / (4N)
        rng = np.random.RandomState(self.random_state)
        v = rng.randn(p)
        for _ in range(5):
            v = X.T @ (X @ v)
            v_norm = np.linalg.norm(v)
            if v_norm > 0:
                v = v / v_norm
        L = np.linalg.norm(X.T @ (X @ v)) / (4.0 * N)
        step_size = 1.0 / (L + 1e-8)  # 最佳 FISTA 步长 t = 1/L

        # --- FISTA 变量初始化 ---
        if warm_start_theta is None:
            theta = np.zeros(p)
        else:
            theta = warm_start_theta.copy()

        if warm_start_b is None:
            b = 0.0
        else:
            b = warm_start_b

        y_k = theta.copy()   # Nesterov 加速变量
        b_k = b
        t_k = 1.0            # Nesterov 进度参数

        # --- 沿 alpha 路径热启动滑动 ---
        for i, alpha in enumerate(alphas):
            for iteration in range(self.max_iter):
                theta_old = theta.copy()
                b_old = b

                # 梯度步（Gradient Step）：对数损失梯度
                sigma = expit(X @ y_k + b_k)
                err = sigma - y
                grad_theta = (X.T @ err) / N
                grad_b = np.mean(err)

                theta_step = y_k - step_size * grad_theta
                b_step = b_k - step_size * grad_b

                # 近端步（Proximal Step）：非负 L1 = ReLU ∘ soft_thresholding
                theta = np.maximum(0.0, theta_step - step_size * alpha)
                b = b_step  # 截距不受 L1 惩罚

                # Nesterov 动量加速
                t_k_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
                momentum = (t_k - 1.0) / t_k_next
                y_k = theta + momentum * (theta - theta_old)
                b_k = b + momentum * (b - b_old)

                # 收敛检测
                if (np.max(np.abs(theta - theta_old)) < self.tol and
                        np.abs(b - b_old) < self.tol):
                    break

            coefs_path[:, i] = theta
            intercepts_path[i] = b
            t_k = 1.0  # 每个 alpha 重新初始化 Nesterov 参数
            y_k = theta.copy()
            b_k = b

        return coefs_path, intercepts_path

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, cv_splits=None):
        from sklearn.model_selection import StratifiedKFold
        from scipy.special import expit

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

        # 严格二值化
        y = np.asarray(y).ravel()
        y_binary = np.where(y > 0.5, 1, 0)
        self.classes_ = np.array([0, 1])

        if len(np.unique(y_binary)) != 2:
            raise ValueError(f"PFLClassifierCV only supports binary classification, got labels: {np.unique(y_binary)}")

        y_continuous = y_binary.astype(_DTYPE)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=_DTYPE)

        # =================================================================
        # Stage 0: 全局统一 alpha 网格（所有 fold 共用）
        # 关键：全局标准化仅用于生成 alpha 网格，不参与 CV 寻优
        # =================================================================
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X_global = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_global = X.copy()

        # 边缘筛选先验（Marginal Correlation）：高效且与 FISTA 逻辑一致
        y_centered = y_continuous - np.mean(y_continuous)
        marginal_corr = X_global.T @ y_centered

        signs_global = np.sign(marginal_corr)
        signs_global[signs_global == 0] = 1.0

        raw_weights_global = 1.0 / (np.abs(marginal_corr) + 1e-10) ** self.gamma
        weights_global = raw_weights_global / np.min(raw_weights_global)
        if self.weight_cap is not None:
            weights_global = np.clip(weights_global, 1.0, self.weight_cap)

        X_adaptive_global = (X_global * signs_global) / weights_global

        # alpha 网格（从强到弱）
        alpha_max = np.max(np.abs(X_adaptive_global.T @ y_continuous)) / len(y_continuous)
        alpha_min = alpha_max * self.alpha_min_ratio
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

        self.signs_ = signs_global

        if self.verbose:
            print(f"[Stage 0] alpha grid: len={len(alphas)}, "
                  f"range=[{alphas[-1]:.2e}, {alphas[0]:.2e}], "
                  f"signs: +={np.sum(signs_global > 0)}, -={np.sum(signs_global < 0)}")

        # =================================================================
        # Stage 1: K 折严格隔离并行寻优
        # =================================================================
        if cv_splits is not None:
            n_folds = len(cv_splits)
            splits = cv_splits
        else:
            n_folds = self.cv
            skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            splits = list(skf.split(X, y_continuous))

        eps = 1e-10

        def _compute_single_fold(fold_idx, train_idx, val_idx):
            """单折并行：折内标准化 → 边缘筛选先验 → FISTA 路径 → Log-Loss 路径"""
            X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
            y_tr, y_va = y_continuous[train_idx], y_continuous[val_idx]

            # Step 1: 折内独立标准化（严格隔离）
            if self.standardize:
                scaler_fold = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
                X_tr = scaler_fold.fit_transform(X_tr_raw)
                X_va = scaler_fold.transform(X_va_raw)
            else:
                X_tr, X_va = X_tr_raw, X_va_raw

            # Step 2: 折内边缘筛选先验
            y_tr_c = y_tr - np.mean(y_tr)
            marg_corr_f = X_tr.T @ y_tr_c
            signs_f = np.sign(marg_corr_f)
            signs_f[signs_f == 0] = 1.0

            w_raw_f = 1.0 / (np.abs(marg_corr_f) + eps) ** self.gamma
            w_f = w_raw_f / np.min(w_raw_f)
            if self.weight_cap is not None:
                w_f = np.clip(w_f, 1.0, self.weight_cap)

            Xa_tr = (X_tr * signs_f) / w_f
            Xa_va = (X_va * signs_f) / w_f

            # Step 3: 🚀 FISTA 逻辑回归路径（热启动）
            c_path, b_path = self._fista_logistic_path(Xa_tr, y_tr, alphas)

            # Step 4: 验证集 Log-Loss 路径
            fold_losses = np.full(self.n_alpha, np.inf)
            for i in range(self.n_alpha):
                prob_va = expit(Xa_va @ c_path[:, i] + b_path[i])
                prob_va = np.clip(prob_va, 1e-15, 1.0 - 1e-15)
                fold_losses[i] = log_loss(y_va, prob_va)

            nselected = np.sum(c_path != 0, axis=0)
            return fold_idx, fold_losses, nselected

        # joblib 并行
        n_jobs = self.n_jobs if hasattr(self, 'n_jobs') and self.n_jobs is not None else -1
        parallel_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_compute_single_fold)(fold_idx, train_idx, val_idx)
            for fold_idx, (train_idx, val_idx) in enumerate(splits)
        )

        # 收集结果
        error_matrix = np.zeros((self.n_alpha, n_folds))
        nsel_matrix = np.zeros((self.n_alpha, n_folds))
        for fold_idx, fold_losses, nselected in parallel_results:
            error_matrix[:, fold_idx] = fold_losses
            nsel_matrix[:, fold_idx] = nselected
            if self.verbose:
                print(f"[Fold {fold_idx + 1}/{n_folds}] done")

        # =================================================================
        # Stage 2: 1-SE 法则选拔最优 alpha（Log-Loss，越小越好）
        # =================================================================
        mean_err = np.mean(error_matrix, axis=1)
        std_err = np.std(error_matrix, axis=1) / np.sqrt(n_folds)

        best_idx = np.argmin(mean_err)
        se_threshold = mean_err[best_idx] + std_err[best_idx]

        within_se = np.where(mean_err <= se_threshold)[0]
        mean_nselected = np.mean(nsel_matrix, axis=1)
        best_alpha_idx = int(within_se[np.argmin(mean_nselected[within_se])])
        self.best_alpha_ = alphas[best_alpha_idx]

        if self.verbose:
            print(f"\n[Stage 2] 1-SE rule: best_log_loss={mean_err[best_idx]:.6f}, "
                  f"threshold={se_threshold:.6f}, candidates={len(within_se)}, "
                  f"best_alpha={self.best_alpha_:.6e}, "
                  f"n_selected={int(mean_nselected[best_alpha_idx])}")

        # =================================================================
        # Stage 3: 全量数据终极拟合 + 时空逆转与截距还原
        # =================================================================
        # FISTA 全量拟合（单 alpha）
        c_final, b_final = self._fista_logistic_path(
            X_adaptive_global, y_continuous, [self.best_alpha_]
        )
        theta_final = c_final[:, 0]
        b_final = b_final[0]

        # 智能兜底：若 1-SE 导致全零，退回 Min Log-Loss 重拟合
        if np.sum(theta_final != 0) == 0:
            if self.verbose:
                print(f"[Stage 3] WARNING: 1-SE alpha={self.best_alpha_:.6e} yields "
                      f"0 non-zeros, falling back to min Log-Loss alpha")
            self.best_alpha_ = alphas[best_idx]
            c_final, b_final = self._fista_logistic_path(
                X_adaptive_global, y_continuous, [self.best_alpha_]
            )
            theta_final = c_final[:, 0]
            b_final = b_final[0]

        # 时空逆转：还原到标准化空间
        beta_std = (theta_final * signs_global) / weights_global

        if self.standardize:
            self.coef_ = beta_std / self.scaler_.scale_
            # FISTA 的截距建立在 X_adaptive_global（标准化空间）之上
            # 物理截距 = b_final - X_adaptive_global_mean @ theta_final
            X_adaptive_mean = np.mean(X_adaptive_global, axis=0)
            self.intercept_ = b_final - np.sum(X_adaptive_mean * theta_final)
        else:
            self.coef_ = beta_std
            self.intercept_ = b_final

        if self.verbose:
            print(f"[Stage 3] Final: alpha={self.best_alpha_:.6e}, "
                  f"non_zero={np.sum(self.coef_ != 0)}/{len(self.coef_)}, "
                  f"intercept={self.intercept_:.6f}")

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        from scipy.special import expit
        check_is_fitted(self, 'is_fitted_')
        z = X @ self.coef_ + self.intercept_
        proba_1 = expit(z)
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