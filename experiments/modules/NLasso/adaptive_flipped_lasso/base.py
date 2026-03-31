"""
AdaptiveFlippedLasso 基类定义
遵循 scikit-learn Estimator 接口规范
"""
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV, lasso_path, LinearRegression
from joblib import Parallel, delayed

# 性能优化常量
_COPY_WHEN_POSSIBLE = False
_DTYPE = np.float64


def _detect_collinearity_mode(X):
    """
    Detect data collinearity structure via SVD spectral decay.
    Returns 'dense_collinear' or 'sparse_independent'.
    """
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    energy_ratio = s[0] / np.sum(s)
    if energy_ratio > 0.40:
        return "dense_collinear", energy_ratio
    else:
        return "sparse_independent", energy_ratio


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
        weight_clip_max: float = 100.0,  # Max value for weight clipping, set None to disable
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
        # Guard: convert string 'null' to Python None (handles YAML parsing edge cases)
        self.weight_clip_max = None if (weight_clip_max is None or weight_clip_max == 'null') else weight_clip_max

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

        # Min-Anchored Normalization - anchors strongest signal at 1.0
        eps = 1e-5
        raw_weights = 1.0 / (np.abs(beta_ridge) + eps) ** self.gamma
        min_w = np.min(raw_weights)
        w_normalized = raw_weights / min_w
        clip_max = self.weight_clip_max if self.weight_clip_max is not None else float('inf')
        weights = np.clip(w_normalized, 1.0, clip_max)

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
                self.intercept_ = np.mean(y) - np.sum(self.coef_ * self.scaler_.mean_)
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


class AdaptiveFlippedLassoCV(BaseAdaptiveFlippedLasso, RegressorMixin):
    """
    严格隔离的 AdaptiveFlippedLasso 交叉验证

    算法流程（Strict NLasso CV）：
    阶段一：K 折严格内部寻优
        for each fold k:
            严格隔离训练折 / 验证折
            RidgeCV(仅训练折) → beta_ridge_k
            for each gamma:
                计算 weights_k（基于训练折 Ridge）
                扭曲训练折和验证折
                LassoPath(训练折) → 全部 alpha 的系数路径
                在验证折上打分 → Error_Matrix[gamma, alpha, k]
    阶段二：选拔最优参数（平均 MSE 最小的 gamma, alpha）
    阶段三：全量数据终极拟合（带回最佳参数）

    支持与 sklearn.model_selection.GridSearchCV 配合使用
    """

    def __init__(
        self,
        lambda_ridge_list: tuple = (0.1, 1.0, 10.0, 100.0),
        gamma_list: tuple = (0.5, 1.0, 2.0),
        cv: int = 5,
        alpha_min_ratio: float = 1e-4,
        n_alpha: int = 100,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = False,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
        use_post_ols_debiasing: bool = False,  # Post-selection OLS debiasing
        auto_tune_collinearity: bool = True,  # Auto-detect collinearity and adjust gamma_list
        weight_clip_max: float = 100.0,  # Max value for weight clipping, set None to disable
        eps: float = 1e-5,  # Small constant to prevent division by zero in weight computation
        n_jobs: int = -1,  # Number of parallel jobs for CV folds (-1 = all cores)
    ):
        self.lambda_ridge_list = lambda_ridge_list
        self.gamma_list = gamma_list
        self.cv = cv
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.use_post_ols_debiasing = use_post_ols_debiasing
        self.auto_tune_collinearity = auto_tune_collinearity
        # Guard: convert string 'null' to Python None (handles YAML parsing edge cases)
        self.weight_clip_max = None if (weight_clip_max is None or weight_clip_max == 'null') else weight_clip_max
        self.eps = eps
        self.n_jobs = n_jobs

        self._init_fitted_attributes()

    def get_params(self, deep: bool = True) -> dict:
        params = {
            'lambda_ridge_list': self.lambda_ridge_list,
            'gamma_list': self.gamma_list,
            'cv': self.cv,
            'alpha_min_ratio': self.alpha_min_ratio,
            'n_alpha': self.n_alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'standardize': self.standardize,
            'fit_intercept': self.fit_intercept,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'use_post_ols_debiasing': self.use_post_ols_debiasing,
            'auto_tune_collinearity': self.auto_tune_collinearity,
            'weight_clip_max': self.weight_clip_max,
            'eps': self.eps,
            'n_jobs': self.n_jobs,
        }
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, cv_splits=None):
        """
        拟合 AdaptiveFlippedLassoCV 模型（严格隔离版）

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
            This ensures all algorithms use the same CV splits for fair comparison.
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error

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

        # Auto-detect collinearity mode if enabled
        if self.auto_tune_collinearity:
            data_mode, energy_ratio = _detect_collinearity_mode(X)
            if self.verbose:
                print(f"[ADL] Detected mode: {data_mode} (energy_ratio={energy_ratio:.3f})")

            if data_mode == "dense_collinear":
                # For Tecator-like data: use higher gamma, lower ridge
                self.gamma_list = (1.0, 2.0, 3.0)
            else:
                # For Riboflavin-like data: use lower gamma, higher ridge
                self.gamma_list = (0.3, 0.5, 1.0)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=_DTYPE)

        # 标准化：在 CV 循环内部进行（每个 fold 独立），避免验证集数据泄露
        # 阶段三最终拟合时会用全量数据重新 fit scaler
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            # 仅用于存储 fitted scaler 的占位，后续在 CV 循环内和阶段三重新处理
            self.scaler_._fitted = False
        else:
            self.scaler_ = None

        n = X.shape[0]
        n_gamma = len(self.gamma_list)
        eps = self.eps

        # ============================================================
        # 阶段一：K 折严格内部寻优
        # ============================================================
        # 如果传入了 cv_splits，使用它；否则自己创建 KFold
        if cv_splits is not None:
            n_folds = len(cv_splits)
            splits = cv_splits
            if self.verbose:
                print(f"[ADL] Using provided {n_folds}-fold CV splits")
        else:
            n_folds = self.cv
            kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            splits = list(kfold.split(X))

        error_matrix = np.full((n_gamma, self.n_alpha, n_folds), np.inf)
        nselected_matrix = np.zeros((n_gamma, self.n_alpha, n_folds))

        # 并行化 fold 处理：每个 fold 独立计算
        def _compute_single_fold_errors(fold_idx, train_idx, val_idx):
            """Compute error matrix for a single fold (used for parallel execution)."""
            X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # 每个 fold 独立标准化
            if self.standardize:
                scaler_fold = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
                X_tr = scaler_fold.fit_transform(X_tr_raw)
                X_va = scaler_fold.transform(X_va_raw)
            else:
                X_tr, X_va = X_tr_raw, X_va_raw

            # 先验提取：仅在训练折上做 RidgeCV（绝对隔离）
            ridge_cv = RidgeCV(alphas=self.lambda_ridge_list, cv=3)
            ridge_cv.fit(X_tr, y_tr)
            beta_ridge_fold = ridge_cv.coef_

            signs_fold = np.sign(beta_ridge_fold)
            signs_fold[signs_fold == 0] = 1.0

            # 存储该 fold 的结果
            fold_errors = np.zeros((n_gamma, self.n_alpha))
            fold_nselected = np.zeros((n_gamma, self.n_alpha))

            # 对每个 gamma 分别计算
            for gamma_idx, gamma in enumerate(self.gamma_list):
                # Min-Anchored Normalization
                raw_weights = 1.0 / (np.abs(beta_ridge_fold) + eps) ** gamma
                min_w = np.min(raw_weights)
                w_normalized = raw_weights / min_w
                clip_max = self.weight_clip_max if self.weight_clip_max is not None else float('inf')
                weights = np.clip(w_normalized, 1.0, clip_max)

                # 空间重构
                X_adaptive_tr = (X_tr * signs_fold) / weights
                X_adaptive_va = (X_va * signs_fold) / weights

                # alpha 搜索路径
                alpha_max = np.max(np.abs(X_adaptive_tr.T @ y_tr)) / len(y_tr)
                alpha_min = alpha_max * self.alpha_min_ratio
                alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

                # lasso_path
                _, coefs_path, _ = lasso_path(
                    X_adaptive_tr, y_tr,
                    alphas=alphas,
                    positive=True,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )

                # 验证集打分
                preds = X_adaptive_va @ coefs_path
                mse_path = np.mean((y_va[:, np.newaxis] - preds) ** 2, axis=0)
                fold_errors[gamma_idx, :] = mse_path

                # 非零系数数量
                fold_nselected[gamma_idx, :] = np.sum(coefs_path != 0, axis=0)

            return fold_idx, fold_errors, fold_nselected

        # 使用 joblib 并行执行所有 fold
        n_jobs = self.n_jobs if hasattr(self, 'n_jobs') else -1
        parallel_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_compute_single_fold_errors)(fold_idx, train_idx, val_idx)
            for fold_idx, (train_idx, val_idx) in enumerate(splits)
        )

        # 收集结果
        for fold_idx, fold_errors, fold_nselected in parallel_results:
            error_matrix[:, :, fold_idx] = fold_errors
            nselected_matrix[:, :, fold_idx] = fold_nselected
            if self.verbose:
                print(f"[Fold {fold_idx + 1}/{n_folds}] completed (parallel)")

        # ============================================================
        # Stage 2: Min-MSE 法则选 gamma（只选物理结构，不选 alpha）
        # ============================================================
        mean_error = np.mean(error_matrix, axis=2)  # (n_gamma, n_alpha)
        mean_nselected = np.mean(nselected_matrix, axis=2)  # (n_gamma, n_alpha)

        # 对每个 gamma，取其全部 alpha 上的最小平均 MSE
        # min_MSE_gamma[gamma_idx] = min_alpha(mean_error[gamma_idx, alpha])
        min_MSE_gamma = np.min(mean_error, axis=1)  # (n_gamma,)
        best_gamma_idx = int(np.argmin(min_MSE_gamma))
        self.best_gamma_ = self.gamma_list[best_gamma_idx]
        best_cv_mse = float(min_MSE_gamma[best_gamma_idx])
        best_cv_nselected = float(mean_nselected[best_gamma_idx].min())
        self.cv_score_ = -best_cv_mse

        if self.verbose:
            print(f"\n[Stage 2] Min-MSE Gamma Selection:")
            for gi, g in enumerate(self.gamma_list):
                marker = " <-- best" if gi == best_gamma_idx else ""
                print(f"  gamma={g:.2f}: min_MSE={float(min_MSE_gamma[gi]):.6f}{marker}")
            print(f"  selected: gamma={self.best_gamma_}, CV_MSE={best_cv_mse:.6f}")

        # ============================================================
        # Stage 3: 全量数据终极拟合 + Fresh LassoCV + 1-SE 选 alpha
        # ============================================================
        if self.standardize:
            X_for_cv = self.scaler_.fit_transform(X)
        else:
            X_for_cv = X

        # Step 1: 全量 RidgeCV → beta_ridge_final（最准的 Ridge 先验）
        ridge_final = RidgeCV(alphas=self.lambda_ridge_list, cv=self.cv)
        ridge_final.fit(X_for_cv, y)
        self.beta_ridge_ = ridge_final.coef_
        self.best_lambda_ridge_ = ridge_final.alpha_

        # Step 2: 固定 best_gamma → weights
        signs_final = np.sign(self.beta_ridge_)
        signs_final[signs_final == 0] = 1.0
        self.signs_ = signs_final

        raw_weights_final = 1.0 / (np.abs(self.beta_ridge_) + eps) ** self.best_gamma_
        min_w = np.min(raw_weights_final)
        self.weights_ = np.clip(raw_weights_final / min_w, 1.0, self.weight_clip_max if self.weight_clip_max is not None else float('inf'))

        # Step 3: 全量扭曲空间
        X_adaptive_final = (X_for_cv * signs_final) / self.weights_

        # Step 4: Fresh LassoCV — 在新空间里重新丈量尺度
        alpha_max = np.max(np.abs(X_adaptive_final.T @ y)) / len(y)
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
        )
        lasso_cv.fit(X_adaptive_final, y)

        # Step 5: 1-SE 法则选 alpha
        # mse_path_: shape (n_alphas, n_folds)
        mse_path = lasso_cv.mse_path_
        mean_mse = np.mean(mse_path, axis=1)
        std_mse = np.std(mse_path, axis=1) / np.sqrt(self.cv)

        min_idx = np.argmin(mean_mse)
        min_mse_value = float(mean_mse[min_idx])
        std_at_min = float(std_mse[min_idx])
        threshold = min_mse_value + std_at_min

        candidates_mask = mean_mse <= threshold
        if np.any(candidates_mask):
            # 在候选中选 alpha 最大（最稀疏）的
            candidate_alphas = alphas[candidates_mask]
            self.best_alpha_ = float(np.max(candidate_alphas))
        else:
            self.best_alpha_ = float(lasso_cv.alpha_)

        if self.verbose:
            print(f"[Stage 3] 1-SE Alpha Selection:")
            print(f"  min_mse={min_mse_value:.6f}, threshold={threshold:.6f}")
            print(f"  selected: alpha={self.best_alpha_:.6f}")

        # Step 6: 最终 Lasso（用 best_alpha）
        lasso_final = Lasso(
            alpha=self.best_alpha_,
            positive=True,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        lasso_final.fit(X_adaptive_final, y)

        coef_final = (lasso_final.coef_ / self.weights_) * signs_final

        # Post-selection OLS debiasing: fit OLS on selected features to reduce MSE
        # selected_mask comes from lasso_final.coef_ (adaptive space), so we must
        # select from X_adaptive_final (not X_for_cv) to maintain index consistency
        intercept_ols = None
        if self.use_post_ols_debiasing:
            selected_mask = lasso_final.coef_ != 0
            n_selected = np.sum(selected_mask)

            if n_selected > 0 and n_selected < X_adaptive_final.shape[1]:
                X_selected = X_adaptive_final[:, selected_mask]
                ols = LinearRegression(fit_intercept=self.fit_intercept)
                ols.fit(X_selected, y)
                # OLS coefficients are in adaptive space → inverse transform to standardized/original space
                coef_debiased = np.zeros(X_adaptive_final.shape[1])
                coef_debiased[selected_mask] = ols.coef_
                coef_debiased = (coef_debiased / self.weights_) * signs_final
                if self.standardize:
                    coef_debiased = coef_debiased / self.scaler_.scale_
                coef_final = coef_debiased
                if self.fit_intercept:
                    intercept_ols = ols.intercept_
            else:
                if self.fit_intercept:
                    intercept_ols = lasso_final.intercept_
        else:
            if self.fit_intercept:
                intercept_ols = lasso_final.intercept_

        # 逆标准化到原始空间
        if self.standardize:
            self.coef_ = coef_final / self.scaler_.scale_
            if self.fit_intercept:
                self.intercept_ = np.mean(y) - np.sum(self.coef_ * self.scaler_.mean_)
        else:
            self.coef_ = coef_final
            if self.fit_intercept:
                self.intercept_ = intercept_ols if intercept_ols is not None else lasso_final.intercept_

        self.is_fitted_ = True
        return self

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """默认负 MSE 评分（与 GridSearchCV 兼容）"""
        from sklearn.metrics import mean_squared_error
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred, sample_weight=sample_weight)


class AdaptiveFlippedLassoEBIC(BaseAdaptiveFlippedLasso, RegressorMixin):
    """
    AdaptiveFlippedLasso with EBIC-based parameter selection.

    EBIC (Extended BIC) formula:
        EBIC = n * ln(RSS/n) + |S| * ln(n) + 2 * gamma * ln(C(p, |S|))

    EBIC 厌恶假阳性，会自然地把 lambda_ridge 压在较小值，保证权重有效性。
    """

    def __init__(
        self,
        lambda_ridge_list: tuple = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        gamma_list: tuple = (0.3, 0.5, 0.7, 1.0),
        ebic_gamma: float = 0.5,
        alpha_min_ratio: float = 1e-4,
        n_alpha: int = 100,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = False,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
    ):
        self.lambda_ridge_list = lambda_ridge_list
        self.gamma_list = gamma_list
        self.ebic_gamma = ebic_gamma
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose

        self._init_fitted_attributes()

    def get_params(self, deep: bool = True) -> dict:
        params = {
            'lambda_ridge_list': self.lambda_ridge_list,
            'gamma_list': self.gamma_list,
            'ebic_gamma': self.ebic_gamma,
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

    def _compute_ebic(self, y_true: np.ndarray, y_pred: np.ndarray, n_selected: int, p: int) -> float:
        """
        计算 EBIC 值。

        EBIC = n * ln(RSS/n) + |S| * ln(n) + 2 * gamma * ln(C(p, |S|))

        其中 C(p, |S|) = p! / (|S|! * (p - |S|)!)
        """
        n = len(y_true)
        rss = np.sum((y_true - y_pred) ** 2)

        # 第一项：拟合误差
        term1 = n * np.log(rss / n + 1e-10)

        # 第二项：模型复杂度（特征数量）
        term2 = n_selected * np.log(n + 1e-10)

        # 第三项：高维惩罚
        # ln(C(p, |S|)) = ln(p!) - ln(|S|!) - ln((p-|S|)!)
        # 使用 Stirling 近似避免阶乘溢出
        if n_selected == 0:
            term3 = 0
        elif n_selected == p:
            term3 = 2 * self.ebic_gamma * np.log(1)  # C(p,p) = 1
        else:
            # log(C(p,k)) ≈ k * log(p/k) for large p, small k
            # 更精确：使用 scipy.special.gammaln
            from scipy.special import gammaln
            log_comb = gammaln(p + 1) - gammaln(n_selected + 1) - gammaln(p - n_selected + 1)
            term3 = 2 * self.ebic_gamma * log_comb

        ebic = term1 + term2 + term3
        return ebic

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        使用 EBIC 准则拟合 AdaptiveFlippedLasso。
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error
        from scipy.special import gammaln

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
        n = X.shape[0]
        p = self.n_features_in_

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=_DTYPE)

        # 标准化
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None

        eps = 1e-5

        if self.verbose:
            print(f"[EBIC] n={n}, p={p}, searching {len(self.gamma_list)} gammas x {self.n_alpha} alphas")

        # ============================================================
        # 阶段一：对每个 (gamma, lambda_ridge, alpha) 计算 EBIC
        # ============================================================
        best_ebic = np.inf
        best_gamma = None
        best_lambda_ridge = None
        best_alpha = None
        best_coefs = None

        for gamma in self.gamma_list:
            for lambda_ridge in self.lambda_ridge_list:
                # Stage 1: Ridge 回归
                ridge = Ridge(alpha=lambda_ridge, fit_intercept=True, random_state=self.random_state)
                ridge.fit(X, y)
                beta_ridge = ridge.coef_

                signs = np.sign(beta_ridge)
                signs[signs == 0] = 1.0

                # Min-Anchored Normalization
                raw_weights = 1.0 / (np.abs(beta_ridge) + eps) ** gamma
                min_w = np.min(raw_weights)
                clip_max = self.weight_clip_max if self.weight_clip_max is not None else float('inf')
                weights = np.clip(raw_weights / min_w, 1.0, clip_max)

                # 空间重构
                X_adaptive = (X * signs) / weights

                # alpha 搜索路径
                alpha_max = np.max(np.abs(X_adaptive.T @ y)) / n
                alpha_min = alpha_max * self.alpha_min_ratio
                alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

                for alpha in alphas:
                    # Stage 2: Lasso 拟合
                    lasso = Lasso(
                        alpha=alpha,
                        positive=True,
                        fit_intercept=self.fit_intercept,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        random_state=self.random_state,
                    )
                    lasso.fit(X_adaptive, y)

                    # 还原系数到原始空间
                    coef_adaptive = lasso.coef_
                    n_selected = np.sum(coef_adaptive > 0)

                    if n_selected == 0:
                        # 全零解，跳过
                        continue

                    # 逆变换
                    coef_final = (coef_adaptive / weights) * signs

                    # 预测并计算 RSS
                    if self.fit_intercept:
                        intercept = np.mean(y) - np.mean(X @ coef_final)
                        y_pred = X @ coef_final + intercept
                    else:
                        y_pred = X @ coef_final

                    # 计算 EBIC
                    ebic = self._compute_ebic(y, y_pred, n_selected, p)

                    if ebic < best_ebic:
                        best_ebic = ebic
                        best_gamma = gamma
                        best_lambda_ridge = lambda_ridge
                        best_alpha = alpha
                        best_coefs = coef_final.copy()
                        best_intercept = lasso.intercept_ if self.fit_intercept else 0

                        if self.verbose:
                            print(f"[EBIC] New best: gamma={gamma:.2f}, lambda_ridge={lambda_ridge:.2f}, "
                                  f"alpha={alpha:.6f}, n_selected={n_selected}, EBIC={ebic:.4f}")

        if best_gamma is None:
            raise RuntimeError("No valid model found (all n_selected=0)")

        self.best_gamma_ = best_gamma
        self.best_lambda_ridge_ = best_lambda_ridge
        self.best_alpha_ = best_alpha
        self.best_ebic_ = best_ebic
        self.ebic_score_ = -best_ebic

        if self.verbose:
            print(f"\n[EBIC] Selected: gamma={best_gamma}, lambda_ridge={best_lambda_ridge}, "
                  f"alpha={best_alpha:.6f}, n_selected={np.sum(best_coefs > 0)}")

        # ============================================================
        # 阶段二：用最优参数在全量数据上最终拟合
        # ============================================================
        # Stage 1
        ridge_final = Ridge(alpha=self.best_lambda_ridge_, fit_intercept=True, random_state=self.random_state)
        ridge_final.fit(X, y)
        self.beta_ridge_ = ridge_final.coef_

        signs_final = np.sign(self.beta_ridge_)
        signs_final[signs_final == 0] = 1.0
        self.signs_ = signs_final

        raw_weights_final = 1.0 / (np.abs(self.beta_ridge_) + eps) ** self.best_gamma_
        min_w = np.min(raw_weights_final)
        self.weights_ = np.clip(raw_weights_final / min_w, 1.0, self.weight_clip_max if self.weight_clip_max is not None else float('inf'))

        X_adaptive_final = (X * signs_final) / self.weights_

        # Stage 2
        lasso_final = Lasso(
            alpha=self.best_alpha_,
            positive=True,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        lasso_final.fit(X_adaptive_final, y)

        coef_final = (lasso_final.coef_ / self.weights_) * signs_final

        # 逆标准化
        if self.standardize:
            self.coef_ = coef_final / self.scaler_.scale_
            if self.fit_intercept:
                self.intercept_ = np.mean(y) - np.sum(self.coef_ * self.scaler_.mean_)
        else:
            self.coef_ = coef_final
            if self.fit_intercept:
                self.intercept_ = lasso_final.intercept_

        self.is_fitted_ = True
        return self

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """默认负 EBIC 评分"""
        return self.ebic_score_


class AdaptiveFlippedLassoCV_UnifiedPrior(BaseAdaptiveFlippedLasso, RegressorMixin):
    """
    AdaptiveFlippedLassoCV with Unified Prior (统一先验版)

    与标准 AdaptiveFlippedLassoCV 的区别：
    - Stage 1 中，beta_ridge 和 weights 基于全部训练数据提取一次
    - 各 fold 只承担验证职责，使用统一的 weights 和 alpha 路径
    - Stage 2 中所有 fold 共享同一个 alpha 索引，平均 MSE 有物理意义

    算法流程：
    阶段一：统一先验 + K 折验证
        - RidgeCV(全部训练数据) → beta_ridge_all (统一先验)
        - 对每个 gamma：计算 weights_all + 统一 alpha 路径
        - for each fold k:
            用统一 weights 变换 fold 数据
            lasso_path(训练折) → 全部 alpha 的系数路径
            在验证折上打分 → Error_Matrix[gamma, alpha, k]
    阶段二：选拔最优参数（平均 MSE 最小的 gamma, alpha）
    阶段三：全量数据终极拟合（带回最佳参数）
    """

    def __init__(
        self,
        lambda_ridge_list: tuple = (0.1, 1.0, 10.0, 100.0),
        gamma_list: tuple = (0.5, 1.0, 2.0),
        cv: int = 5,
        alpha_min_ratio: float = 1e-4,
        n_alpha: int = 100,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = False,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
        use_post_ols_debiasing: bool = False,
        auto_tune_collinearity: bool = True,
        weight_clip_max: float = 100.0,
    ):
        self.lambda_ridge_list = lambda_ridge_list
        self.gamma_list = gamma_list
        self.cv = cv
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.use_post_ols_debiasing = use_post_ols_debiasing
        self.auto_tune_collinearity = auto_tune_collinearity
        # Guard: convert string 'null' to Python None (handles YAML parsing edge cases)
        self.weight_clip_max = None if (weight_clip_max is None or weight_clip_max == 'null') else weight_clip_max

        self._init_fitted_attributes()

    def get_params(self, deep: bool = True) -> dict:
        params = {
            'lambda_ridge_list': self.lambda_ridge_list,
            'gamma_list': self.gamma_list,
            'cv': self.cv,
            'alpha_min_ratio': self.alpha_min_ratio,
            'n_alpha': self.n_alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'standardize': self.standardize,
            'fit_intercept': self.fit_intercept,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'use_post_ols_debiasing': self.use_post_ols_debiasing,
            'auto_tune_collinearity': self.auto_tune_collinearity,
            'weight_clip_max': self.weight_clip_max,
        }
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, cv_splits=None):
        """
        拟合 AdaptiveFlippedLassoCV_UnifiedPrior 模型

        核心改进：先验提取基于全部训练数据，各 fold 只做验证
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error

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

        # Auto-detect collinearity mode if enabled
        if self.auto_tune_collinearity:
            data_mode, energy_ratio = _detect_collinearity_mode(X)
            if self.verbose:
                print(f"[ADL-Unified] Detected mode: {data_mode} (energy_ratio={energy_ratio:.3f})")

            if data_mode == "dense_collinear":
                self.gamma_list = (1.0, 2.0, 3.0)
            else:
                self.gamma_list = (0.3, 0.5, 1.0)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=_DTYPE)

        # 标准化
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            self.scaler_._fitted = False
        else:
            self.scaler_ = None

        n = X.shape[0]
        n_gamma = len(self.gamma_list)
        eps = 1e-5

        # ============================================================
        # 阶段一（修改版）：统一先验提取 + K 折验证
        # ============================================================
        if cv_splits is not None:
            n_folds = len(cv_splits)
            splits = cv_splits
            if self.verbose:
                print(f"[ADL-Unified] Using provided {n_folds}-fold CV splits")
        else:
            n_folds = self.cv
            kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            splits = list(kfold.split(X))

        # ---- 统一先验提取：在 CV 循环之前，基于全部训练数据 ----
        if self.standardize:
            X_std = self.scaler_.fit_transform(X)
        else:
            X_std = X

        ridge_cv = RidgeCV(alphas=self.lambda_ridge_list, cv=self.cv)
        ridge_cv.fit(X_std, y)
        beta_ridge_all = ridge_cv.coef_
        best_lambda_ridge = ridge_cv.alpha_

        signs_all = np.sign(beta_ridge_all)
        signs_all[signs_all == 0] = 1.0

        if self.verbose:
            print(f"[ADL-Unified] Prior from ALL data: lambda_ridge={best_lambda_ridge:.4f}")
            print(f"[ADL-Unified] beta_ridge range: [{np.min(beta_ridge_all):.4f}, {np.max(beta_ridge_all):.4f}]")

        # 对每个 gamma 预计算：统一权重 + 统一 alpha 路径
        weights_per_gamma = {}
        for gamma_idx, gamma in enumerate(self.gamma_list):
            raw_weights = 1.0 / (np.abs(beta_ridge_all) + eps) ** gamma
            min_w = np.min(raw_weights)
            w_normalized = raw_weights / min_w
            clip_max = self.weight_clip_max if self.weight_clip_max is not None else float('inf')
            weights = np.clip(w_normalized, 1.0, clip_max)

            # 用全部训练数据计算 alpha_max，统一 alpha 路径
            X_adaptive_all = (X_std * signs_all) / weights
            alpha_max = np.max(np.abs(X_adaptive_all.T @ y)) / len(y)
            alpha_min = alpha_max * self.alpha_min_ratio
            alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

            weights_per_gamma[gamma_idx] = (weights, alphas)

            if self.verbose:
                print(f"[ADL-Unified] gamma={gamma}: weight range [{np.min(weights):.4f}, {np.max(weights):.4f}], "
                      f"alpha range [{alphas[-1]:.6f}, {alphas[0]:.4f}]")

        # ---- K 折验证：使用统一先验在各 fold 上验证 ----
        error_matrix = np.full((n_gamma, self.n_alpha, n_folds), np.inf)
        nselected_matrix = np.zeros((n_gamma, self.n_alpha, n_folds))

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # 每个 fold 独立标准化
            if self.standardize:
                scaler_fold = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
                X_tr = scaler_fold.fit_transform(X_tr_raw)
                X_va = scaler_fold.transform(X_va_raw)
            else:
                X_tr, X_va = X_tr_raw, X_va_raw

            if self.verbose:
                print(f"[Fold {fold_idx + 1}/{n_folds}] train={len(train_idx)}, val={len(val_idx)}")

            # 对每个 gamma：用统一先验在 fold 上验证
            for gamma_idx, gamma in enumerate(self.gamma_list):
                weights, alphas = weights_per_gamma[gamma_idx]

                # 空间重构：用统一的 signs 和 weights 变换 fold 数据
                X_adaptive_tr = (X_tr * signs_all) / weights
                X_adaptive_va = (X_va * signs_all) / weights

                # lasso_path：基于统一 alpha 路径
                _, coefs_path, _ = lasso_path(
                    X_adaptive_tr, y_tr,
                    alphas=alphas,
                    positive=True,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )

                # 在验证集上打分
                preds = X_adaptive_va @ coefs_path  # (n_val, n_alphas)
                mse_path = np.mean((y_va[:, np.newaxis] - preds) ** 2, axis=0)
                error_matrix[gamma_idx, :, fold_idx] = mse_path

                # 追踪非零系数数量
                nselected_path = np.sum(coefs_path != 0, axis=0)
                nselected_matrix[gamma_idx, :, fold_idx] = nselected_path

        # ============================================================
        # 阶段二：选拔最优参数（1-SE 法则）
        # ============================================================
        mean_error = np.mean(error_matrix, axis=2)
        std_error = np.std(error_matrix, axis=2) / np.sqrt(n_folds)
        mean_nselected = np.mean(nselected_matrix, axis=2)

        min_mse = np.min(mean_error)
        min_mse_idx = np.unravel_index(np.argmin(mean_error), mean_error.shape)
        min_std = std_error[min_mse_idx]
        threshold = min_mse + min_std

        candidates_mask = mean_error <= threshold

        if not np.any(candidates_mask):
            if self.verbose:
                print("[Stage 2] Warning: No candidates within 1-SE, using standard min-MSE")
            best_gamma_idx, best_alpha_idx = np.unravel_index(
                np.argmin(mean_error), mean_error.shape
            )
        else:
            masked_nselected = np.where(candidates_mask, mean_nselected, np.inf)
            best_flat_idx = np.argmin(masked_nselected)
            best_gamma_idx, best_alpha_idx = np.unravel_index(best_flat_idx, mean_error.shape)

        # High-dimensional fallback
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if n_features > n_samples * 2:
            if self.verbose:
                print("[Stage 2] High-dimensional data detected, using min MSE instead of 1-SE")
            best_gamma_idx, best_alpha_idx = np.unravel_index(
                np.argmin(mean_error), mean_error.shape
            )

        self.best_gamma_ = self.gamma_list[best_gamma_idx]
        best_gamma_weights, best_gamma_alphas = weights_per_gamma[best_gamma_idx]
        self.best_alpha_ = best_gamma_alphas[best_alpha_idx]
        best_cv_mse = mean_error[best_gamma_idx, best_alpha_idx]
        best_cv_nselected = mean_nselected[best_gamma_idx, best_alpha_idx]
        self.cv_score_ = -best_cv_mse

        if self.verbose:
            print(f"\n[Stage 2] 1-SE Rule:")
            print(f"  min_MSE={min_mse:.6f}, threshold={threshold:.6f}")
            print(f"  n_candidates={np.sum(candidates_mask)}")
            print(f"  selected: gamma={self.best_gamma_}, alpha={self.best_alpha_:.6f}")
            print(f"  selected: CV_MSE={best_cv_mse:.6f}, n_selected≈{best_cv_nselected:.1f}")

        # ============================================================
        # 阶段三：全量数据终极拟合
        # ============================================================
        self.beta_ridge_ = beta_ridge_all
        self.best_lambda_ridge_ = best_lambda_ridge
        self.signs_ = signs_all
        self.weights_ = best_gamma_weights

        X_adaptive_final = (X_std * signs_all) / self.weights_

        lasso_final = Lasso(
            alpha=self.best_alpha_,
            positive=True,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        lasso_final.fit(X_adaptive_final, y)

        coef_final = (lasso_final.coef_ / self.weights_) * signs_all

        # 逆标准化到原始空间
        if self.standardize:
            self.coef_ = coef_final / self.scaler_.scale_
            if self.fit_intercept:
                self.intercept_ = np.mean(y) - np.sum(self.coef_ * self.scaler_.mean_)
        else:
            self.coef_ = coef_final
            if self.fit_intercept:
                self.intercept_ = lasso_final.intercept_

        self.is_fitted_ = True
        return self

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """默认负 MSE 评分"""
        from sklearn.metrics import mean_squared_error
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred, sample_weight=sample_weight)


class AdaptiveFlippedLassoEBIC_Simple(BaseAdaptiveFlippedLasso, RegressorMixin):
    """
    AdaptiveFlippedLasso with EBIC - Simplified Data Flow

    完全按照用户指定的数据流实现：

    1. 收到 X_train → 绝对不切分！100% 全量数据
    2. 在全量 X_train 上提取 Ridge 先验
    3. 遍历网格 (gamma, tau_clip, alpha)
    4. 直接在全量 X_train 上计算 RSS 和 |S|
    5. 代入 EBIC 公式计算得分
    6. 选择最低 EBIC 对应的系数，直接输出（无需重新拟合）

    EBIC formula:
        EBIC = n * ln(RSS/n) + |S| * ln(n) + 2 * gamma_ebic * ln(C(p, |S|))

    特点：
    - 无 CV，无数据切分
    - weight_clip_max 作为网格搜索参数（tau_clip）
    - 计算量小，速度快
    """

    def __init__(
        self,
        lambda_ridge: float = 1.0,
        gamma_list: tuple = (0.3, 0.5, 0.7, 1.0),
        tau_clip_list: tuple = (1.0, 10.0, 100.0, None),  # weight_clip_max grid, None=unlimited
        alpha_min_ratio: float = 1e-4,
        n_alpha: int = 50,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = True,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
        ebic_gamma: float = 0.5,
        eps: float = 1e-5,
        n_jobs: int = -1,  # Number of parallel jobs for Lasso grid search
    ):
        self.lambda_ridge = lambda_ridge
        self.gamma_list = gamma_list
        # Guard: convert 'null' strings to Python None in tau_clip_list
        self.tau_clip_list = tuple(None if (x is None or x == 'null') else x for x in tau_clip_list)
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.ebic_gamma = ebic_gamma
        self.eps = eps
        self.n_jobs = n_jobs

        self._init_fitted_attributes()

    def get_params(self, deep: bool = True) -> dict:
        return {
            'lambda_ridge': self.lambda_ridge,
            'gamma_list': self.gamma_list,
            'tau_clip_list': self.tau_clip_list,
            'alpha_min_ratio': self.alpha_min_ratio,
            'n_alpha': self.n_alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'standardize': self.standardize,
            'fit_intercept': self.fit_intercept,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'ebic_gamma': self.ebic_gamma,
            'eps': self.eps,
            'n_jobs': self.n_jobs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    def _compute_ebic(self, n: int, rss: float, n_selected: int, p: int) -> float:
        """计算 EBIC 值"""
        term1 = n * np.log(rss / n + 1e-10)
        term2 = n_selected * np.log(n + 1e-10)
        if n_selected == 0:
            term3 = 0
        elif n_selected == p:
            term3 = 2 * self.ebic_gamma * np.log(1)
        else:
            from scipy.special import gammaln
            log_comb = gammaln(p + 1) - gammaln(n_selected + 1) - gammaln(p - n_selected + 1)
            term3 = 2 * self.ebic_gamma * log_comb
        return term1 + term2 + term3

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """拟合模型 - 完全按照用户指定的数据流"""
        from sklearn.linear_model import Ridge

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
        n = X.shape[0]
        p = self.n_features_in_

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=_DTYPE)

        # 标准化
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None

        eps = self.eps

        if self.verbose:
            print(f"[EBIC-Simple] n={n}, p={p}")
            print(f"[EBIC-Simple] Grid: {len(self.gamma_list)} gammas × {len(self.tau_clip_list)} tau_clips × {self.n_alpha} alphas")

        # ============================================================
        # Stage 1: 全量 X 上 Ridge 先验（只做一次）
        # ============================================================
        ridge = Ridge(alpha=self.lambda_ridge, fit_intercept=True, random_state=self.random_state)
        ridge.fit(X, y)
        beta_ridge = ridge.coef_

        signs = np.sign(beta_ridge)
        signs[signs == 0] = 1.0

        if self.verbose:
            print(f"[EBIC-Simple] Ridge prior: lambda_ridge={self.lambda_ridge:.4f}, nonzero={np.sum(beta_ridge != 0)}/{p}")

        # ============================================================
        # Stage 2: 网格搜索 (gamma, tau_clip, alpha) - 并行化
        # ============================================================
        def _evaluate_single(gamma, tau_clip, alpha):
            """评估单个 (gamma, tau_clip, alpha) 组合"""
            # 预计算权重
            raw_weights = 1.0 / (np.abs(beta_ridge) + eps) ** gamma
            min_w = np.min(raw_weights)
            w_normalized = raw_weights / min_w
            # Guard against string 'null' being passed (should be None after YAML parsing)
            _tau_clip = None if (tau_clip is None or tau_clip == 'null') else tau_clip
            weights = w_normalized.copy() if _tau_clip is None else np.clip(w_normalized, 1.0, _tau_clip)

            X_adaptive = (X * signs) / weights

            lasso = Lasso(
                alpha=alpha,
                positive=True,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            lasso.fit(X_adaptive, y)

            coef_adaptive = lasso.coef_
            n_selected = np.sum(coef_adaptive > 0)

            if n_selected == 0:
                return None  # 全零解，跳过

            coef_final = (coef_adaptive / weights) * signs

            if self.fit_intercept:
                intercept = np.mean(y) - np.mean(X @ coef_final)
                y_pred = X @ coef_final + intercept
            else:
                intercept = 0.0
                y_pred = X @ coef_final

            rss = np.sum((y - y_pred) ** 2)
            ebic = self._compute_ebic(n, rss, n_selected, p)

            return {
                'gamma': gamma,
                'tau_clip': tau_clip,
                'alpha': alpha,
                'ebic': ebic,
                'coef': coef_final.copy(),
                'intercept': intercept,
                'n_selected': n_selected,
            }

        # 构建所有组合列表
        tasks = []
        for gamma in self.gamma_list:
            for tau_clip in self.tau_clip_list:
                # 预计算 alpha 路径
                raw_weights = 1.0 / (np.abs(beta_ridge) + eps) ** gamma
                min_w = np.min(raw_weights)
                w_normalized = raw_weights / min_w
                # Guard against string 'null' being passed (should be None after YAML parsing)
                _tau_clip = None if (tau_clip is None or tau_clip == 'null') else tau_clip
                weights = w_normalized.copy() if _tau_clip is None else np.clip(w_normalized, 1.0, _tau_clip)
                X_adaptive = (X * signs) / weights

                alpha_max = np.max(np.abs(X_adaptive.T @ y)) / n
                alpha_min = alpha_max * self.alpha_min_ratio
                alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

                for alpha in alphas:
                    tasks.append((gamma, _tau_clip, alpha))

        total = len(tasks)

        # 并行执行
        n_jobs = self.n_jobs if hasattr(self, 'n_jobs') and self.n_jobs is not None else -1
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_evaluate_single)(gamma, tau_clip, alpha)
            for gamma, tau_clip, alpha in tasks
        )

        # 找最优
        best_ebic = np.inf
        best_gamma = None
        best_tau_clip = None
        best_alpha = None
        best_coefs = None
        best_intercept = 0.0
        best_n_selected = 0

        for i, result in enumerate(results):
            if result is not None and result['ebic'] < best_ebic:
                best_ebic = result['ebic']
                best_gamma = result['gamma']
                best_tau_clip = result['tau_clip']
                best_alpha = result['alpha']
                best_coefs = result['coef']
                best_intercept = result['intercept']
                best_n_selected = result['n_selected']

                if self.verbose:
                    print(f"[EBIC-Simple] New best ({i+1}/{total}): gamma={best_gamma}, tau={best_tau_clip}, "
                          f"alpha={best_alpha:.6f}, |S|={best_n_selected}, EBIC={best_ebic:.4f}")

        if best_coefs is None:
            raise RuntimeError("No valid model found (all n_selected=0)")

        # ============================================================
        # Stage 3: 直接输出最优系数
        # ============================================================
        self.best_gamma_ = best_gamma
        self.best_tau_clip_ = best_tau_clip
        self.best_alpha_ = best_alpha
        self.best_ebic_ = best_ebic
        self.ebic_score_ = -best_ebic
        self.beta_ridge_ = beta_ridge
        self.signs_ = signs

        # 记录最优权重
        raw_w = 1.0 / (np.abs(beta_ridge) + eps) ** best_gamma
        min_w = np.min(raw_w)
        w_norm = raw_w / min_w
        self.weights_ = w_norm.copy() if best_tau_clip is None else np.clip(w_norm, 1.0, best_tau_clip)

        # 逆标准化
        if self.standardize:
            self.coef_ = best_coefs / self.scaler_.scale_
            if self.fit_intercept:
                self.intercept_ = np.mean(y) - np.sum(self.coef_ * self.scaler_.mean_)
        else:
            self.coef_ = best_coefs
            if self.fit_intercept:
                self.intercept_ = best_intercept

        if self.verbose:
            print(f"\n[EBIC-Simple] Selected: gamma={best_gamma}, tau_clip={best_tau_clip}, alpha={best_alpha:.6f}")
            print(f"[EBIC-Simple] |S|={np.sum(self.coef_ != 0)}, EBIC={best_ebic:.4f}")

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """默认负 EBIC 评分"""
        return self.ebic_score_


class AdaptiveFlippedLassoCV_EN(BaseAdaptiveFlippedLasso, RegressorMixin):
    """
    AFL-CV-EN — Per-fold ENCV with Strict Data Isolation
    ====================================================

    Stage 1: 严格隔离的折内联合寻优 (Pure CV)
        K 折划分。每折内部独立：
        - StandardScaler(仅训练折)
        - ENCV(cv=3, 仅训练折) → beta_en_fold
        - 遍历 gamma 生成折内 weights
        - lasso_path + 验证折打分
        → 输出诚实 3D 误差矩阵 error_matrix[gamma, alpha, fold]

    Stage 2: 全局唯一大选 (Single 1-SE Selection)
        - error_matrix 在 fold 上求平均 → mean_error[gamma, alpha]
        - 一次性 1-SE 法则同时锁定 (best_gamma, best_alpha)
        - 绝对不分两步！

    Stage 3: 全量数据终极拟合 (Final Re-fit)
        - 全量数据 StandardScaler + ENCV(cv=5) → final_beta_ridge
        - 严格使用 Stage 2 选定的 best_gamma + best_alpha
        - 全量数据最后一次 Lasso 拟合 → 逆变换 → coef_
    """

    def __init__(
        self,
        lambda_ridge_list: tuple = (0.1, 1.0, 10.0, 100.0),
        l1_ratio_list: tuple = (0.1, 0.5, 0.9),  # Elastic Net l1_ratio search
        gamma_list: tuple = (0.5, 1.0, 2.0),
        cv: int = 5,
        alpha_min_ratio: float = 1e-4,
        n_alpha: int = 100,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = False,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
        use_post_ols_debiasing: bool = False,
        auto_tune_collinearity: bool = True,
        weight_clip_max: float = None,  # None = no limit (unbounded weights)
        eps: float = 1e-5,
        n_jobs: int = -1,
    ):
        self.lambda_ridge_list = lambda_ridge_list
        self.l1_ratio_list = l1_ratio_list
        self.gamma_list = gamma_list
        self.cv = cv
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.use_post_ols_debiasing = use_post_ols_debiasing
        self.auto_tune_collinearity = auto_tune_collinearity
        # Guard: convert string 'null' to Python None (handles YAML parsing edge cases)
        self.weight_clip_max = None if (weight_clip_max is None or weight_clip_max == 'null') else weight_clip_max
        self.eps = eps
        self.n_jobs = n_jobs

        self._init_fitted_attributes()

    def get_params(self, deep: bool = True) -> dict:
        params = {
            'lambda_ridge_list': self.lambda_ridge_list,
            'l1_ratio_list': self.l1_ratio_list,
            'gamma_list': self.gamma_list,
            'cv': self.cv,
            'alpha_min_ratio': self.alpha_min_ratio,
            'n_alpha': self.n_alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'standardize': self.standardize,
            'fit_intercept': self.fit_intercept,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'use_post_ols_debiasing': self.use_post_ols_debiasing,
            'auto_tune_collinearity': self.auto_tune_collinearity,
            'weight_clip_max': self.weight_clip_max,
            'eps': self.eps,
            'n_jobs': self.n_jobs,
        }
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    def _compute_en_prior(self, X_tr, y_tr, train_idx=None):
        """
        使用 Elastic Net 计算先验系数
        搜索最优 (lambda_ridge, l1_ratio) 组合
        使用内部 3-fold CV 评估（并行化）

        Returns:
            beta_en: Elastic Net 系数 (p,)
            best_lambda_ridge: 最佳的 lambda_ridge 值
            best_l1_ratio: 最佳的 l1_ratio 值
        """
        from sklearn.model_selection import KFold

        # Ensure numeric types (YAML may parse as strings)
        l1_ratio_list = [float(x) for x in self.l1_ratio_list]
        lambda_ridge_list = [float(x) for x in self.lambda_ridge_list]

        # 构建所有 (l1_ratio, lambda_ridge) 组合列表
        combinations = [(l1, lr) for l1 in l1_ratio_list for lr in lambda_ridge_list]

        # 内部 3-fold CV
        internal_cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        splits = list(internal_cv.split(X_tr))

        def _evaluate_single_combination(l1_ratio, lambda_ridge):
            """评估单个 (l1_ratio, lambda_ridge) 组合"""
            scores = []
            for internal_train, internal_val in splits:
                X_it, X_iv = X_tr[internal_train], X_tr[internal_val]
                y_it, y_iv = y_tr[internal_train], y_tr[internal_val]

                en = ElasticNet(
                    alpha=lambda_ridge,
                    l1_ratio=l1_ratio,
                    fit_intercept=True,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                )
                en.fit(X_it, y_it)
                y_pred = en.predict(X_iv)
                mse = np.mean((y_iv - y_pred) ** 2)
                scores.append(-mse)  # 负 MSE（越大越好）

            mean_score = np.mean(scores)

            # 用全部训练数据重新拟合得到最终系数
            en_final = ElasticNet(
                alpha=lambda_ridge,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            en_final.fit(X_tr, y_tr)

            return mean_score, en_final.coef_.copy(), lambda_ridge, l1_ratio

        # 并行评估所有组合
        n_en_jobs = self.n_en_jobs if hasattr(self, 'n_en_jobs') else -1
        results = Parallel(n_jobs=n_en_jobs, verbose=0)(
            delayed(_evaluate_single_combination)(l1, lr)
            for l1, lr in combinations
        )

        # 找最优组合
        best_score = -np.inf
        best_beta = None
        best_lambda = float(lambda_ridge_list[0])
        best_l1 = float(l1_ratio_list[0])

        for mean_score, beta, lambda_ridge, l1_ratio in results:
            if mean_score > best_score:
                best_score = mean_score
                best_beta = beta.copy()
                best_lambda = float(lambda_ridge)
                best_l1 = float(l1_ratio)

        if best_beta is None:
            # Fallback: 使用默认参数
            en_fallback = ElasticNet(
                alpha=lambda_ridge_list[0],
                l1_ratio=l1_ratio_list[0],
                fit_intercept=True,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            en_fallback.fit(X_tr, y_tr)
            best_beta = en_fallback.coef_.copy()
            best_lambda = float(lambda_ridge_list[0])
            best_l1 = float(l1_ratio_list[0])

        return best_beta, best_lambda, best_l1

    def _select_en_params(self, X_tr, y_tr):
        """
        使用内部 3-fold CV 选择最优 EN 超参数 (lambda_ridge, l1_ratio)

        Returns:
            best_lambda_ridge: 最优 lambda_ridge
            best_l1_ratio: 最优 l1_ratio
        """
        from sklearn.model_selection import KFold

        # Ensure numeric types (YAML may parse as strings)
        l1_ratio_list = [float(x) for x in self.l1_ratio_list]
        lambda_ridge_list = [float(x) for x in self.lambda_ridge_list]

        # 构建所有 (l1_ratio, lambda_ridge) 组合列表
        combinations = [(l1, lr) for l1 in l1_ratio_list for lr in lambda_ridge_list]

        # 内部 3-fold CV
        internal_cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        splits = list(internal_cv.split(X_tr))

        def _evaluate_single_combination(l1_ratio, lambda_ridge):
            """评估单个 (l1_ratio, lambda_ridge) 组合"""
            scores = []
            for internal_train, internal_val in splits:
                X_it, X_iv = X_tr[internal_train], X_tr[internal_val]
                y_it, y_iv = y_tr[internal_train], y_tr[internal_val]

                en = ElasticNet(
                    alpha=lambda_ridge,
                    l1_ratio=l1_ratio,
                    fit_intercept=True,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                )
                en.fit(X_it, y_it)
                y_pred = en.predict(X_iv)
                mse = np.mean((y_iv - y_pred) ** 2)
                scores.append(-mse)  # 负 MSE（越大越好）

            return np.mean(scores)

        # 并行评估所有组合
        n_en_jobs = self.n_en_jobs if hasattr(self, 'n_en_jobs') else -1
        results = Parallel(n_jobs=n_en_jobs, verbose=0)(
            delayed(_evaluate_single_combination)(l1, lr)
            for l1, lr in combinations
        )

        # 找最优组合
        best_score = -np.inf
        best_lambda = float(lambda_ridge_list[0])
        best_l1 = float(l1_ratio_list[0])

        for (l1, lr), mean_score in zip(combinations, results):
            if mean_score > best_score:
                best_score = mean_score
                best_lambda = float(lr)
                best_l1 = float(l1)

        return best_lambda, best_l1

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, cv_splits=None):
        """
        拟合 AdaptiveFlippedLassoCV_EN 模型（Per-fold ENCV + 严格数据隔离）

        Stage 1: 折内 ENCV 评估 (gamma, alpha) 网格
        Stage 2: 全局 1-SE 选拔 (best_gamma, best_alpha)
        Stage 3: 全量数据终极拟合
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

        eps = self.eps
        gamma_list = [float(x) for x in self.gamma_list]
        n_gamma = len(gamma_list)
        n_jobs = self.n_jobs if self.n_jobs is not None else -1

        # ============================================================
        # Stage 1: 严格隔离的折内 ENCV 寻优
        # ============================================================
        if cv_splits is not None:
            n_folds = len(cv_splits)
            splits = cv_splits
        else:
            n_folds = self.cv
            kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            splits = list(kfold.split(X))

        error_matrix = np.full((n_gamma, self.n_alpha, n_folds), np.inf)
        nselected_matrix = np.zeros((n_gamma, self.n_alpha, n_folds))

        def _compute_single_fold_errors(fold_idx, train_idx, val_idx):
            """
            每折独立计算：
            1. 折内 StandardScaler
            2. 折内 ENCV → beta_en_fold（仅用训练折）
            3. 对每个 gamma 生成折内 weights
            4. lasso_path + 验证折打分
            """
            X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # 折内标准化（仅用训练折 fit）
            if self.standardize:
                scaler_fold = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
                X_tr = scaler_fold.fit_transform(X_tr_raw)
                X_va = scaler_fold.transform(X_va_raw)
            else:
                X_tr, X_va = X_tr_raw.copy(), X_va_raw.copy()

            # 折内 ENCV（仅用训练折）→ 折内先验
            en_cv = ElasticNetCV(
                l1_ratio=list(self.l1_ratio_list),
                alphas=list(self.lambda_ridge_list),
                cv=3,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                fit_intercept=True,
            )
            en_cv.fit(X_tr, y_tr)
            beta_en_fold = en_cv.coef_

            if self.verbose:
                print(f"  [Fold {fold_idx + 1}] ENCV: lambda={en_cv.alpha_:.4f}, "
                      f"l1_ratio={en_cv.l1_ratio_:.2f}, nonzero={np.sum(beta_en_fold != 0)}/{len(beta_en_fold)}")

            signs_fold = np.sign(beta_en_fold)
            signs_fold[signs_fold == 0] = 1.0

            fold_errors = np.zeros((n_gamma, self.n_alpha))
            fold_nselected = np.zeros((n_gamma, self.n_alpha))

            # 并行化 gamma 维度
            def _eval_gamma(gamma_idx, gamma):
                raw_weights = 1.0 / (np.abs(beta_en_fold) + eps) ** gamma
                min_w = np.min(raw_weights)
                w_norm = raw_weights / min_w
                clip_max = self.weight_clip_max if self.weight_clip_max is not None else float('inf')
                weights = np.clip(w_norm, 1.0, clip_max)

                X_adaptive_tr = (X_tr * signs_fold) / weights
                X_adaptive_va = (X_va * signs_fold) / weights

                alpha_max = np.max(np.abs(X_adaptive_tr.T @ y_tr)) / len(y_tr)
                alpha_min = alpha_max * self.alpha_min_ratio
                alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

                _, coefs_path, _ = lasso_path(
                    X_adaptive_tr, y_tr,
                    alphas=alphas,
                    positive=True,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )

                preds = X_adaptive_va @ coefs_path
                mse_path = np.mean((y_va[:, np.newaxis] - preds) ** 2, axis=0)
                nselected_path = np.sum(coefs_path != 0, axis=0)

                return gamma_idx, mse_path, nselected_path

            gamma_results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_eval_gamma)(gamma_idx, gamma)
                for gamma_idx, gamma in enumerate(gamma_list)
            )

            for gamma_idx, mse_path, nselected_path in gamma_results:
                fold_errors[gamma_idx, :] = mse_path
                fold_nselected[gamma_idx, :] = nselected_path

            return fold_idx, fold_errors, fold_nselected

        # 外层 fold 串行执行（避免嵌套并行死锁），内层 gamma 已并行
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_errors, fold_nselected = _compute_single_fold_errors(fold_idx, train_idx, val_idx)
            error_matrix[:, :, fold_idx] = fold_errors
            nselected_matrix[:, :, fold_idx] = fold_nselected
            if self.verbose:
                print(f"[Fold {fold_idx + 1}/{n_folds}] completed (parallel gamma)")

        # ============================================================
        # Stage 2: Min-MSE 法则选出 best_gamma（绝对隔离）
        # alpha 的选择在 Stage 3 用 1-SE 完成
        # ============================================================
        mean_error = np.mean(error_matrix, axis=2)       # (n_gamma, n_alpha)
        mean_nselected = np.mean(nselected_matrix, axis=2)

        # Min-MSE 法则：选平均验证 MSE 最小的 gamma（不选 alpha！）
        best_gamma_idx = int(np.argmin(np.min(mean_error, axis=1)))
        self.best_gamma_ = gamma_list[best_gamma_idx]

        # 记录 Stage 2 评估结果（供参考）
        best_cv_mse = float(np.min(mean_error, axis=1)[best_gamma_idx])
        best_cv_nselected = float(mean_nselected[best_gamma_idx].min())
        self.cv_score_ = -best_cv_mse

        if self.verbose:
            gamma_mse = np.min(mean_error, axis=1)
            print(f"\n[Stage 2] Min-MSE Gamma Selection:")
            for gi, g in enumerate(gamma_list):
                min_e = float(gamma_mse[gi])
                marker = " <-- best" if gi == best_gamma_idx else ""
                print(f"  gamma={g:.2f}: min_MSE={min_e:.6f}{marker}")
            print(f"  selected: gamma={self.best_gamma_}, CV_MSE={best_cv_mse:.6f}")

        # ============================================================
        # Stage 3: 全量数据终极拟合
        # 固定 (best_gamma, best_alpha)，网格搜索 (lambda_ridge, l1_ratio)
        # ============================================================
        # Stage 3: 全量数据终极拟合
        # Step 1: 全量 ENCV(cv=5) → (λ_final, ρ_final) + beta_final
        # Step 2: 固定 best_gamma → weights
        # Step 3: 扭曲空间里 LassoCV(cv=5) → best_alpha
        # ============================================================
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X_std = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_std = X.copy()

        # Step 1: 全量 ENCV
        en_cv_final = ElasticNetCV(
            l1_ratio=list(self.l1_ratio_list),
            alphas=list(self.lambda_ridge_list),
            cv=self.cv,
            max_iter=self.max_iter, tol=self.tol,
            random_state=self.random_state, fit_intercept=True,
        )
        en_cv_final.fit(X_std, y)
        self.beta_ridge_ = en_cv_final.coef_
        self.best_lambda_ridge_ = en_cv_final.alpha_
        self.best_l1_ratio_ = en_cv_final.l1_ratio_

        if self.verbose:
            print(f"\n[Stage 3 Step1] ENCV: lambda={self.best_lambda_ridge_:.4f}, "
                  f"l1_ratio={self.best_l1_ratio_:.2f}")

        # Step 2: 固定 best_gamma → weights
        signs_final = np.sign(self.beta_ridge_)
        signs_final[signs_final == 0] = 1.0
        self.signs_ = signs_final

        raw_weights_final = 1.0 / (np.abs(self.beta_ridge_) + eps) ** self.best_gamma_
        min_w = np.min(raw_weights_final)
        c_max = self.weight_clip_max if self.weight_clip_max is not None else float('inf')
        self.weights_ = np.clip(raw_weights_final / min_w, 1.0, c_max)

        X_adaptive_final = (X_std * signs_final) / self.weights_

        # Step 3: LassoCV + 手动 1-SE 选 alpha
        alpha_max = np.max(np.abs(X_adaptive_final.T @ y)) / len(y)
        alpha_min = alpha_max * self.alpha_min_ratio
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

        lasso_cv = LassoCV(
            alphas=alphas,
            cv=self.cv,
            positive=True,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter, tol=self.tol,
            random_state=self.random_state,
        )
        lasso_cv.fit(X_adaptive_final, y)

        # 手动 1-SE 法则：在 mse_path_ 上找最稀疏的模型
        # mse_path_: shape (n_alphas, n_folds)
        mse_path = lasso_cv.mse_path_
        mean_mse = np.mean(mse_path, axis=1)
        std_mse = np.std(mse_path, axis=1) / np.sqrt(self.cv)

        min_idx = np.argmin(mean_mse)
        min_mse = float(mean_mse[min_idx])
        std_at_min = float(std_mse[min_idx])
        threshold = min_mse + std_at_min
        candidates_mask = mean_mse <= threshold

        if np.any(candidates_mask):
            # 在候选中选最大 alpha（最稀疏）
            candidate_alphas = alphas[candidates_mask]
            self.best_alpha_ = float(np.max(candidate_alphas))
        else:
            self.best_alpha_ = float(lasso_cv.alpha_)

        if self.verbose:
            print(f"[Stage 3 Step3] 1-SE Alpha: alpha={self.best_alpha_:.6f}, "
                  f"min_mse={min_mse:.6f}, threshold={threshold:.6f}")

        # 最终 Lasso
        lasso_final = Lasso(
            alpha=self.best_alpha_,
            positive=True,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter, tol=self.tol,
            random_state=self.random_state,
        )
        lasso_final.fit(X_adaptive_final, y)

        coef_final = (lasso_final.coef_ / self.weights_) * signs_final
        intercept_final = lasso_final.intercept_ if self.fit_intercept else 0.0

        # 逆标准化
        if self.standardize:
            self.coef_ = coef_final / self.scaler_.scale_
            if self.fit_intercept:
                self.intercept_ = np.mean(y) - np.sum(self.coef_ * self.scaler_.mean_)
        else:
            self.coef_ = coef_final
            if self.fit_intercept:
                self.intercept_ = intercept_final

        if self.verbose:
            print(f"\n[Stage 3] Final: lambda={self.best_lambda_ridge_:.4f}, "
                  f"l1_ratio={self.best_l1_ratio_:.2f}, alpha={self.best_alpha_:.6f}")
            print(f"[Stage 3] n_nonzero={np.sum(lasso_final.coef_ > 0)}")

        self.is_fitted_ = True
        return self

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """默认负 MSE 评分"""
        from sklearn.metrics import mean_squared_error
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred, sample_weight=sample_weight)


class AdaptiveFlippedLassoCV_EN_V2(BaseAdaptiveFlippedLasso, RegressorMixin):
    """
    AFL-CV-EN V2 — "主外主内" 策略
    ====================================================

    与 V1 的区别：
    - Stage 2: 1-SE 法则选 best_gamma（严苛考验权重空间稳健性）
    - Stage 3: Min-MSE 法则选 alpha（LassoCV 默认）

    Stage 1: 严格隔离的折内 ENCV 寻优 (Pure CV)
        K 折划分。每折内部独立：
        - StandardScaler(仅训练折)
        - ENCV(cv=3, 仅训练折) → beta_en_fold
        - 遍历 gamma 生成折内 weights
        - lasso_path + 验证折打分
        → 输出诚实 3D 误差矩阵 error_matrix[gamma, alpha, fold]

    Stage 2: 1-SE 法则选 best_gamma
        - 对每个 gamma，取其在所有 alpha 上的最小平均 MSE
        - 对各 gamma 的 min_MSE 施以 1-SE 法则
        - 选出 1-SE 容忍红线内最稀疏的 gamma

    Stage 3: 全量数据终极拟合 + Min-MSE 选 alpha
        - 全量 StandardScaler + ENCV(cv=5) → final_beta_ridge
        - 固定 best_gamma → weights
        - LassoCV(cv=5) → 直接用 LassoCV 默认的 Min-MSE alpha
    """

    def __init__(
        self,
        lambda_ridge_list: tuple = (0.1, 1.0, 10.0, 100.0),
        l1_ratio_list: tuple = (0.1, 0.5, 0.9),
        gamma_list: tuple = (0.5, 1.0, 2.0),
        cv: int = 5,
        alpha_min_ratio: float = 1e-4,
        n_alpha: int = 100,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = False,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
        use_post_ols_debiasing: bool = False,
        auto_tune_collinearity: bool = True,
        weight_clip_max: float = None,
        eps: float = 1e-5,
        n_jobs: int = -1,
    ):
        self.lambda_ridge_list = lambda_ridge_list
        self.l1_ratio_list = l1_ratio_list
        self.gamma_list = gamma_list
        self.cv = cv
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.use_post_ols_debiasing = use_post_ols_debiasing
        self.auto_tune_collinearity = auto_tune_collinearity
        self.weight_clip_max = None if (weight_clip_max is None or weight_clip_max == 'null') else weight_clip_max
        self.eps = eps
        self.n_jobs = n_jobs

        self._init_fitted_attributes()

    def get_params(self, deep: bool = True) -> dict:
        params = {
            'lambda_ridge_list': self.lambda_ridge_list,
            'l1_ratio_list': self.l1_ratio_list,
            'gamma_list': self.gamma_list,
            'cv': self.cv,
            'alpha_min_ratio': self.alpha_min_ratio,
            'n_alpha': self.n_alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'standardize': self.standardize,
            'fit_intercept': self.fit_intercept,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'use_post_ols_debiasing': self.use_post_ols_debiasing,
            'auto_tune_collinearity': self.auto_tune_collinearity,
            'weight_clip_max': self.weight_clip_max,
            'eps': self.eps,
            'n_jobs': self.n_jobs,
        }
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, cv_splits=None):
        """
        拟合 AdaptiveFlippedLassoCV_EN_V2 模型（"主外主内" 策略）

        Stage 1: 折内 ENCV 评估 (gamma, alpha) 网格
        Stage 2: 1-SE 法则选 best_gamma
        Stage 3: 全量数据终极拟合 + Min-MSE 选 alpha
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

        eps = self.eps
        gamma_list = [float(x) for x in self.gamma_list]
        n_gamma = len(gamma_list)
        n_jobs = self.n_jobs if self.n_jobs is not None else -1

        # ============================================================
        # Stage 1: 严格隔离的折内 ENCV 寻优
        # ============================================================
        if cv_splits is not None:
            n_folds = len(cv_splits)
            splits = cv_splits
        else:
            n_folds = self.cv
            kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            splits = list(kfold.split(X))

        error_matrix = np.full((n_gamma, self.n_alpha, n_folds), np.inf)
        nselected_matrix = np.zeros((n_gamma, self.n_alpha, n_folds))

        def _compute_single_fold_errors(fold_idx, train_idx, val_idx):
            X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            if self.standardize:
                scaler_fold = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
                X_tr = scaler_fold.fit_transform(X_tr_raw)
                X_va = scaler_fold.transform(X_va_raw)
            else:
                X_tr, X_va = X_tr_raw.copy(), X_va_raw.copy()

            en_cv = ElasticNetCV(
                l1_ratio=list(self.l1_ratio_list),
                alphas=list(self.lambda_ridge_list),
                cv=3,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                fit_intercept=True,
            )
            en_cv.fit(X_tr, y_tr)
            beta_en_fold = en_cv.coef_

            if self.verbose:
                print(f"  [Fold {fold_idx + 1}] ENCV: lambda={en_cv.alpha_:.4f}, "
                      f"l1_ratio={en_cv.l1_ratio_:.2f}, nonzero={np.sum(beta_en_fold != 0)}/{len(beta_en_fold)}")

            signs_fold = np.sign(beta_en_fold)
            signs_fold[signs_fold == 0] = 1.0

            fold_errors = np.zeros((n_gamma, self.n_alpha))
            fold_nselected = np.zeros((n_gamma, self.n_alpha))

            def _eval_gamma(gamma_idx, gamma):
                raw_weights = 1.0 / (np.abs(beta_en_fold) + eps) ** gamma
                min_w = np.min(raw_weights)
                w_norm = raw_weights / min_w
                clip_max = self.weight_clip_max if self.weight_clip_max is not None else float('inf')
                weights = np.clip(w_norm, 1.0, clip_max)

                X_adaptive_tr = (X_tr * signs_fold) / weights
                X_adaptive_va = (X_va * signs_fold) / weights

                alpha_max = np.max(np.abs(X_adaptive_tr.T @ y_tr)) / len(y_tr)
                alpha_min = alpha_max * self.alpha_min_ratio
                alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

                _, coefs_path, _ = lasso_path(
                    X_adaptive_tr, y_tr,
                    alphas=alphas,
                    positive=True,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )

                preds = X_adaptive_va @ coefs_path
                mse_path = np.mean((y_va[:, np.newaxis] - preds) ** 2, axis=0)
                nselected_path = np.sum(coefs_path != 0, axis=0)

                return gamma_idx, mse_path, nselected_path

            gamma_results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_eval_gamma)(gamma_idx, gamma)
                for gamma_idx, gamma in enumerate(gamma_list)
            )

            for gamma_idx, mse_path, nselected_path in gamma_results:
                fold_errors[gamma_idx, :] = mse_path
                fold_nselected[gamma_idx, :] = nselected_path

            return fold_idx, fold_errors, fold_nselected

        # 外层 fold 串行执行（避免嵌套并行死锁），内层 gamma 已并行
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_errors, fold_nselected = _compute_single_fold_errors(fold_idx, train_idx, val_idx)
            error_matrix[:, :, fold_idx] = fold_errors
            nselected_matrix[:, :, fold_idx] = fold_nselected
            if self.verbose:
                print(f"[Fold {fold_idx + 1}/{n_folds}] completed (parallel gamma)")

        # ============================================================
        # Stage 2: 1-SE 法则选 best_gamma
        # ============================================================
        mean_error = np.mean(error_matrix, axis=2)
        std_error = np.std(error_matrix, axis=2) / np.sqrt(n_folds)
        mean_nselected = np.mean(nselected_matrix, axis=2)

        # 每个 gamma 在其所有 alpha 上取最小平均 MSE
        gamma_min_mse = np.min(mean_error, axis=1)           # (n_gamma,)
        gamma_min_mse_idx = np.argmin(gamma_min_mse)         # index of best gamma

        min_mse = gamma_min_mse[gamma_min_mse_idx]
        # 找 min_mse 点对应的 std
        min_mse_alpha_idx = np.argmin(mean_error[gamma_min_mse_idx, :])
        min_std = std_error[gamma_min_mse_idx, min_mse_alpha_idx]
        threshold = min_mse + min_std

        # 1-SE 容忍红线内，选最稀疏的 gamma
        candidates_mask = gamma_min_mse <= threshold
        if np.any(candidates_mask):
            masked_nselected = np.where(candidates_mask,
                mean_nselected[np.arange(n_gamma), np.argmin(mean_error, axis=1)],
                np.inf)
            best_gamma_idx = int(np.argmin(masked_nselected))
        else:
            best_gamma_idx = gamma_min_mse_idx

        self.best_gamma_ = gamma_list[best_gamma_idx]

        if self.verbose:
            print(f"\n[Stage 2] 1-SE Gamma Selection:")
            for gi, g in enumerate(gamma_list):
                marker = " <-- best" if gi == best_gamma_idx else ""
                print(f"  gamma={g:.2f}: min_MSE={gamma_min_mse[gi]:.6f}{marker}")
            print(f"  threshold={threshold:.6f}, selected: gamma={self.best_gamma_}")

        best_cv_mse = float(gamma_min_mse[best_gamma_idx])
        self.cv_score_ = -best_cv_mse

        # ============================================================
        # Stage 3: 全量数据终极拟合 + Min-MSE 选 alpha
        # ============================================================
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X_std = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_std = X.copy()

        # Step 1: 全量 ENCV
        en_cv_final = ElasticNetCV(
            l1_ratio=list(self.l1_ratio_list),
            alphas=list(self.lambda_ridge_list),
            cv=self.cv,
            max_iter=self.max_iter, tol=self.tol,
            random_state=self.random_state, fit_intercept=True,
        )
        en_cv_final.fit(X_std, y)
        self.beta_ridge_ = en_cv_final.coef_
        self.best_lambda_ridge_ = en_cv_final.alpha_
        self.best_l1_ratio_ = en_cv_final.l1_ratio_

        if self.verbose:
            print(f"\n[Stage 3 Step1] ENCV: lambda={self.best_lambda_ridge_:.4f}, "
                  f"l1_ratio={self.best_l1_ratio_:.2f}")

        # Step 2: 固定 best_gamma → weights
        signs_final = np.sign(self.beta_ridge_)
        signs_final[signs_final == 0] = 1.0
        self.signs_ = signs_final

        raw_weights_final = 1.0 / (np.abs(self.beta_ridge_) + eps) ** self.best_gamma_
        min_w = np.min(raw_weights_final)
        c_max = self.weight_clip_max if self.weight_clip_max is not None else float('inf')
        self.weights_ = np.clip(raw_weights_final / min_w, 1.0, c_max)

        X_adaptive_final = (X_std * signs_final) / self.weights_

        # Step 3: LassoCV + Min-MSE 选 alpha（LassoCV 默认行为）
        alpha_max = np.max(np.abs(X_adaptive_final.T @ y)) / len(y)
        alpha_min = alpha_max * self.alpha_min_ratio
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

        lasso_cv = LassoCV(
            alphas=alphas,
            cv=self.cv,
            positive=True,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter, tol=self.tol,
            random_state=self.random_state,
        )
        lasso_cv.fit(X_adaptive_final, y)
        self.best_alpha_ = float(lasso_cv.alpha_)

        if self.verbose:
            print(f"[Stage 3 Step3] Min-MSE Alpha: alpha={self.best_alpha_:.6f}, "
                  f"n_nonzero={np.sum(lasso_cv.coef_ > 0)}")

        # 最终 Lasso
        lasso_final = Lasso(
            alpha=self.best_alpha_,
            positive=True,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter, tol=self.tol,
            random_state=self.random_state,
        )
        lasso_final.fit(X_adaptive_final, y)

        coef_final = (lasso_final.coef_ / self.weights_) * signs_final
        intercept_final = lasso_final.intercept_ if self.fit_intercept else 0.0

        # 逆标准化
        if self.standardize:
            self.coef_ = coef_final / self.scaler_.scale_
            if self.fit_intercept:
                self.intercept_ = np.mean(y) - np.sum(self.coef_ * self.scaler_.mean_)
        else:
            self.coef_ = coef_final
            if self.fit_intercept:
                self.intercept_ = intercept_final

        if self.verbose:
            print(f"\n[Stage 3] Final: lambda={self.best_lambda_ridge_:.4f}, "
                  f"l1_ratio={self.best_l1_ratio_:.2f}, alpha={self.best_alpha_:.6f}")
            print(f"[Stage 3] n_nonzero={np.sum(lasso_final.coef_ > 0)}")

        self.is_fitted_ = True
        return self

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """默认负 MSE 评分"""
        from sklearn.metrics import mean_squared_error
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred, sample_weight=sample_weight)

class ConfidenceCalibratedAFL(BaseAdaptiveFlippedLasso, RegressorMixin):
    """
    CC-AFL — Data-Driven Confidence-Calibrated Adaptive Flipped Lasso
    ================================================================

    一种融合了 MAD 自适应阈值、变量增广（影分身）与 1-SE 法则的终极 AFL 形态。

    算法流程：
    Stage 1: 严格隔离的折内结构探路 (Profile CV for γ)
        - K 折划分，每折内部独立
        - 折内 RidgeCV → β_prior
        - MAD 自适应阈值 τ = c · median(|β_prior_j|)
        - 划分自信集 M_conf 与摇摆集 M_unc
        - 对每个 γ：构造增广矩阵 + 非负 Lasso 路径 + 验证集 MSE

    Stage 2: Min-MSE 法则选 γ* (物理拓扑结构定调)
        - 聚合 K 折平均最小 MSE
        - 选出平均误差最小的 γ

    Stage 3: 全量空间终极校准 (1-SE Rule)
        - 全量 RidgeCV → 全局先验 β_final
        - MAD 阈值划分全局自信/摇摆集
        - 固定 γ* 构造全量增广矩阵
        - LassoCV(cv=K, positive=True) + 1-SE 法则选 α
        - 逆向缝合 → 物理空间系数 β_final

    核心创新：
    1. MAD 动态截断：优雅解决"符号崩塌"问题
    2. 变量增广技术：对摇摆集变量创建正负两个版本，保留双向搜索自由度
    3. 非负优化器约束：严格恪守底层运算规律，唤醒 Oracle Property

    参数
    ----
    lambda_ridge_list : tuple, default=(0.1, 1.0, 10.0, 100.0)
        Ridge 正则化强度候选网格
    gamma_list : tuple, default=(0.5, 1.0, 2.0)
        权重衰减指数候选网格
    cv : int, default=5
        交叉验证折数
    mad_c : float, default=0.5
        MAD 松弛系数 (τ = c · MAD)
    alpha_min_ratio : float, default=1e-4
        自动搜索时 alpha 的最小值比例
    n_alpha : int, default=100
        自动搜索时的 alpha 候选数量
    max_iter : int, default=1000
        Lasso 最大迭代次数
    tol : float, default=1e-4
        Lasso 收敛容忍度
    standardize : bool, default=False
        是否标准化特征
    fit_intercept : bool, default=True
        是否拟合截距
    random_state : int, default=2026
        随机种子
    verbose : bool, default=False
        是否输出详细信息
    weight_clip_max : float, default=100.0
        权重截断上限
    eps : float, default=1e-5
        防止除零的小常数
    n_jobs : int, default=-1
        并行任务数 (-1 = 全部核心)
    """

    def __init__(
        self,
        lambda_ridge_list: tuple = (0.1, 1.0, 10.0, 100.0),
        gamma_list: tuple = (0.5, 1.0, 2.0),
        cv: int = 5,
        mad_c: float = 0.5,
        alpha_min_ratio: float = 1e-4,
        n_alpha: int = 100,
        max_iter: int = 1000,
        tol: float = 1e-4,
        standardize: bool = False,
        fit_intercept: bool = True,
        random_state: int = 2026,
        verbose: bool = False,
        weight_clip_max: float = 100.0,
        eps: float = 1e-5,
        n_jobs: int = -1,
    ):
        self.lambda_ridge_list = lambda_ridge_list
        self.gamma_list = gamma_list
        self.cv = cv
        self.mad_c = mad_c
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.weight_clip_max = None if (weight_clip_max is None or weight_clip_max == 'null') else weight_clip_max
        self.eps = eps
        self.n_jobs = n_jobs

        self._init_fitted_attributes()

    def get_params(self, deep: bool = True) -> dict:
        return {
            'lambda_ridge_list': self.lambda_ridge_list,
            'gamma_list': self.gamma_list,
            'cv': self.cv,
            'mad_c': self.mad_c,
            'alpha_min_ratio': self.alpha_min_ratio,
            'n_alpha': self.n_alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'standardize': self.standardize,
            'fit_intercept': self.fit_intercept,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'weight_clip_max': self.weight_clip_max,
            'eps': self.eps,
            'n_jobs': self.n_jobs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    def _compute_mad_threshold(self, beta: np.ndarray) -> float:
        """
        计算 MAD 自适应阈值

        τ = c · median(|β_j| for β_j ≠ 0)

        Parameters
        ----------
        beta : np.ndarray
            Ridge 系数向量 (p,)

        Returns
        -------
        float
            MAD 阈值 τ
        """
        nonzero_vals = np.abs(beta[beta != 0])
        if len(nonzero_vals) == 0:
            return 0.0
        median_val = np.median(nonzero_vals)
        return self.mad_c * median_val

    def _compute_adaptive_weights(self, beta_ridge: np.ndarray, gamma: float) -> np.ndarray:
        """
        计算 Min-Anchored 归一化自适应权重

        w_j = min((|β_j| + ε)^(-γ) / min_m((|β_m| + ε)^(-γ)), w_max)

        Parameters
        ----------
        beta_ridge : np.ndarray
            Ridge 系数向量 (p,)
        gamma : float
            权重衰减指数

        Returns
        -------
        np.ndarray
            归一化权重向量 (p,)
        """
        eps = self.eps
        raw_weights = 1.0 / (np.abs(beta_ridge) + eps) ** gamma
        min_w = np.min(raw_weights)
        w_normalized = raw_weights / min_w
        clip_max = self.weight_clip_max if self.weight_clip_max is not None else float('inf')
        weights = np.clip(w_normalized, 1.0, clip_max)
        return weights

    def _build_augmented_matrix(
        self,
        X: np.ndarray,
        beta_ridge: np.ndarray,
        weights: np.ndarray,
        gamma: float,
    ) -> tuple:
        """
        向量化影分身增广矩阵重构 (Variable Splitting)

        对于自信集变量：X_conf = X_j · sign(β_j) / w_j
        对于摇摆集变量：X_unc^+ = X_j / w_j, X_unc^- = -X_j / w_j

        Parameters
        ----------
        X : np.ndarray
            标准化后的特征矩阵 (n, p)
        beta_ridge : np.ndarray
            Ridge 系数向量 (p,)
        weights : np.ndarray
            自适应权重 (p,)

        Returns
        -------
        tuple
            (增广矩阵 X_aug, 自信索引数组, 摇摆索引数组,
             自信起始位置, 摇摆变量起始位置, n_conf, n_unc)
        """
        signs = np.sign(beta_ridge)
        signs[signs == 0] = 1.0

        # MAD 阈值划分
        tau = self._compute_mad_threshold(beta_ridge)
        conf_mask = np.abs(beta_ridge) >= tau
        unc_mask = ~conf_mask

        n_conf = np.sum(conf_mask)
        n_unc = np.sum(unc_mask)
        n_aug = n_conf + 2 * n_unc

        if self.verbose:
            print(f"  [Augment] MAD τ={tau:.4f}, conf={n_conf}, unc={n_unc}")

        # 预分配增广矩阵（向量化操作）
        X_aug = np.zeros((X.shape[0], n_aug), dtype=X.dtype)

        # 自信集：位置 [0, n_conf)
        conf_indices = np.where(conf_mask)[0]
        if n_conf > 0:
            # 批量向量化：X_conf = X[:, conf_indices] * signs[conf_indices] / weights[conf_indices]
            X_aug[:, :n_conf] = (
                X[:, conf_indices] * signs[conf_indices] / weights[conf_indices]
            )

        # 摇摆集：位置 [n_conf, n_conf + 2*n_unc)，成对存储（阳面, 阴面）
        unc_indices = np.where(unc_mask)[0]
        if n_unc > 0:
            # 阳面：+X / w
            X_aug[:, n_conf:n_conf + n_unc] = (
                X[:, unc_indices] / weights[unc_indices]
            )
            # 阴面：-X / w
            X_aug[:, n_conf + n_unc:n_conf + 2 * n_unc] = (
                -X[:, unc_indices] / weights[unc_indices]
            )

        return X_aug, conf_indices, unc_indices, 0, n_conf, n_conf, n_unc

    def _reconstruct_coefficients(
        self,
        coef_aug: np.ndarray,
        conf_indices: np.ndarray,
        unc_indices: np.ndarray,
        conf_start: int,
        unc_start: int,
        n_conf: int,
        n_unc: int,
        signs: np.ndarray,
        weights: np.ndarray,
        n_features: int,
    ) -> np.ndarray:
        """
        向量化系数还原

        数学推导：
        - 前向变换：X_aug = X / w（除以权重）
        - Lasso 求解：y = X_aug @ θ
        - 逆变换：y = (X / w) @ θ = X @ (θ / w)

        因此还原系数必须除以权重：
        - 自信变量：β_j = θ_j * sign(β_j) / w_j
        - 摇摆变量：β_j = (θ_unc⁺ - θ_unc⁻) / w_j

        Parameters
        ----------
        coef_aug : np.ndarray
            增广空间系数向量 (n_conf + 2*n_unc,)
        conf_indices : np.ndarray
            自信集原始特征索引
        unc_indices : np.ndarray
            摇摆集原始特征索引
        conf_start : int
            自信集在增广向量中的起始位置
        unc_start : int
            摇摆集在增广向量中的起始位置
        n_conf : int
            自信集变量数
        n_unc : int
            摇摆集变量数
        signs : np.ndarray
            Ridge 系数符号向量 (p,)
        weights : np.ndarray
            自适应权重 (p,)
        n_features : int
            原始特征数 p

        Returns
        -------
        np.ndarray
            还原后的 p 维系数向量
        """
        coef_std = np.zeros(n_features, dtype=coef_aug.dtype)

        # 自信变量：β_j = θ_j * sign(β_j) / w_j
        if n_conf > 0:
            conf_coefs = coef_aug[conf_start:conf_start + n_conf]
            coef_std[conf_indices] = conf_coefs * signs[conf_indices] / weights[conf_indices]

        # 摇摆变量：β_j = (θ_unc⁺ - θ_unc⁻) / w_j
        if n_unc > 0:
            unc_plus = coef_aug[unc_start:unc_start + n_unc]
            unc_minus = coef_aug[unc_start + n_unc:unc_start + 2 * n_unc]
            coef_std[unc_indices] = (unc_plus - unc_minus) / weights[unc_indices]

        return coef_std

    def _transform_to_augmented(
        self,
        X: np.ndarray,
        beta_ridge: np.ndarray,
        weights: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        """向量化将原始矩阵变换为增广空间"""
        signs = np.sign(beta_ridge)
        signs[signs == 0] = 1.0
        tau = self._compute_mad_threshold(beta_ridge)

        conf_mask = np.abs(beta_ridge) >= tau
        unc_mask = ~conf_mask

        n_conf = np.sum(conf_mask)
        n_unc = np.sum(unc_mask)
        n_aug = n_conf + 2 * n_unc

        X_aug = np.zeros((X.shape[0], n_aug), dtype=X.dtype)

        # 自信集
        conf_indices = np.where(conf_mask)[0]
        if n_conf > 0:
            X_aug[:, :n_conf] = X[:, conf_indices] * signs[conf_indices] / weights[conf_indices]

        # 摇摆集（阳面和阴面）
        unc_indices = np.where(unc_mask)[0]
        if n_unc > 0:
            X_aug[:, n_conf:n_conf + n_unc] = X[:, unc_indices] / weights[unc_indices]
            X_aug[:, n_conf + n_unc:n_conf + 2 * n_unc] = -X[:, unc_indices] / weights[unc_indices]

        return X_aug

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, cv_splits=None):
        """
        拟合 CC-AFL 模型

        Parameters
        ----------
        X : np.ndarray
            Training data (n, p)
        y : np.ndarray
            Target values (n,)
        sample_weight : np.ndarray, optional
            Sample weights
        cv_splits : list of tuples, optional
            Pre-generated CV splits for fair comparison
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error

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
        n = X.shape[0]
        p = self.n_features_in_

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=_DTYPE)

        eps = self.eps
        gamma_list = [float(x) for x in self.gamma_list]
        n_gamma = len(gamma_list)

        # ============================================================
        # Stage 1: 严格隔离的折内结构探路
        # ============================================================
        if cv_splits is not None:
            n_folds = len(cv_splits)
            splits = cv_splits
            if self.verbose:
                print(f"[CC-AFL] Using provided {n_folds}-fold CV splits")
        else:
            n_folds = self.cv
            kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            splits = list(kfold.split(X))

        error_matrix = np.full((n_gamma, self.n_alpha, n_folds), np.inf)
        nselected_matrix = np.zeros((n_gamma, self.n_alpha, n_folds))

        # 外层 fold 串行执行（避免嵌套并行死锁）
        # 内层 gamma 并行执行
        n_jobs = self.n_jobs if self.n_jobs is not None else -1

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            if self.standardize:
                scaler_fold = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
                X_tr = scaler_fold.fit_transform(X_tr_raw)
                X_va = scaler_fold.transform(X_va_raw)
            else:
                X_tr, X_va = X_tr_raw.copy(), X_va_raw.copy()

            # 折内 RidgeCV
            ridge_cv = RidgeCV(alphas=self.lambda_ridge_list, cv=3)
            ridge_cv.fit(X_tr, y_tr)
            beta_ridge_fold = ridge_cv.coef_

            fold_errors = np.zeros((n_gamma, self.n_alpha))
            fold_nselected = np.zeros((n_gamma, self.n_alpha))

            # 内层 gamma 并行
            def _eval_gamma(gamma_idx, gamma):
                weights = self._compute_adaptive_weights(beta_ridge_fold, gamma)

                # 构造增广矩阵（训练集）
                X_aug_tr, conf_idx, unc_idx, _, _, n_conf, n_unc = self._build_augmented_matrix(
                    X_tr, beta_ridge_fold, weights, gamma
                )

                # 预计算验证集的增广矩阵
                X_aug_va = self._transform_to_augmented(X_va, beta_ridge_fold, weights, gamma)

                # alpha 路径
                alpha_max = np.max(np.abs(X_aug_tr.T @ y_tr)) / len(y_tr)
                alpha_min = alpha_max * self.alpha_min_ratio
                alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

                # lasso_path
                _, coefs_path, _ = lasso_path(
                    X_aug_tr, y_tr,
                    alphas=alphas,
                    positive=True,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )

                # 验证集预测（直接用增广矩阵）
                if X_aug_va.shape[1] > 0:
                    preds = X_aug_va @ coefs_path
                else:
                    preds = np.zeros((len(y_va), len(alphas)))

                mse_path = np.mean((y_va[:, np.newaxis] - preds) ** 2, axis=0)
                nselected_path = np.sum(coefs_path != 0, axis=0)

                return gamma_idx, mse_path, nselected_path

            gamma_results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_eval_gamma)(gamma_idx, gamma)
                for gamma_idx, gamma in enumerate(gamma_list)
            )

            for gamma_idx, mse_path, nselected_path in gamma_results:
                fold_errors[gamma_idx, :] = mse_path
                fold_nselected[gamma_idx, :] = nselected_path

            error_matrix[:, :, fold_idx] = fold_errors
            nselected_matrix[:, :, fold_idx] = fold_nselected
            if self.verbose:
                print(f"[Fold {fold_idx + 1}/{n_folds}] completed (parallel gamma)")

        # ============================================================
        # Stage 2: Min-MSE 法则选 γ*
        # ============================================================
        mean_error = np.mean(error_matrix, axis=2)  # (n_gamma, n_alpha)
        mean_nselected = np.mean(nselected_matrix, axis=2)

        # 每个 gamma 取其所有 alpha 上的最小平均 MSE
        min_MSE_gamma = np.min(mean_error, axis=1)  # (n_gamma,)
        best_gamma_idx = int(np.argmin(min_MSE_gamma))
        self.best_gamma_ = gamma_list[best_gamma_idx]
        best_cv_mse = float(min_MSE_gamma[best_gamma_idx])
        self.cv_score_ = -best_cv_mse

        if self.verbose:
            print(f"\n[Stage 2] Min-MSE Gamma Selection:")
            for gi, g in enumerate(gamma_list):
                marker = " <-- best" if gi == best_gamma_idx else ""
                print(f"  gamma={g:.2f}: min_MSE={float(min_MSE_gamma[gi]):.6f}{marker}")
            print(f"  selected: gamma={self.best_gamma_}, CV_MSE={best_cv_mse:.6f}")

        # ============================================================
        # Stage 3: 全量空间终极校准
        # ============================================================
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X_std = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_std = X.copy()

        # Step 1: 全量 RidgeCV
        ridge_final = RidgeCV(alphas=self.lambda_ridge_list, cv=self.cv)
        ridge_final.fit(X_std, y)
        self.beta_ridge_ = ridge_final.coef_
        self.best_lambda_ridge_ = ridge_final.alpha_

        if self.verbose:
            print(f"\n[Stage 3] Full-data Ridge: lambda_ridge={self.best_lambda_ridge_:.4f}")

        # Step 2: MAD 阈值划分
        tau_final = self._compute_mad_threshold(self.beta_ridge_)
        conf_mask_final = np.abs(self.beta_ridge_) >= tau_final
        n_conf = np.sum(conf_mask_final)
        n_unc = np.sum(~conf_mask_final)

        if self.verbose:
            print(f"[Stage 3] MAD tau={tau_final:.4f}, conf={n_conf}, unc={n_unc}")

        # Step 3: 固定 gamma* 计算权重 + 增广矩阵
        signs_final = np.sign(self.beta_ridge_)
        signs_final[signs_final == 0] = 1.0
        self.signs_ = signs_final

        weights_final = self._compute_adaptive_weights(self.beta_ridge_, self.best_gamma_)
        self.weights_ = weights_final

        X_aug_final, conf_idx_final, unc_idx_final, conf_start, unc_start, n_conf_final, n_unc_final = self._build_augmented_matrix(
            X_std, self.beta_ridge_, weights_final, self.best_gamma_
        )

        if self.verbose:
            print(f"[Stage 3] Augmented matrix: {X_aug_final.shape}")

        # Step 4: LassoCV + 1-SE 法则
        alpha_max = np.max(np.abs(X_aug_final.T @ y)) / len(y)
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
        )
        lasso_cv.fit(X_aug_final, y)

        # 1-SE 法则
        mse_path = lasso_cv.mse_path_
        mean_mse = np.mean(mse_path, axis=1)
        std_mse = np.std(mse_path, axis=1) / np.sqrt(self.cv)

        min_idx = np.argmin(mean_mse)
        min_mse_value = float(mean_mse[min_idx])
        std_at_min = float(std_mse[min_idx])
        threshold = min_mse_value + std_at_min

        candidates_mask = mean_mse <= threshold
        if np.any(candidates_mask):
            candidate_alphas = alphas[candidates_mask]
            self.best_alpha_ = float(np.max(candidate_alphas))
        else:
            self.best_alpha_ = float(lasso_cv.alpha_)

        if self.verbose:
            print(f"[Stage 3] 1-SE Alpha: alpha={self.best_alpha_:.6f}, "
                  f"min_mse={min_mse_value:.6f}, threshold={threshold:.6f}")

        # Step 5: 最终 Lasso
        lasso_final = Lasso(
            alpha=self.best_alpha_,
            positive=True,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        lasso_final.fit(X_aug_final, y)

        # Step 6: 逆向缝合（必须除以权重还原到物理空间）
        coef_aug_final = lasso_final.coef_
        coef_std = self._reconstruct_coefficients(
            coef_aug_final, conf_idx_final, unc_idx_final,
            conf_start, unc_start, n_conf_final, n_unc_final,
            signs_final, weights_final, p
        )

        # 逆标准化
        if self.standardize:
            self.coef_ = coef_std / self.scaler_.scale_
            if self.fit_intercept:
                self.intercept_ = np.mean(y) - np.sum(self.coef_ * self.scaler_.mean_)
        else:
            self.coef_ = coef_std
            if self.fit_intercept:
                self.intercept_ = lasso_final.intercept_

        if self.verbose:
            n_nonzero = np.sum(self.coef_ != 0)
            print(f"\n[Stage 3] Final: gamma={self.best_gamma_}, "
                  f"alpha={self.best_alpha_:.6f}, n_nonzero={n_nonzero}")

        self.is_fitted_ = True
        return self

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """默认负 MSE 评分"""
        from sklearn.metrics import mean_squared_error
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred, sample_weight=sample_weight)


class ConfidenceCalibratedAFLClassifier(BaseAdaptiveFlippedLasso, ClassifierMixin):
    """
    CC-AFL 二分类器

    使用 ConfidenceCalibratedAFL 作为回归核心，通过 sigmoid 链接函数实现二分类。
    """

    def __init__(self, **kwargs):
        # Extract CC-AFL specific params
        self.lambda_ridge_list = kwargs.pop('lambda_ridge_list', (0.1, 1.0, 10.0, 100.0))
        self.gamma_list = kwargs.pop('gamma_list', (0.5, 1.0, 2.0))
        self.cv = kwargs.pop('cv', 5)
        self.mad_c = kwargs.pop('mad_c', 0.5)
        self.alpha_min_ratio = kwargs.pop('alpha_min_ratio', 1e-4)
        self.n_alpha = kwargs.pop('n_alpha', 100)
        self.max_iter = kwargs.pop('max_iter', 1000)
        self.tol = kwargs.pop('tol', 1e-4)
        self.standardize = kwargs.pop('standardize', False)
        self.fit_intercept = kwargs.pop('fit_intercept', True)
        self.random_state = kwargs.pop('random_state', 2026)
        self.verbose = kwargs.pop('verbose', False)
        self.weight_clip_max = kwargs.pop('weight_clip_max', 100.0)
        self.eps = kwargs.pop('eps', 1e-5)
        self.n_jobs = kwargs.pop('n_jobs', -1)

        self._init_fitted_attributes()
        self.classes_ = None
        self.task_type_ = 'classification'

    def get_params(self, deep: bool = True) -> dict:
        return {
            'lambda_ridge_list': self.lambda_ridge_list,
            'gamma_list': self.gamma_list,
            'cv': self.cv,
            'mad_c': self.mad_c,
            'alpha_min_ratio': self.alpha_min_ratio,
            'n_alpha': self.n_alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'standardize': self.standardize,
            'fit_intercept': self.fit_intercept,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'weight_clip_max': self.weight_clip_max,
            'eps': self.eps,
            'n_jobs': self.n_jobs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, cv_splits=None):
        """拟合 CC-AFL 分类器"""
        from sklearn.utils.validation import check_X_y

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("ConfidenceCalibratedAFLClassifier only supports binary classification")

        # 转换为连续响应
        y_continuous = (y == self.classes_[1]).astype(np.float64)

        # 使用 CC-AFL 回归器
        ccaf = ConfidenceCalibratedAFL(
            lambda_ridge_list=self.lambda_ridge_list,
            gamma_list=self.gamma_list,
            cv=self.cv,
            mad_c=self.mad_c,
            alpha_min_ratio=self.alpha_min_ratio,
            n_alpha=self.n_alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            standardize=self.standardize,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
            verbose=self.verbose,
            weight_clip_max=self.weight_clip_max,
            eps=self.eps,
            n_jobs=self.n_jobs,
        )
        ccaf.fit(X, y_continuous, sample_weight, cv_splits)

        # 复制属性
        self.coef_ = ccaf.coef_
        self.intercept_ = ccaf.intercept_
        self.scaler_ = ccaf.scaler_
        self.is_fitted_ = True
        self.best_gamma_ = ccaf.best_gamma_
        self.best_alpha_ = ccaf.best_alpha_
        self.cv_score_ = ccaf.cv_score_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        check_is_fitted(self, 'is_fitted_')
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        check_is_fitted(self, 'is_fitted_')
        z = X @ self.coef_ + self.intercept_
        proba_1 = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - proba_1, proba_1])

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """默认准确率评分"""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
