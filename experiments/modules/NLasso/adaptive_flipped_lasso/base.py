"""
AdaptiveFlippedLasso 基类定义
遵循 scikit-learn Estimator 接口规范
"""
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, RidgeCV, lasso_path, LinearRegression

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
        self.weight_clip_max = weight_clip_max

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
        self.weight_clip_max = weight_clip_max

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
        eps = 1e-5

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

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # 每个 fold 独立标准化：仅用训练集 fit，用训练集统计量 transform 验证集
            if self.standardize:
                scaler_fold = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
                X_tr = scaler_fold.fit_transform(X_tr_raw)
                X_va = scaler_fold.transform(X_va_raw)
            else:
                X_tr, X_va = X_tr_raw, X_va_raw

            if self.verbose:
                print(f"[Fold {fold_idx + 1}/{n_folds}] train={len(train_idx)}, val={len(val_idx)}")

            # 先验提取：仅在训练折上做 RidgeCV（绝对隔离）
            ridge_cv = RidgeCV(alphas=self.lambda_ridge_list, cv=3)
            ridge_cv.fit(X_tr, y_tr)
            beta_ridge_fold = ridge_cv.coef_

            signs_fold = np.sign(beta_ridge_fold)
            signs_fold[signs_fold == 0] = 1.0

            # 对每个 gamma 分别计算其专属的 alpha 搜索路径
            # （不同 gamma 下 X_adaptive 尺度不同，alpha_max 差异可达数个数量级）
            for gamma_idx, gamma in enumerate(self.gamma_list):
                # Min-Anchored Normalization
                raw_weights = 1.0 / (np.abs(beta_ridge_fold) + eps) ** gamma
                min_w = np.min(raw_weights)
                w_normalized = raw_weights / min_w
                clip_max = self.weight_clip_max if self.weight_clip_max is not None else float('inf')
                weights = np.clip(w_normalized, 1.0, clip_max)

                # 空间重构：分别扭曲训练折和验证折
                X_adaptive_tr = (X_tr * signs_fold) / weights
                X_adaptive_va = (X_va * signs_fold) / weights

                # alpha 搜索路径（每个 gamma 有自己专属的 alpha_max）
                alpha_max = np.max(np.abs(X_adaptive_tr.T @ y_tr)) / len(y_tr)
                alpha_min = alpha_max * self.alpha_min_ratio
                alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

                # lasso_path：一次性算出全部 alpha 的系数
                _, coefs_path, _ = lasso_path(
                    X_adaptive_tr, y_tr,
                    alphas=alphas,
                    positive=True,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
                # coefs_path: (p, n_alphas)

                # 在验证集上打分
                preds = X_adaptive_va @ coefs_path  # (n_val, n_alphas)
                mse_path = np.mean((y_va[:, np.newaxis] - preds) ** 2, axis=0)
                error_matrix[gamma_idx, :, fold_idx] = mse_path

                # 追踪每个 alpha 的非零系数数量（用于 1-SE 法则）
                nselected_path = np.sum(coefs_path != 0, axis=0)  # (n_alphas,)
                nselected_matrix[gamma_idx, :, fold_idx] = nselected_path

        # ============================================================
        # 阶段二：选拔最优参数（1-SE 法则）
        # ============================================================
        mean_error = np.mean(error_matrix, axis=2)  # (n_gamma, n_alpha)
        std_error = np.std(error_matrix, axis=2) / np.sqrt(n_folds)  # SE of mean
        mean_nselected = np.mean(nselected_matrix, axis=2)  # (n_gamma, n_alpha)

        # 1-SE 法则：选择最稀疏的模型，其 MSE 在 min(MSE) + 1*SE 范围内
        min_mse = np.min(mean_error)
        min_mse_idx = np.unravel_index(np.argmin(mean_error), mean_error.shape)
        min_std = std_error[min_mse_idx]
        threshold = min_mse + min_std

        # 找到所有 MSE <= threshold 的候选组合
        candidates_mask = mean_error <= threshold

        if not np.any(candidates_mask):
            # Fallback：如果没有候选（不应该发生），使用原始最小 MSE
            if self.verbose:
                print("[Stage 2] Warning: No candidates within 1-SE, using standard min-MSE")
            best_gamma_idx, best_alpha_idx = np.unravel_index(
                np.argmin(mean_error), mean_error.shape
            )
        else:
            # 在候选中选择 n_selected 最少的（最稀疏的）
            masked_nselected = np.where(candidates_mask, mean_nselected, np.inf)
            best_flat_idx = np.argmin(masked_nselected)
            best_gamma_idx, best_alpha_idx = np.unravel_index(best_flat_idx, mean_error.shape)

        # For high-dimensional data (p >> n), prefer min MSE over sparse 1-SE
        # This prevents the algorithm from selecting empty models due to unstable ridge coefficients
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if n_features > n_samples * 2:
            if self.verbose:
                print("[Stage 2] High-dimensional data detected, using min MSE instead of 1-SE")
            best_gamma_idx, best_alpha_idx = np.unravel_index(
                np.argmin(mean_error), mean_error.shape
            )

        self.best_gamma_ = self.gamma_list[best_gamma_idx]

        # 重建该 gamma 对应的 alpha 路径（用于取 best_alpha）
        if self.standardize:
            X_for_stage2 = self.scaler_.fit_transform(X)
        else:
            X_for_stage2 = X
        ridge_cv_tmp = RidgeCV(alphas=self.lambda_ridge_list, cv=3)
        ridge_cv_tmp.fit(X_for_stage2, y)
        beta_ridge_tmp = ridge_cv_tmp.coef_
        signs_tmp = np.sign(beta_ridge_tmp)
        signs_tmp[signs_tmp == 0] = 1.0
        raw_weights_tmp = 1.0 / (np.abs(beta_ridge_tmp) + eps) ** self.best_gamma_
        min_w = np.min(raw_weights_tmp)
        weights_tmp = np.clip(raw_weights_tmp / min_w, 1.0, self.weight_clip_max if self.weight_clip_max is not None else float('inf'))
        X_adaptive_tmp = (X_for_stage2 * signs_tmp) / weights_tmp
        alpha_max_tmp = np.max(np.abs(X_adaptive_tmp.T @ y)) / len(y)
        alpha_min_tmp = alpha_max_tmp * self.alpha_min_ratio
        alphas_final = np.logspace(np.log10(alpha_min_tmp), np.log10(alpha_max_tmp), self.n_alpha)[::-1]
        self.best_alpha_ = alphas_final[best_alpha_idx]

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
        if self.standardize:
            X_for_cv = self.scaler_.fit_transform(X)
        else:
            X_for_cv = X

        ridge_final = RidgeCV(alphas=self.lambda_ridge_list, cv=self.cv)
        ridge_final.fit(X_for_cv, y)
        self.beta_ridge_ = ridge_final.coef_
        self.best_lambda_ridge_ = ridge_final.alpha_

        signs_final = np.sign(self.beta_ridge_)
        signs_final[signs_final == 0] = 1.0
        self.signs_ = signs_final

        raw_weights_final = 1.0 / (np.abs(self.beta_ridge_) + eps) ** self.best_gamma_
        min_w = np.min(raw_weights_final)
        self.weights_ = np.clip(raw_weights_final / min_w, 1.0, self.weight_clip_max if self.weight_clip_max is not None else float('inf'))

        X_adaptive_final = (X_for_cv * signs_final) / self.weights_

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
        intercept_ols = None
        if self.use_post_ols_debiasing:
            selected_mask = lasso_final.coef_ != 0
            n_selected = np.sum(selected_mask)

            if n_selected > 0 and n_selected < X.shape[1]:
                # Fit OLS on selected features only (un-penalized)
                X_selected = X_for_cv[:, selected_mask]
                ols = LinearRegression(fit_intercept=self.fit_intercept)
                ols.fit(X_selected, y)
                # Create full coef vector with OLS coefficients at selected indices
                coef_debiased = np.zeros(X.shape[1])
                coef_debiased[selected_mask] = ols.coef_
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
                self.intercept_ = self.scaler_.mean_[0] - np.sum(self.coef_ * self.scaler_.mean_)
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
                self.intercept_ = self.scaler_.mean_[0] - np.sum(self.coef_ * self.scaler_.mean_)
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
        self.weight_clip_max = weight_clip_max

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
                self.intercept_ = self.scaler_.mean_[0] - np.sum(self.coef_ * self.scaler_.mean_)
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