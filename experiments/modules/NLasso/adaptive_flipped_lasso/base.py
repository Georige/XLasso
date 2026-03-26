"""
AdaptiveFlippedLasso 基类定义
遵循 scikit-learn Estimator 接口规范
"""
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, RidgeCV, lasso_path

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
        }
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        拟合 AdaptiveFlippedLassoCV 模型（严格隔离版）
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

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=_DTYPE)

        # 标准化（仅在阶段三最终拟合时对全量数据做一次，
        # 阶段一/二的 CV 严格隔离中不涉及标准化，保持与 spec 一致）
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X_for_cv = self.scaler_.fit_transform(X)
        else:
            X_for_cv = X
            self.scaler_ = None

        n = X_for_cv.shape[0]
        n_gamma = len(self.gamma_list)
        eps = 1e-5

        # ============================================================
        # 阶段一：K 折严格内部寻优
        # ============================================================
        error_matrix = np.full((n_gamma, self.n_alpha, self.cv), np.inf)
        kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_for_cv)):
            X_tr, X_va = X_for_cv[train_idx], X_for_cv[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            if self.verbose:
                print(f"[Fold {fold_idx + 1}/{self.cv}] train={len(train_idx)}, val={len(val_idx)}")

            # 先验提取：仅在训练折上做 RidgeCV（绝对隔离）
            ridge_cv = RidgeCV(alphas=self.lambda_ridge_list, cv=3)
            ridge_cv.fit(X_tr, y_tr)
            beta_ridge_fold = ridge_cv.coef_

            signs_fold = np.sign(beta_ridge_fold)
            signs_fold[signs_fold == 0] = 1.0

            # 对每个 gamma 分别计算其专属的 alpha 搜索路径
            # （不同 gamma 下 X_adaptive 尺度不同，alpha_max 差异可达数个数量级）
            for gamma_idx, gamma in enumerate(self.gamma_list):
                # 计算该 gamma 对应的权重
                raw_weights = 1.0 / (np.abs(beta_ridge_fold) + eps) ** gamma
                weights = raw_weights / np.mean(raw_weights)

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

        # ============================================================
        # 阶段二：选拔最优参数
        # ============================================================
        mean_error = np.mean(error_matrix, axis=2)  # (n_gamma, n_alpha)

        best_gamma_idx, best_alpha_idx = np.unravel_index(
            np.argmin(mean_error), mean_error.shape
        )
        self.best_gamma_ = self.gamma_list[best_gamma_idx]

        # 重建该 gamma 对应的 alpha 路径（用于取 best_alpha）
        ridge_cv_tmp = RidgeCV(alphas=self.lambda_ridge_list, cv=3)
        ridge_cv_tmp.fit(X_for_cv, y)
        beta_ridge_tmp = ridge_cv_tmp.coef_
        signs_tmp = np.sign(beta_ridge_tmp)
        signs_tmp[signs_tmp == 0] = 1.0
        raw_weights_tmp = 1.0 / (np.abs(beta_ridge_tmp) + eps) ** self.best_gamma_
        weights_tmp = raw_weights_tmp / np.mean(raw_weights_tmp)
        X_adaptive_tmp = (X_for_cv * signs_tmp) / weights_tmp
        alpha_max_tmp = np.max(np.abs(X_adaptive_tmp.T @ y)) / len(y)
        alpha_min_tmp = alpha_max_tmp * self.alpha_min_ratio
        alphas_final = np.logspace(np.log10(alpha_min_tmp), np.log10(alpha_max_tmp), self.n_alpha)[::-1]
        self.best_alpha_ = alphas_final[best_alpha_idx]
        best_cv_mse = mean_error[best_gamma_idx, best_alpha_idx]
        self.cv_score_ = -best_cv_mse

        if self.verbose:
            print(f"\n[Stage 2] Best: gamma={self.best_gamma_}, alpha={self.best_alpha_:.6f}, CV_MSE={best_cv_mse:.6f}")

        # ============================================================
        # 阶段三：全量数据终极拟合
        # ============================================================
        ridge_final = RidgeCV(alphas=self.lambda_ridge_list, cv=self.cv)
        ridge_final.fit(X_for_cv, y)
        self.beta_ridge_ = ridge_final.coef_
        self.best_lambda_ridge_ = ridge_final.alpha_

        signs_final = np.sign(self.beta_ridge_)
        signs_final[signs_final == 0] = 1.0
        self.signs_ = signs_final

        raw_weights_final = 1.0 / (np.abs(self.beta_ridge_) + eps) ** self.best_gamma_
        self.weights_ = raw_weights_final / np.mean(raw_weights_final)

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
        """默认负 MSE 评分（与 GridSearchCV 兼容）"""
        from sklearn.metrics import mean_squared_error
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred, sample_weight=sample_weight)
