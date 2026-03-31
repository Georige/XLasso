"""
AP-AFL: Asymmetrically Penalized Adaptive Flipped Lasso
非对称自适应翻转 Lasso (Asymmetrically Penalized Adaptive Lasso)

通过"影分身（Variable Splitting）"数学技巧，将非对称惩罚转化为标准 Lasso(positive=True) 可解的形式。

算法流程：
    Stage 1: K 折严格隔离的折内结构探路 (Profile CV for γ)
    Stage 2: 物理结构定调 (Min-MSE Selection of γ)
    Stage 3: 全量空间终极校准 (1-SE Rule on Full Data)

特性：
    - 容错性：通过 κ 倍惩罚的逆流变量，即使 Ridge 先验给出错误符号也能跨过软墙
    - 解耦优势：Stage 1/2 专注 γ（权重拓扑），Stage 3 专注 α（稀疏尺度）
    - 单重收缩：只在最后一步施加 1-SE 法则，避免二次征税
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, lasso_path
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import warnings

# 性能优化常量
_COPY_WHEN_POSSIBLE = False
_DTYPE = np.float64


class APAFLRegressor(BaseEstimator, RegressorMixin):
    """
    AP-AFL 回归器：非对称自适应翻转 Lasso

    通过变量增广（Variable Splitting）将非对称惩罚转化为标准 Lasso 可解形式。

    参数
    ----
    kappa : float, default=100.0
        非对称因子。逆流变量（阴面）承受 κ 倍惩罚。
        建议值：50 或 100。值越大，对错误符号的容忍度越高。
    gamma_list : tuple, default=(0.3, 0.5, 1.0, 2.0)
        自适应权重的指数候选网格。
    lambda_ridge_list : tuple, default=(0.1, 1.0, 10.0, 100.0)
        第一阶段 Ridge 正则化强度候选。
    cv : int, default=5
        交叉验证折数。
    alpha_min_ratio : float, default=1e-4
        Lasso alpha 路径的最小值比例。
    n_alpha : int, default=100
        Lasso alpha 候选数量。
    max_iter : int, default=1000
        Lasso 最大迭代次数。
    tol : float, default=1e-4
        Lasso 收敛容忍度。
    standardize : bool, default=False
        是否在 CV 内部标准化（推荐 False 避免数据泄露）。
    fit_intercept : bool, default=True
        是否拟合截距。
    random_state : int, default=2026
        随机种子。
    verbose : bool, default=False
        是否输出详细信息。
    weight_clip_max : float, default=100.0
        权重裁剪上限。None 表示不裁剪。
    eps : float, default=1e-5
        防止除零的小常数。
    n_jobs : int, default=-1
        并行 jobs 数量（-1 表示全部核心）。
    """

    def __init__(
        self,
        kappa: float = 100.0,
        gamma_list: tuple = (0.3, 0.5, 1.0, 2.0),
        lambda_ridge_list: tuple = (0.1, 1.0, 10.0, 100.0),
        cv: int = 5,
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
        self.kappa = kappa
        self.gamma_list = gamma_list
        self.lambda_ridge_list = lambda_ridge_list
        self.cv = cv
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.weight_clip_max = weight_clip_max
        self.eps = eps
        self.n_jobs = n_jobs

        # 初始化属性
        self.coef_ = None
        self.intercept_ = 0.0
        self.scaler_ = None
        self.is_fitted_ = False
        self.n_features_in_ = None

    def get_params(self, deep: bool = True) -> dict:
        return {
            'kappa': self.kappa,
            'gamma_list': self.gamma_list,
            'lambda_ridge_list': self.lambda_ridge_list,
            'cv': self.cv,
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

    def _build_augmented_matrix(
        self,
        X: np.ndarray,
        signs: np.ndarray,
        weights: np.ndarray,
    ) -> tuple:
        """
        构建非对称增广矩阵（影分身/Variable Splitting）

        X_con (阳面/顺流):  (X * signs) / weights
        X_dis (阴面/逆流): (-X * signs) / (kappa * weights)

        Returns:
            X_con: (n, p)
            X_dis: (n, p)
            X_aug: (n, 2p) - 拼接后的增广矩阵
        """
        kappa = self.kappa

        # 阳面（保守方向）：轻微惩罚
        # weights >= 1.0（已通过clip保证），无需eps防除零
        X_con = (X * signs) / weights

        # 阴面（激进方向）：κ 倍惩罚
        X_dis = (-X * signs) / (kappa * weights)

        # 增广矩阵：阳面拼接阴面
        X_aug = np.hstack([X_con, X_dis])

        return X_con, X_dis, X_aug

    def _reconstruct_coef(
        self,
        theta_con: np.ndarray,
        theta_dis: np.ndarray,
        signs: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        从增广系数还原原始空间系数

        β_std = sign(β_ridge) * (θ_con / w - θ_dis / (κ * w))

        参数
        ----
        theta_con : (p,) - 阳面系数
        theta_dis : (p,) - 阴面系数
        signs : (p,) - Ridge 系数符号
        weights : (p,) - 自适应权重

        Returns
        ----
        beta_std : (p,) - 标准化空间的系数
        """
        kappa = self.kappa

        # β_std = signs * (θ_con / w - θ_dis / (κ * w))
        # weights >= 1.0（已通过clip保证），无需eps防除零
        beta_std = signs * (theta_con / weights - theta_dis / (kappa * weights))

        return beta_std

    def _compute_weights(self, beta_ridge: np.ndarray, gamma: float) -> np.ndarray:
        """
        计算自适应权重：w_j = clip((|β_j| + ε)^(-γ) / norm, 1, w_max)

        使用 Min-Anchored Normalization：最强制信号锚定为 1.0
        """
        eps = self.eps
        clip_max = self.weight_clip_max if self.weight_clip_max is not None else float('inf')

        raw_weights = 1.0 / (np.abs(beta_ridge) + eps) ** gamma
        min_w = np.min(raw_weights)
        w_normalized = raw_weights / min_w
        weights = np.clip(w_normalized, 1.0, clip_max)

        return weights

    def _stage1_profile_cv(self, X, y, cv_splits=None):
        """
        Stage 1: 严格隔离的折内结构探路

        对每个 gamma，在 K 折上计算验证集 MSE 路径。
        返回 error_matrix[gamma_idx, alpha_idx, fold_idx]

        并行策略：
        - 外层：Fold 顺序循环（RidgeCV很快）
        - 内层：Gamma 并行 (n_gamma jobs)
        """
        n_gamma = len(self.gamma_list)
        n_folds = self.cv if cv_splits is None else len(cv_splits)
        eps = self.eps

        # CV splits
        if cv_splits is not None:
            splits = cv_splits
        else:
            kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            splits = list(kfold.split(X))

        # 预分配误差矩阵
        error_matrix = np.full((n_gamma, self.n_alpha, n_folds), np.inf)
        nselected_matrix = np.zeros((n_gamma, self.n_alpha, n_folds))

        def _compute_single_gamma(gamma, X_tr, y_tr, X_va, y_va, signs_fold, beta_ridge_fold):
            """对单个 gamma 计算验证集 MSE 路径"""
            # weights_fold 依赖 gamma，必须在此计算
            weights_fold = self._compute_weights(beta_ridge_fold, gamma)

            # 构造增广矩阵
            X_con_tr, X_dis_tr, X_aug_tr = self._build_augmented_matrix(
                X_tr, signs_fold, weights_fold
            )

            # alpha 搜索路径
            alpha_max = np.max(np.abs(X_aug_tr.T @ y_tr)) / len(y_tr)
            alpha_min = alpha_max * self.alpha_min_ratio
            alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), self.n_alpha)[::-1]

            # lasso_path 在增广矩阵上求解
            _, coefs_path, _ = lasso_path(
                X_aug_tr, y_tr,
                alphas=alphas,
                positive=True,
                max_iter=self.max_iter,
                tol=self.tol,
            )

            # 系数还原
            p = X_tr.shape[1]
            theta_con_path = coefs_path[:p, :]
            theta_dis_path = coefs_path[p:, :]

            w_expanded = weights_fold[:, np.newaxis]
            signs_expanded = signs_fold[:, np.newaxis]

            beta_std_path = signs_expanded * (
                theta_con_path / w_expanded - theta_dis_path / (self.kappa * w_expanded)
            )

            # 预测并计算 MSE
            preds_va = X_va @ beta_std_path
            mse_path = np.mean((y_va[:, np.newaxis] - preds_va) ** 2, axis=0)
            nselected = np.sum(beta_std_path != 0, axis=0)

            return gamma, mse_path, nselected, alphas

        # 外层 Fold 顺序循环，内层 Gamma 并行
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # 折内标准化（严格隔离）
            if self.standardize:
                scaler_fold = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
                X_tr = scaler_fold.fit_transform(X_tr_raw)
                X_va = scaler_fold.transform(X_va_raw)
            else:
                X_tr, X_va = X_tr_raw, X_va_raw

            # Stage 1a: RidgeCV 仅在训练折上提取先验
            ridge_cv = RidgeCV(alphas=self.lambda_ridge_list, cv=3)
            ridge_cv.fit(X_tr, y_tr)
            beta_ridge_fold = ridge_cv.coef_

            signs_fold = np.sign(beta_ridge_fold)
            signs_fold[signs_fold == 0] = 1.0

            # 内层并行：gamma 级别
            # weights_fold 在 _compute_single_gamma 内部计算（依赖 gamma）
            gamma_results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(_compute_single_gamma)(
                    gamma, X_tr, y_tr, X_va, y_va,
                    signs_fold, beta_ridge_fold
                )
                for gamma in self.gamma_list
            )

            for gamma, mse_path, nselected, alphas in gamma_results:
                gamma_idx = self.gamma_list.index(gamma)
                error_matrix[gamma_idx, :, fold_idx] = mse_path
                nselected_matrix[gamma_idx, :, fold_idx] = nselected

        return error_matrix, nselected_matrix

    def fit(self, X, y, sample_weight=None, cv_splits=None):
        """
        拟合 AP-AFL 模型

        Stage 1: K 折 Profile CV for γ
        Stage 2: Min-MSE Selection of γ
        Stage 3: Full Data Final Calibration with 1-SE Rule
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
        n = X.shape[0]
        p = X.shape[1]
        eps = self.eps

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=_DTYPE)

        # ============================================================
        # Stage 1: K 折 Profile CV
        # ============================================================
        error_matrix, nselected_matrix = self._stage1_profile_cv(X, y, cv_splits)

        # ============================================================
        # Stage 2: Min-MSE Selection of γ
        # ============================================================
        mean_error = np.mean(error_matrix, axis=2)  # (n_gamma, n_alpha)
        mean_nselected = np.mean(nselected_matrix, axis=2)  # (n_gamma, n_alpha)

        # 对每个 gamma，取其全部 alpha 上的最小平均 MSE
        min_MSE_gamma = np.min(mean_error, axis=1)  # (n_gamma,)
        best_gamma_idx = int(np.argmin(min_MSE_gamma))
        self.best_gamma_ = self.gamma_list[best_gamma_idx]
        best_cv_mse = float(min_MSE_gamma[best_gamma_idx])
        self.cv_score_ = -best_cv_mse

        if self.verbose:
            print(f"\n[Stage 2] Min-MSE Gamma Selection:")
            for gi, g in enumerate(self.gamma_list):
                marker = " <-- best" if gi == best_gamma_idx else ""
                print(f"  gamma={g:.2f}: min_MSE={float(min_MSE_gamma[gi]):.6f}{marker}")
            print(f"  selected: gamma={self.best_gamma_}, CV_MSE={best_cv_mse:.6f}")

        # ============================================================
        # Stage 3: 全量数据终极校准
        # ============================================================
        # Stage 3a: 全量标准化
        if self.standardize:
            self.scaler_ = StandardScaler(copy=_COPY_WHEN_POSSIBLE)
            X_for_cv = self.scaler_.fit_transform(X)
        else:
            X_for_cv = X.copy()

        # Stage 3b: 全量 RidgeCV → 最优先验
        ridge_final = RidgeCV(alphas=self.lambda_ridge_list, cv=self.cv)
        ridge_final.fit(X_for_cv, y)
        self.beta_ridge_ = ridge_final.coef_
        self.best_lambda_ridge_ = ridge_final.alpha_

        signs_final = np.sign(self.beta_ridge_)
        signs_final[signs_final == 0] = 1.0
        self.signs_ = signs_final

        # Stage 3c: 使用 best_gamma 计算全局权重
        weights_final = self._compute_weights(self.beta_ridge_, self.best_gamma_)
        self.weights_ = weights_final

        # Stage 3d: 构造全局增广矩阵
        X_con_final, X_dis_final, X_aug_final = self._build_augmented_matrix(
            X_for_cv, signs_final, weights_final
        )

        # Stage 3e: LassoCV 寻找最优 alpha
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

        # Stage 3f: 1-SE 法则选 alpha
        mse_path = lasso_cv.mse_path_  # (n_alphas, n_folds)
        mean_mse = np.mean(mse_path, axis=1)
        # 无偏样本标准差 (ddof=1)
        std_mse = np.std(mse_path, axis=1, ddof=1) / np.sqrt(self.cv)

        min_idx = np.argmin(mean_mse)
        min_mse_value = float(mean_mse[min_idx])
        std_at_min = float(std_mse[min_idx])
        threshold = min_mse_value + std_at_min

        candidates_mask = mean_mse <= threshold
        if np.any(candidates_mask):
            candidate_alphas = alphas[candidates_mask]
            self.best_alpha_ = float(np.max(candidate_alphas))  # 最大（最稀疏）
        else:
            self.best_alpha_ = float(lasso_cv.alpha_)

        if self.verbose:
            print(f"[Stage 3] 1-SE Alpha Selection:")
            print(f"  min_mse={min_mse_value:.6f}, threshold={threshold:.6f}")
            print(f"  selected: alpha={self.best_alpha_:.6f}")

        # Stage 3g: 最终 Lasso 拟合
        lasso_final = Lasso(
            alpha=self.best_alpha_,
            positive=True,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        lasso_final.fit(X_aug_final, y)

        # Stage 3h: 系数还原
        theta_final = lasso_final.coef_  # (2p,)
        theta_con_final = theta_final[:p]
        theta_dis_final = theta_final[p:]

        coef_standardized = self._reconstruct_coef(
            theta_con_final, theta_dis_final, signs_final, weights_final
        )

        # 逆标准化到原始空间
        if self.standardize:
            self.coef_ = coef_standardized / self.scaler_.scale_
            if self.fit_intercept:
                self.intercept_ = np.mean(y) - np.sum(self.coef_ * self.scaler_.mean_)
        else:
            self.coef_ = coef_standardized
            if self.fit_intercept:
                self.intercept_ = np.mean(y) - np.mean(X_for_cv @ coef_standardized)

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
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

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> float:
        from sklearn.metrics import mean_squared_error
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred, sample_weight=sample_weight)

    def get_feature_importance(self) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        return np.abs(self.coef_)


class APAFLClassifier(APAFLRegressor):
    """
    AP-AFL 分类器（二分类）

    将二分类问题转化为连续回归问题处理。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classes_ = None

    def fit(self, X, y, sample_weight=None, cv_splits=None):
        from sklearn.utils.validation import check_X_y

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
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("APAFLClassifier only supports binary classification")

        # 转为连续问题
        y_continuous = (y == self.classes_[1]).astype(_DTYPE)

        return super().fit(X, y_continuous, sample_weight, cv_splits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        z = X @ self.coef_ + self.intercept_
        proba_1 = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return self.classes_[np.argmax(np.column_stack([1 - proba_1, proba_1]), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        z = X @ self.coef_ + self.intercept_
        proba_1 = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - proba_1, proba_1])

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> float:
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
