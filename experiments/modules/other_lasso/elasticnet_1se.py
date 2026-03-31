"""
ElasticNet with 1-SE Rule Implementation

带有严格 1-SE Rule 约束的 ElasticNet 模型，用于稀疏特征选择。
"""
import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import warnings


class ElasticNet1SE(BaseEstimator, RegressorMixin):
    """
    ElasticNet with 1-SE Rule for sparse feature selection.

    参数:
        cv_folds: 交叉验证折数
        l1_ratios: l1_ratio 候选列表
        max_iter: 最大迭代次数
        random_state: 随机种子
        verbose: 是否打印详细信息
    """

    def __init__(
        self,
        cv_folds: int = 5,
        l1_ratios=None,
        max_iter: int = 5000,
        random_state: int = 42,
        verbose: bool = True,
        standardize: bool = True,  # Accepted but not used (ElasticNetCV handles it internally)
        fit_intercept: bool = True,  # Accepted but not used (ElasticNetCV handles it internally)
    ):
        self.cv_folds = cv_folds
        self.l1_ratios = l1_ratios or [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.standardize = standardize
        self.fit_intercept = fit_intercept

        # 拟合后填充的属性
        self.coef_: np.ndarray = None
        self.intercept_: float = None
        self.alpha_: float = None
        self.l1_ratio_: float = None
        self.n_iter_: int = None

    def fit(self, X, y, cv_splits=None):
        """
        使用 1-SE Rule 拟合 ElasticNet 模型。

        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标向量 (n_samples,)
            cv_splits: list of tuples, optional
                Pre-generated CV splits (list of (train_idx, val_idx) tuples).
                If provided, uses these splits instead of creating new KFold.

        返回:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if cv_splits is not None:
            # === 使用外部提供的 CV splits ===
            self._fit_with_splits(X, y, cv_splits)
        else:
            # === 使用内部 CV (原始行为) ===
            self._fit_internal_cv(X, y)

        return self

    def _fit_with_splits(self, X, y, cv_splits):
        """使用外部提供的 cv_splits 进行 1-SE 寻优。"""
        n_folds = len(cv_splits)
        n_l1 = len(self.l1_ratios)

        if self.verbose:
            print(f"正在进行 ElasticNet 2D 交叉验证搜索 (external splits, {n_folds} folds)...")

        # Compute alpha grid using lasso_path (same logic as ElasticNetCV)
        # First, fit a Lasso to get alpha_max and alpha_min
        lasso_tmp = ElasticNet(l1_ratio=1.0, alpha=0.001, max_iter=self.max_iter, random_state=self.random_state)
        lasso_tmp.fit(X, y)
        alpha_max = np.abs(lasso_tmp.coef_).max() if np.abs(lasso_tmp.coef_).max() > 0 else 1.0
        alpha_min = alpha_max * 0.0001
        alphas = np.linspace(alpha_max, alpha_min, 100)

        # mse_path shape: (n_l1, n_alphas, n_folds)
        mse_path = np.zeros((n_l1, len(alphas), n_folds))

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Standardize within fold (only if needed)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            # === Fold 串行，alpha 并行 ===
            def _eval_enet(l1_ratio, alpha, X_tr, y_tr, X_va, y_va):
                model = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    fit_intercept=True
                )
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_va)
                return float(np.mean((y_va - y_pred) ** 2))

            # Build (l1_idx, alpha_idx, l1_ratio, alpha) task list
            tasks = [
                (l1_idx, alpha_idx, l1_ratio, alpha)
                for l1_idx, l1_ratio in enumerate(self.l1_ratios)
                for alpha_idx, alpha in enumerate(alphas)
            ]

            mse_results = Parallel(n_jobs=-1, prefer="threads", verbose=0)(
                delayed(_eval_enet)(l1_ratio, alpha, X_train_s, y_train, X_val_s, y_val)
                for (_, _, l1_ratio, alpha) in tasks
            )

            for (l1_idx, alpha_idx, _, _), mse_val in zip(tasks, mse_results):
                mse_path[l1_idx, alpha_idx, fold_idx] = mse_val

        # Compute mean and SE across folds
        mean_mse_path = mse_path.mean(axis=-1)
        std_mse_path = mse_path.std(axis=-1)
        se_mse_path = std_mse_path / np.sqrt(n_folds)

        # Find global optimum
        best_l1_idx, best_alpha_idx = np.unravel_index(
            np.argmin(mean_mse_path), mean_mse_path.shape
        )
        best_mse = mean_mse_path[best_l1_idx, best_alpha_idx]
        best_se = se_mse_path[best_l1_idx, best_alpha_idx]
        best_l1_ratio = self.l1_ratios[best_l1_idx]

        # 1-SE rule
        threshold_1se = best_mse + best_se
        mse_for_best_l1 = mean_mse_path[best_l1_idx]
        alphas_for_best_l1 = alphas

        candidates_mask = mse_for_best_l1 <= threshold_1se
        candidate_indices = np.where(candidates_mask)[0]

        if len(candidate_indices) > 0:
            n_nonzero_list = []
            for idx in candidate_indices:
                alpha = alphas_for_best_l1[idx]
                model_tmp = ElasticNet(
                    alpha=alpha,
                    l1_ratio=best_l1_ratio,
                    max_iter=self.max_iter,
                    random_state=self.random_state
                )
                model_tmp.fit(X, y)
                n_nonzero = np.sum(model_tmp.coef_ != 0)
                n_nonzero_list.append(n_nonzero)

            best_candidate_idx = candidate_indices[np.argmin(n_nonzero_list)]
            alpha_1se = alphas_for_best_l1[best_candidate_idx]
        else:
            alpha_1se = alphas_for_best_l1[best_alpha_idx]

        if self.verbose:
            print(f"全局最小 MSE 对应的 alpha: {alphas[best_alpha_idx]:.6f}")
            print(f"严格 1-SE Rule 选出的 alpha: {alpha_1se:.6f}")
            print(f"锁定的最佳 l1_ratio: {best_l1_ratio}")

        # Final refit on all data
        final_model = ElasticNet(
            alpha=alpha_1se,
            l1_ratio=best_l1_ratio,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        final_model.fit(X, y)

        self.coef_ = final_model.coef_
        self.intercept_ = final_model.intercept_
        self.alpha_ = alpha_1se
        self.l1_ratio_ = best_l1_ratio
        self.n_iter_ = final_model.n_iter_

    def _fit_internal_cv(self, X, y):
        """使用 sklearn ElasticNetCV 的内部 CV (原始行为)。"""
        if self.verbose:
            print("正在进行 ElasticNet 2D 交叉验证搜索...")

        enet_cv = ElasticNetCV(
            l1_ratio=self.l1_ratios,
            cv=self.cv_folds,
            random_state=self.random_state,
            n_jobs=-1,
            max_iter=self.max_iter
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enet_cv.fit(X, y)

        # --- 下面进入 1-SE 寻优逻辑 ---
        mean_mse_path = enet_cv.mse_path_.mean(axis=-1)
        std_mse_path = enet_cv.mse_path_.std(axis=-1)
        se_mse_path = std_mse_path / np.sqrt(self.cv_folds)

        best_l1_idx, best_alpha_idx = np.unravel_index(
            np.argmin(mean_mse_path), mean_mse_path.shape
        )

        best_mse = mean_mse_path[best_l1_idx, best_alpha_idx]
        best_se = se_mse_path[best_l1_idx, best_alpha_idx]

        if isinstance(enet_cv.l1_ratio_, np.ndarray):
            best_l1_ratio = enet_cv.l1_ratio_[best_l1_idx]
        else:
            best_l1_ratio = self.l1_ratios[best_l1_idx]

        alphas_for_best_l1 = enet_cv.alphas_[best_l1_idx]
        mse_for_best_l1 = mean_mse_path[best_l1_idx]

        threshold_1se = best_mse + best_se

        candidates_mask = mse_for_best_l1 <= threshold_1se
        candidate_indices = np.where(candidates_mask)[0]

        if len(candidate_indices) > 0:
            n_nonzero_list = []
            for idx in candidate_indices:
                alpha = alphas_for_best_l1[idx]
                model_tmp = ElasticNet(
                    alpha=alpha,
                    l1_ratio=best_l1_ratio,
                    max_iter=self.max_iter,
                    random_state=self.random_state
                )
                model_tmp.fit(X, y)
                n_nonzero = np.sum(model_tmp.coef_ != 0)
                n_nonzero_list.append(n_nonzero)

            best_candidate_idx = candidate_indices[np.argmin(n_nonzero_list)]
            alpha_1se = alphas_for_best_l1[best_candidate_idx]
        else:
            alpha_1se = alphas_for_best_l1[best_alpha_idx]

        if self.verbose:
            print(f"全局最小 MSE 对应的 alpha: {enet_cv.alphas_[best_l1_idx, best_alpha_idx]:.6f}")
            print(f"严格 1-SE Rule 选出的 alpha: {alpha_1se:.6f} (模型更精简)")
            print(f"锁定的最佳 l1_ratio: {best_l1_ratio}")

        final_model = ElasticNet(
            alpha=alpha_1se,
            l1_ratio=best_l1_ratio,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        final_model.fit(X, y)

        self.coef_ = final_model.coef_
        self.intercept_ = final_model.intercept_
        self.alpha_ = alpha_1se
        self.l1_ratio_ = best_l1_ratio
        self.n_iter_ = final_model.n_iter_

    def predict(self, X):
        """使用拟合的模型进行预测。"""
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """计算 R² 分数。"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_elasticnet_1se(X, y, cv_folds=5, l1_ratios=None, max_iter=5000, random_state=42, verbose=True):
    """
    带有严格 1-SE Rule 约束的 ElasticNet 模型。

    参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标向量 (n_samples,)
        cv_folds: 交叉验证折数
        l1_ratios: l1_ratio 候选列表
        max_iter: 最大迭代次数
        random_state: 随机种子
        verbose: 是否打印详细信息

    返回:
        tuple: (fitted_model, best_l1_ratio, alpha_1se)
    """
    model = ElasticNet1SE(
        cv_folds=cv_folds,
        l1_ratios=l1_ratios,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose
    )
    model.fit(X, y)
    return model, model.l1_ratio_, model.alpha_
