"""
Relaxed Lasso with 1-SE Rule Implementation

Stage 1: LassoCV (1-SE) 进行严格的特征筛选
Stage 2: OLS 对选中的特征进行无偏估计（彻底去偏）
"""
import numpy as np
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import warnings


class RelaxedLassoCV1SE(BaseEstimator, RegressorMixin):
    """
    基于 1-SE 规则的 Relaxed Lasso (Lasso-OLS 极端去偏版)

    Stage 1: LassoCV (1-SE) 进行严格的特征筛选
    Stage 2: OLS 对选中的特征进行无偏估计

    参数:
        cv: 交叉验证折数
        random_state: 随机种子
        eps: alpha 路径长度参数
        n_alphas: alpha 候选数量
        verbose: 是否打印详细信息
    """

    def __init__(
        self,
        cv: int = 5,
        random_state: int = 42,
        eps: float = 1e-3,
        n_alphas: int = 100,
        verbose: bool = False,
        standardize: bool = True,  # Accepted but not used (LassoCV handles it internally)
        fit_intercept: bool = True,  # Accepted but not used (LassoCV handles it internally)
    ):
        self.cv = cv
        self.random_state = random_state
        self.eps = eps
        self.n_alphas = n_alphas
        self.verbose = verbose
        self.standardize = standardize
        self.fit_intercept = fit_intercept

        # 拟合后填充的属性
        self.coef_: np.ndarray = None
        self.intercept_: float = None
        self.alpha_1se_: float = None
        self.support_: np.ndarray = None
        self.n_selected_: int = None

    def fit(self, X, y, cv_splits=None):
        """
        使用 Relaxed Lasso (1-SE) 拟合模型。

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
            self._fit_with_splits(X, y, cv_splits)
        else:
            self._fit_internal_cv(X, y)

        return self

    def _fit_with_splits(self, X, y, cv_splits):
        """使用外部提供的 cv_splits 进行 1-SE 寻优。
        Fold 串行，fold 内部 alpha 并行 (joblib)。
        """
        n_folds = len(cv_splits)

        if self.verbose:
            print(f"Relaxed Lasso Stage 1: 运行 LassoCV 寻找 1-SE 阈值 (external splits, {n_folds} folds)...")

        # Compute alpha grid similar to LassoCV
        lasso_tmp = Lasso(alpha=0.001, max_iter=20000, random_state=self.random_state)
        lasso_tmp.fit(X, y)
        alpha_max = np.abs(lasso_tmp.coef_).max() if np.abs(lasso_tmp.coef_).max() > 0 else 1.0
        alpha_min = alpha_max * self.eps
        alphas = np.linspace(alpha_max, alpha_min, self.n_alphas)

        # mse_path shape: (n_alphas, n_folds)
        mse_path = np.zeros((len(alphas), n_folds))

        # === Fold 串行，alpha 并行 ===
        def _eval_lasso(alpha, X_tr, y_tr, X_va, y_va):
            model = Lasso(alpha=alpha, max_iter=20000, random_state=self.random_state)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            return float(np.mean((y_va - y_pred) ** 2))

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            # 并行：对当前 fold 的所有 alpha 候选求 MSE
            mse_results = Parallel(n_jobs=-1, prefer="threads", verbose=0)(
                delayed(_eval_lasso)(alpha, X_train_s, y_train, X_val_s, y_val)
                for alpha in alphas
            )
            for alpha_idx, mse_val in enumerate(mse_results):
                mse_path[alpha_idx, fold_idx] = mse_val

        # 1-SE logic
        mean_mse = mse_path.mean(axis=1)
        se_mse = mse_path.std(axis=1) / np.sqrt(n_folds)

        best_idx = np.argmin(mean_mse)
        best_mse = mean_mse[best_idx]
        best_se = se_mse[best_idx]

        threshold_1se = best_mse + best_se

        candidates_mask = mean_mse <= threshold_1se
        candidate_indices = np.where(candidates_mask)[0]

        if len(candidate_indices) > 0:
            max_check = min(20, len(candidate_indices))
            sampled_indices = candidate_indices[:max_check]

            n_nonzero_candidates = []
            for idx in sampled_indices:
                alpha = alphas[idx]
                model_tmp = Lasso(alpha=alpha, max_iter=20000, random_state=self.random_state)
                model_tmp.fit(X, y)
                n_nonzero_candidates.append(np.sum(model_tmp.coef_ != 0))

            best_local_idx = np.argmin(n_nonzero_candidates)
            alpha_1se = alphas[sampled_indices[best_local_idx]]
        else:
            alpha_1se = alphas[best_idx]

        self.alpha_1se_ = alpha_1se

        if self.verbose:
            print(f"Stage 1: 全局最小 MSE = {best_mse:.4f}, 1-SE MSE = {threshold_1se:.4f}")

        # Stage 1 final: fit on all data with alpha_1se
        lasso_stage1 = Lasso(alpha=alpha_1se, max_iter=20000, random_state=self.random_state)
        lasso_stage1.fit(X, y)

        support_mask = (lasso_stage1.coef_ != 0)
        self.support_ = support_mask
        self.n_selected_ = np.sum(support_mask)

        if self.verbose:
            print(f"Stage 1 结束. 1-SE Alpha: {alpha_1se:.6f}, 选出特征数: {self.n_selected_}")

        # Stage 2: OLS debiasing
        self.coef_ = np.zeros(X.shape[1])

        if self.n_selected_ == 0:
            if self.verbose:
                print("警告：Stage 1 砍掉了所有特征！退化为常数截距模型。")
            self.intercept_ = np.mean(y)
            return self

        X_active = X[:, support_mask]
        ols = LinearRegression()
        ols.fit(X_active, y)

        self.coef_[support_mask] = ols.coef_
        self.intercept_ = ols.intercept_

        if self.verbose:
            print(f"Stage 2 结束. OLS 系数已填回 {self.n_selected_} 个选中特征")

    def _fit_internal_cv(self, X, y):
        """使用 sklearn LassoCV 的内部 CV (原始行为)。"""
        if self.verbose:
            print("Relaxed Lasso Stage 1: 运行 LassoCV 寻找 1-SE 阈值...")

        precompute = False if X.shape[0] < X.shape[1] else 'auto'
        lasso_cv = LassoCV(
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1,
            max_iter=20000,
            eps=self.eps,
            n_alphas=self.n_alphas,
            precompute=precompute
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_cv.fit(X, y)

        mean_mse = lasso_cv.mse_path_.mean(axis=1)
        se_mse = lasso_cv.mse_path_.std(axis=1) / np.sqrt(self.cv)

        best_idx = np.argmin(mean_mse)
        best_mse = mean_mse[best_idx]
        best_se = se_mse[best_idx]

        threshold_1se = best_mse + best_se

        candidates_mask = mean_mse <= threshold_1se
        candidate_indices = np.where(candidates_mask)[0]

        if len(candidate_indices) > 0:
            max_check = min(20, len(candidate_indices))
            sampled_indices = candidate_indices[:max_check]

            n_nonzero_candidates = []
            for idx in sampled_indices:
                alpha = lasso_cv.alphas_[idx]
                model_tmp = Lasso(alpha=alpha, max_iter=20000, random_state=self.random_state)
                model_tmp.fit(X, y)
                n_nonzero_candidates.append(np.sum(model_tmp.coef_ != 0))

            best_local_idx = np.argmin(n_nonzero_candidates)
            alpha_1se = lasso_cv.alphas_[sampled_indices[best_local_idx]]
        else:
            alpha_1se = lasso_cv.alphas_[best_idx]

        self.alpha_1se_ = alpha_1se

        if self.verbose:
            print(f"Stage 1: 全局最小 MSE = {best_mse:.4f}, 1-SE MSE = {threshold_1se:.4f}")

        lasso_stage1 = Lasso(
            alpha=alpha_1se,
            max_iter=20000,
            random_state=self.random_state
        )
        lasso_stage1.fit(X, y)

        support_mask = (lasso_stage1.coef_ != 0)
        self.support_ = support_mask
        self.n_selected_ = np.sum(support_mask)

        if self.verbose:
            print(f"Stage 1 结束. 1-SE Alpha: {alpha_1se:.6f}, 选出特征数: {self.n_selected_}")

        self.coef_ = np.zeros(X.shape[1])

        if self.n_selected_ == 0:
            if self.verbose:
                print("警告：Stage 1 砍掉了所有特征！退化为常数截距模型。")
            self.intercept_ = np.mean(y)
            return self

        X_active = X[:, support_mask]
        ols = LinearRegression()
        ols.fit(X_active, y)

        self.coef_[support_mask] = ols.coef_
        self.intercept_ = ols.intercept_

        if self.verbose:
            print(f"Stage 2 结束. OLS 系数已填回 {self.n_selected_} 个选中特征")

    def predict(self, X):
        """使用拟合的模型进行预测。"""
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """计算 R² 分数。"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
