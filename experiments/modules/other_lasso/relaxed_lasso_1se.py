"""
Relaxed Lasso with 1-SE Rule Implementation

Stage 1: LassoCV (1-SE) 进行严格的特征筛选
Stage 2: OLS 对选中的特征进行无偏估计（彻底去偏）
"""
import numpy as np
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
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
        verbose: bool = True,
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

    def fit(self, X, y):
        """
        使用 Relaxed Lasso (1-SE) 拟合模型。

        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标向量 (n_samples,)

        返回:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # ==========================================
        # Stage 1: 严格的 Lasso 1-SE 特征筛选
        # ==========================================
        if self.verbose:
            print("Relaxed Lasso Stage 1: 运行 LassoCV 寻找 1-SE 阈值...")

        lasso_cv = LassoCV(
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1,
            max_iter=5000,
            eps=self.eps,
            n_alphas=self.n_alphas
        )

        # 忽略收敛警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_cv.fit(X, y)

        # --- 严谨的 1-SE 逻辑计算 ---
        # mse_path_ 的 shape 为 (n_alphas, n_folds)
        mean_mse = lasso_cv.mse_path_.mean(axis=1)
        # 必须除以 sqrt(cv) 将 SD 转为 SE!
        se_mse = lasso_cv.mse_path_.std(axis=1) / np.sqrt(self.cv)

        best_idx = np.argmin(mean_mse)
        best_mse = mean_mse[best_idx]
        best_se = se_mse[best_idx]

        # 阈值：容忍更大的误差 (加上 1 个 SE)
        threshold_1se = best_mse + best_se

        # 找出及格的 alpha，并选出非零系数最少的
        candidates_mask = mean_mse <= threshold_1se
        candidate_indices = np.where(candidates_mask)[0]

        if len(candidate_indices) > 0:
            # 对每个候选 alpha 实际 fit 一次，数非零系数，选最稀疏的
            n_nonzero_list = []
            for idx in candidate_indices:
                alpha = lasso_cv.alphas_[idx]
                model_tmp = Lasso(
                    alpha=alpha,
                    max_iter=5000,
                    random_state=self.random_state
                )
                model_tmp.fit(X, y)
                n_nonzero = np.sum(model_tmp.coef_ != 0)
                n_nonzero_list.append(n_nonzero)

            best_candidate_idx = candidate_indices[np.argmin(n_nonzero_list)]
            alpha_1se = lasso_cv.alphas_[best_candidate_idx]
        else:
            alpha_1se = lasso_cv.alphas_[best_idx]
        self.alpha_1se_ = alpha_1se

        if self.verbose:
            print(f"Stage 1: 全局最小 MSE = {best_mse:.4f}, 1-SE MSE = {threshold_1se:.4f}")

        # 带着 1-SE 选出的最强 alpha，在全量训练集上确定最终的"非零支撑集" (Support)
        lasso_stage1 = Lasso(
            alpha=alpha_1se,
            max_iter=5000,
            random_state=self.random_state
        )
        lasso_stage1.fit(X, y)

        # 提取非零特征的布尔索引
        support_mask = (lasso_stage1.coef_ != 0)
        self.support_ = support_mask
        self.n_selected_ = np.sum(support_mask)

        if self.verbose:
            print(f"Stage 1 结束. 1-SE Alpha: {alpha_1se:.6f}, 选出特征数: {self.n_selected_}")

        # ==========================================
        # Stage 2: 无惩罚的 OLS 彻底去偏 (Debiasing)
        # ==========================================
        # 初始化最终的系数数组 (全 0)
        self.coef_ = np.zeros(n_features)

        # 极端边缘情况防崩溃：如果 Lasso 极其激进，把所有特征都砍成 0 了
        if self.n_selected_ == 0:
            if self.verbose:
                print("警告：Stage 1 砍掉了所有特征！退化为常数截距模型。")
            self.intercept_ = np.mean(y)
            return self

        # 如果选出了特征，只在这些特征上跑普通的线性回归 (无任何收缩偏差)
        X_active = X[:, support_mask]
        ols = LinearRegression()
        ols.fit(X_active, y)

        # 把 OLS 算出来的无偏系数，填回原始的 p 维数组中对应的位置
        self.coef_[support_mask] = ols.coef_
        self.intercept_ = ols.intercept_

        if self.verbose:
            print(f"Stage 2 结束. OLS 系数已填回 {self.n_selected_} 个选中特征")

        return self

    def predict(self, X):
        """使用拟合的模型进行预测。"""
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """计算 R² 分数。"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
