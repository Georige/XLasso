"""
ElasticNet with 1-SE Rule Implementation

带有严格 1-SE Rule 约束的 ElasticNet 模型，用于稀疏特征选择。
"""
import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet
import warnings


class ElasticNet1SE:
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

    def fit(self, X, y):
        """
        使用 1-SE Rule 拟合 ElasticNet 模型。

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

        # 1. 调用 sklearn 的底层高效 CV
        if self.verbose:
            print("正在进行 ElasticNet 2D 交叉验证搜索...")

        enet_cv = ElasticNetCV(
            l1_ratio=self.l1_ratios,
            cv=self.cv_folds,
            random_state=self.random_state,
            n_jobs=-1,
            max_iter=self.max_iter
        )

        # 忽略高维下偶尔的收敛警告，保持输出清爽
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enet_cv.fit(X, y)

        # --- 下面进入 1-SE 寻优逻辑 ---

        # mse_path_ 的形状是 (n_l1_ratio, n_alphas, n_folds)
        mean_mse_path = enet_cv.mse_path_.mean(axis=-1)
        std_mse_path = enet_cv.mse_path_.std(axis=-1)
        # 将标准差(SD)转为标准误(SE)
        se_mse_path = std_mse_path / np.sqrt(self.cv_folds)

        # 2. 找到全局的绝对最优 (Min-MSE)
        best_l1_idx, best_alpha_idx = np.unravel_index(
            np.argmin(mean_mse_path), mean_mse_path.shape
        )

        best_mse = mean_mse_path[best_l1_idx, best_alpha_idx]
        best_se = se_mse_path[best_l1_idx, best_alpha_idx]

        # 3. 锁定最佳的 l1_ratio
        if isinstance(enet_cv.l1_ratio_, np.ndarray):
            best_l1_ratio = enet_cv.l1_ratio_[best_l1_idx]
        else:
            best_l1_ratio = self.l1_ratios[best_l1_idx]

        # 4. 提取这条最优 l1_ratio 路径上的所有 alpha 和 mse
        alphas_for_best_l1 = enet_cv.alphas_[best_l1_idx]
        mse_for_best_l1 = mean_mse_path[best_l1_idx]

        # 5. 计算 1-SE 阈值上限 (误差越小越好，所以容忍度是加上 1 个 SE)
        threshold_1se = best_mse + best_se

        # 6. 找出及格的 alpha，并选出非零系数最少的
        candidates_mask = mse_for_best_l1 <= threshold_1se
        candidate_indices = np.where(candidates_mask)[0]

        if len(candidate_indices) > 0:
            # 对每个候选 alpha 实际 fit 一次，数非零系数，选最稀疏的
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

        # 7. 拿着选出的参数，在全量数据上终极重拟合 (Final Refit)
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
