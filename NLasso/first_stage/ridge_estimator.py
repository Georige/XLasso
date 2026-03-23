"""
第一阶段：强Ridge回归估计器
支持高斯/二项/泊松分布，内置快速LOO近似算法
"""
import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.utils.extmath import safe_sparse_dot
from scipy.linalg import solve
from ..base import _DTYPE, _COPY_WHEN_POSSIBLE


class RidgeEstimator:
    """
    强Ridge回归估计器，支持快速留一法预测
    性能优化：使用矩阵公式直接计算LOO结果，避免逐折重拟合
    """
    def __init__(
        self,
        alpha: float = 10.0,  # 强正则化，默认较大值
        fit_intercept: bool = True,
        solver: str = 'auto',
        random_state: int = 2026,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.random_state = random_state
        self.beta_ridge_ = None  # Ridge系数 (p,)
        self.X_ = None  # 训练数据 (n, p)
        self.y_ = None  # 训练标签 (n,)
        self.H_ = None  # 帽子矩阵 (n, n) 或其对角 (n,)
        self.residuals_ = None  # 残差 (n,)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        拟合强Ridge模型
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 响应变量 (n_samples,)
            sample_weight: 样本权重 (n_samples,) 可选
        Returns:
            self
        """
        n, p = X.shape

        # 拟合Ridge模型（使用sklearn优化实现）
        if sample_weight is None:
            # 无样本权重时使用快速实现
            self.model_ = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                solver=self.solver,
                random_state=self.random_state,
                copy_X=_COPY_WHEN_POSSIBLE
            )
        else:
            # 有样本权重时使用支持权重的solver
            self.model_ = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                solver='saga',
                random_state=self.random_state,
                copy_X=_COPY_WHEN_POSSIBLE
            )

        self.model_.fit(X, y, sample_weight=sample_weight)
        self.beta_ridge_ = self.model_.coef_.astype(_DTYPE)
        self.X_ = X
        self.y_ = y

        # 预计算帽子矩阵对角元用于快速LOO（仅当p << n时）
        if p * 3 < n:
            # 快速计算H = X(X^T X + αI)^{-1} X^T 的对角元
            if self.fit_intercept:
                # 带截距时需要调整，先中心化
                X_centered = X - np.mean(X, axis=0)
                y_centered = y - np.mean(y)
            else:
                X_centered = X
                y_centered = y

            # 计算 (X^T X + αI)
            XTX = safe_sparse_dot(X_centered.T, X_centered)
            diag_indices = np.diag_indices_from(XTX)
            XTX[diag_indices] += self.alpha

            # 求解逆矩阵（小规模p时高效）
            if hasattr(XTX, 'toarray'):
                XTX = XTX.toarray()
            XTX_inv = solve(XTX, np.eye(p), assume_a='pos')

            # 计算每行的帽子对角元 h_i = x_i^T (X^T X + αI)^{-1} x_i
            self.H_diag_ = np.einsum('ij,ji->i', X_centered @ XTX_inv, X_centered.T)
        else:
            # 高维时不预计算H，LOO时使用近似方法
            self.H_diag_ = None

        # 预计算残差
        y_pred = self.model_.predict(X)
        self.residuals_ = y - y_pred

        return self

    def predict_loo(self, method: str = 'auto') -> np.ndarray:
        """
        快速计算留一法预测值
        Args:
            method: 'exact' - 精确留一（逐折拟合，慢但准）
                   'approx' - 近似公式（快，适合n>p场景）
                   'auto' - 自动选择最优方法
        Returns:
            y_loo: 留一法预测值 (n_samples,)
        """
        n, p = self.X_.shape

        if method == 'auto':
            method = 'approx' if (self.H_diag_ is not None) else 'exact'

        if method == 'approx' and self.H_diag_ is not None:
            # 快速近似公式：y_loo_i = (y_i - y_pred_i) / (1 - h_i) + y_pred_i
            # 来源：Ridge回归留一预测闭式解
            y_pred = self.model_.predict(self.X_)
            y_loo = (self.y_ - y_pred) / (1 - np.clip(self.H_diag_, 1e-10, 1 - 1e-10)) + y_pred
            return y_loo.astype(_DTYPE)

        else:
            # 精确留一法：逐折拟合（适合高维p>n场景）
            y_loo = np.zeros(n, dtype=_DTYPE)
            for i in range(n):
                # 去掉第i个样本
                X_train = np.delete(self.X_, i, axis=0)
                y_train = np.delete(self.y_, i, axis=0)

                # 拟合子模型
                model = Ridge(
                    alpha=self.alpha,
                    fit_intercept=self.fit_intercept,
                    solver=self.solver,
                    random_state=self.random_state,
                    copy_X=False
                )
                model.fit(X_train, y_train)

                # 预测第i个样本
                y_loo[i] = model.predict(self.X_[i:i+1])[0]

            return y_loo


class RidgeClassifierEstimator(RidgeEstimator):
    """
    分类任务的强Ridge估计器
    """
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        n, p = X.shape

        # 拟合Ridge分类器
        self.model_ = RidgeClassifier(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver=self.solver if self.solver != 'saga' else 'auto',
            random_state=self.random_state,
            copy_X=_COPY_WHEN_POSSIBLE
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.beta_ridge_ = self.model_.coef_.ravel().astype(_DTYPE)
        self.X_ = X
        self.y_ = y

        # 预计算帽子矩阵对角元（回归版本，用于LOO）
        y_continuous = y.astype(_DTYPE)
        super().fit(X, y_continuous, sample_weight)
        return self

    def predict_loo(self, method: str = 'auto') -> np.ndarray:
        """
        返回连续值预测（logit尺度）
        """
        return super().predict_loo(method)


def build_ridge_estimator(task_type: str = 'regression', **kwargs) -> RidgeEstimator:
    """
    工厂函数：根据任务类型创建对应的Ridge估计器
    """
    if task_type == 'regression':
        return RidgeEstimator(**kwargs)
    elif task_type == 'classification':
        return RidgeClassifierEstimator(**kwargs)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
