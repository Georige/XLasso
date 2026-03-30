"""
AdaptiveFlippedLasso API 层
提供用户接口类和交叉验证版本
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted
from .base import BaseAdaptiveFlippedLasso, AdaptiveFlippedLassoRegressor, AdaptiveFlippedLassoClassifier, AdaptiveFlippedLassoCV, AdaptiveFlippedLassoEBIC, AdaptiveFlippedLassoCV_EN


class AdaptiveFlippedLasso(AdaptiveFlippedLassoRegressor):
    """
    AdaptiveFlippedLasso 回归器

    算法特点：
    - 第一阶段使用强 Ridge 回归保留方向信息
    - 基于系数幅度计算归一化自适应权重
    - 特征翻转使所有方向一致
    - 非负 Lasso 施加对称稀疏惩罚
    - 逆重构还原原始空间系数

    参数
    ----
    lambda_ridge : float, default=10.0
        第一阶段 Ridge 正则化强度
    lambda_ : float, optional
        Lasso 正则化强度。若为 None，则自动搜索
    gamma : float, default=1.0
        权重计算的指数衰减参数
    alpha_min_ratio : float, default=1e-4
        自动搜索时 alpha 的最小值比例
    n_alpha : int, default=50
        自动搜索时的 alpha 候选数量
    max_iter : int, default=1000
        Lasso 最大迭代次数
    tol : float, default=1e-4
        Lasso 收敛容忍度
    standardize : bool, default=True
        是否标准化特征
    fit_intercept : bool, default=True
        是否拟合截距
    random_state : int, default=2026
        随机种子
    verbose : bool, default=False
        是否输出详细信息
    """


class AdaptiveFlippedLassoClassifier(AdaptiveFlippedLassoClassifier):
    """
    AdaptiveFlippedLasso 分类器（二分类）

    参数同上
    """
    pass


# AdaptiveFlippedLassoCV 继承自 base.py 中新实现的双层网格搜索版本
# 算法流程：
# 1. 第一阶段：RidgeCV 自动搜索最优 lambda_ridge
# 2. 第二阶段：gamma × alpha 双层网格搜索
#    - 外层：遍历 gamma 候选值
#    - 内层：LassoCV 自动走完 alpha 路径的交叉验证


class AdaptiveFlippedLassoClassifierCV(AdaptiveFlippedLassoClassifier, AdaptiveFlippedLassoCV):
    """
    带交叉验证的 AdaptiveFlippedLasso 分类器（二分类）
    """

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        from sklearn.utils.validation import check_X_y

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("AdaptiveFlippedLassoClassifierCV only supports binary classification")

        y_continuous = (y == self.classes_[1]).astype(np.float64)

        # 显式调用 AdaptiveFlippedLassoCV.fit() 以获得 1-SE CV 逻辑
        # （不能走 super()，因为 MRO 会先到 BaseAdaptiveFlippedLasso）
        return AdaptiveFlippedLassoCV.fit(self, X, y_continuous, sample_weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        z = X @ self.coef_ + self.intercept_
        proba_1 = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - proba_1, proba_1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class AdaptiveFlippedLassoClassifierEBIC(AdaptiveFlippedLassoClassifier, AdaptiveFlippedLassoEBIC):
    """
    带 EBIC 参数选择的 AdaptiveFlippedLasso 分类器（二分类）
    """

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        from sklearn.utils.validation import check_X_y

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("AdaptiveFlippedLassoClassifierEBIC only supports binary classification")

        y_continuous = (y == self.classes_[1]).astype(np.float64)

        # 复用父类 fit 逻辑
        return super().fit(X, y_continuous, sample_weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        z = X @ self.coef_ + self.intercept_
        proba_1 = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - proba_1, proba_1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class AdaptiveFlippedLassoCV_ENClassifier(AdaptiveFlippedLassoClassifier, AdaptiveFlippedLassoCV_EN):
    """
    带交叉验证的 AdaptiveFlippedLassoCV_EN 分类器（二分类）

    使用 Elastic Net 作为第一阶段先验的二分类版本。
    """

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        from sklearn.utils.validation import check_X_y

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("AdaptiveFlippedLassoCV_ENClassifier only supports binary classification")

        y_continuous = (y == self.classes_[1]).astype(np.float64)

        # 显式调用 AdaptiveFlippedLassoCV_EN.fit()
        return AdaptiveFlippedLassoCV_EN.fit(self, X, y_continuous, sample_weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        z = X @ self.coef_ + self.intercept_
        proba_1 = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - proba_1, proba_1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))