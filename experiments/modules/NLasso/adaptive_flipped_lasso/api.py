"""
AdaptiveFlippedLasso API 层
提供用户接口类和交叉验证版本
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted
from .base import BaseAdaptiveFlippedLasso, AdaptiveFlippedLassoRegressor, AdaptiveFlippedLassoClassifier


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


class AdaptiveFlippedLassoCV:
    """
    带交叉验证的 AdaptiveFlippedLasso

    自动搜索最优 lambda_ 参数
    """

    def __init__(
        self,
        lambda_ridge: float = 10.0,
        lambda_min_ratio: float = 1e-4,
        n_lambda: int = 50,
        cv: int = 5,
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
        self.lambda_ridge = lambda_ridge
        self.lambda_min_ratio = lambda_min_ratio
        self.n_lambda = n_lambda
        self.cv = cv
        self.gamma = gamma
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alpha = n_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> dict:
        return {k: v for k, v in self.__dict__.items() if k.endswith('_') or k in [
            'lambda_ridge', 'lambda_min_ratio', 'n_lambda', 'cv', 'gamma',
            'alpha_min_ratio', 'n_alpha', 'max_iter', 'tol', 'standardize',
            'fit_intercept', 'random_state', 'verbose'
        ]}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        交叉验证拟合
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.utils.validation import check_X_y

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], ensure_2d=True)
        self.n_features_in_ = X.shape[1]

        # 标准化
        if self.standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # 计算 lambda 搜索路径
        lambda_max = np.max(np.abs(X.T @ y)) / len(y)
        lambda_min = lambda_max * self.lambda_min_ratio
        lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), self.n_lambda)[::-1]

        # K-Fold CV
        kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        cv_scores = np.zeros(len(lambdas))
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if sample_weight is not None:
                sw_train = sample_weight[train_idx]
            else:
                sw_train = None

            fold_scores = []
            for i, lam in enumerate(lambdas):
                model = AdaptiveFlippedLasso(
                    lambda_ridge=self.lambda_ridge,
                    lambda_=lam,
                    gamma=self.gamma,
                    alpha_min_ratio=self.alpha_min_ratio,
                    n_alpha=self.n_alpha,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    standardize=False,
                    fit_intercept=self.fit_intercept,
                    random_state=self.random_state,
                    verbose=False,
                )
                model.fit(X_train, y_train, sw_train)
                cv_scores[i] += model.score(X_test, y_test)

            if self.verbose:
                print(f"[CV] Fold {fold_idx + 1}/{self.cv} completed")

        cv_scores /= self.cv

        # 选择最优 lambda
        best_idx = np.argmax(cv_scores)
        self.best_lambda_ = lambdas[best_idx]
        self.cv_scores_ = cv_scores

        if self.verbose:
            print(f"[CV] Best lambda={self.best_lambda_:.6f}, CV score={cv_scores[best_idx]:.4f}")

        # 用最优 lambda 在全部数据上拟合
        self.best_model_ = AdaptiveFlippedLasso(
            lambda_ridge=self.lambda_ridge,
            lambda_=self.best_lambda_,
            gamma=self.gamma,
            alpha_min_ratio=self.alpha_min_ratio,
            n_alpha=self.n_alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            standardize=False,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.best_model_.fit(X, y, sample_weight)

        # 复制属性
        self.coef_ = self.best_model_.coef_
        self.intercept_ = self.best_model_.intercept_
        self.scaler_ = scaler if self.standardize else None
        self.is_fitted_ = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        return self.best_model_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        return self.best_model_.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        return np.abs(self.coef_)


class AdaptiveFlippedLassoClassifierCV(AdaptiveFlippedLassoCV):
    """
    带交叉验证的 AdaptiveFlippedLasso 分类器（二分类）
    """

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        from sklearn.preprocessing import StandardScaler
        from sklearn.utils.validation import check_X_y

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("AdaptiveFlippedLassoClassifierCV only supports binary classification")

        y_continuous = (y == self.classes_[1]).astype(np.float64)

        # 标准化
        if self.standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # 计算 lambda 搜索路径
        lambda_max = np.max(np.abs(X.T @ y_continuous)) / len(y_continuous)
        lambda_min = lambda_max * self.lambda_min_ratio
        lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), self.n_lambda)[::-1]

        # K-Fold CV
        kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        cv_scores = np.zeros(len(lambdas))
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_continuous[train_idx], y_continuous[test_idx]

            fold_scores = []
            for i, lam in enumerate(lambdas):
                model = AdaptiveFlippedLasso(
                    lambda_ridge=self.lambda_ridge,
                    lambda_=lam,
                    gamma=self.gamma,
                    alpha_min_ratio=self.alpha_min_ratio,
                    n_alpha=self.n_alpha,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    standardize=False,
                    fit_intercept=self.fit_intercept,
                    random_state=self.random_state,
                    verbose=False,
                )
                model.fit(X_train, y_train)
                cv_scores[i] += model.score(X_test, y_test)

        cv_scores /= self.cv

        best_idx = np.argmax(cv_scores)
        self.best_lambda_ = lambdas[best_idx]
        self.cv_scores_ = cv_scores

        # 全量拟合
        self.best_model_ = AdaptiveFlippedLasso(
            lambda_ridge=self.lambda_ridge,
            lambda_=self.best_lambda_,
            gamma=self.gamma,
            alpha_min_ratio=self.alpha_min_ratio,
            n_alpha=self.n_alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            standardize=False,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.best_model_.fit(X, y_continuous, sample_weight)

        self.coef_ = self.best_model_.coef_
        self.intercept_ = self.best_model_.intercept_
        self.scaler_ = scaler if self.standardize else None
        self.is_fitted_ = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        return self.best_model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        return self.best_model_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
