"""
API层：NLasso交叉验证自动调参版本
自动搜索最优超参数组合
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import get_scorer
from .nlasso import NLasso, NLassoClassifier
from ..base import _DTYPE


class NLassoCV(BaseEstimator, RegressorMixin):
    """
    NLasso 交叉验证版本（回归）
    自动搜索最优超参数组合，支持多参数并行调参
    """
    def __init__(
        self,
        # 超参数搜索空间
        param_grid: dict = None,
        # CV配置
        cv: int = 5,
        scoring: str = 'r2',
        n_jobs: int = -1,
        refit: bool = True,
        verbose: bool = False,
        random_state: int = 2026,
        # 其他固定参数
        **kwargs
    ):
        # 默认超参数搜索空间（paper推荐范围）
        if param_grid is None:
            self.param_grid = {
                'lambda_ridge': [1.0, 5.0, 10.0, 20.0, 50.0],
                'gamma': [0.1, 0.3, 0.5, 0.7, 1.0],
                's': [0.1, 0.5, 1.0, 2.0, 5.0],
                'group_threshold': [0.6, 0.7, 0.8, 0.9],
            }
        else:
            self.param_grid = param_grid

        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.random_state = random_state
        self.fixed_params = kwargs

        # 拟合后属性
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None, **fit_params):
        """
        拟合模型并执行交叉验证调参
        """
        # 创建基础估计器
        base_estimator = NLasso(
            random_state=self.random_state,
            verbose=self.verbose,
            **self.fixed_params
        )

        # 创建交叉验证分割器
        cv_splitter = KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

        # 网格搜索
        self.grid_search_ = GridSearchCV(
            estimator=base_estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=cv_splitter,
            refit=self.refit,
            verbose=self.verbose,
            error_score=np.nan
        )

        self.grid_search_.fit(X, y, groups=groups, **fit_params)

        # 保存结果
        self.best_estimator_ = self.grid_search_.best_estimator_
        self.best_params_ = self.grid_search_.best_params_
        self.best_score_ = self.grid_search_.best_score_
        self.cv_results_ = self.grid_search_.cv_results_
        self.is_fitted_ = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> float:
        """评分"""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.score(X, y, sample_weight=sample_weight)

    @property
    def coef_(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.coef_

    @property
    def intercept_(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.intercept_

    @property
    def groups_(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.groups_


class NLassoClassifierCV(BaseEstimator, ClassifierMixin):
    """
    NLasso 交叉验证版本（分类）
    自动搜索最优超参数组合，支持多参数并行调参
    """
    def __init__(
        self,
        # 超参数搜索空间
        param_grid: dict = None,
        # CV配置
        cv: int = 5,
        scoring: str = 'accuracy',
        n_jobs: int = -1,
        refit: bool = True,
        verbose: bool = False,
        random_state: int = 2026,
        # 其他固定参数
        **kwargs
    ):
        # 默认超参数搜索空间（paper推荐范围）
        if param_grid is None:
            self.param_grid = {
                'lambda_ridge': [1.0, 5.0, 10.0, 20.0],
                'gamma': [0.2, 0.3, 0.5, 0.7],
                's': [0.5, 1.0, 2.0],
                'group_threshold': [0.7, 0.8, 0.9],
            }
        else:
            self.param_grid = param_grid

        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.random_state = random_state
        self.fixed_params = kwargs

        # 拟合后属性
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.is_fitted_ = False
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None, **fit_params):
        """
        拟合模型并执行交叉验证调参
        """
        self.classes_ = np.unique(y)

        # 创建基础估计器
        base_estimator = NLassoClassifier(
            random_state=self.random_state,
            verbose=self.verbose,
            **self.fixed_params
        )

        # 创建分层交叉验证分割器
        cv_splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

        # 网格搜索
        self.grid_search_ = GridSearchCV(
            estimator=base_estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=cv_splitter,
            refit=self.refit,
            verbose=self.verbose,
            error_score=np.nan
        )

        self.grid_search_.fit(X, y, groups=groups, **fit_params)

        # 保存结果
        self.best_estimator_ = self.grid_search_.best_estimator_
        self.best_params_ = self.grid_search_.best_params_
        self.best_score_ = self.grid_search_.best_score_
        self.cv_results_ = self.grid_search_.cv_results_
        self.is_fitted_ = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> float:
        """评分"""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.score(X, y, sample_weight=sample_weight)

    @property
    def coef_(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.coef_

    @property
    def intercept_(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.intercept_

    @property
    def groups_(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.groups_
