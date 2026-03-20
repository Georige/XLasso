"""
Adaptive Sparse Group Lasso 自适应稀疏组Lasso实现
参考: https://github.com/alvaromc317/adaptive-sparse-group-lasso-paper-simulations
同时支持组级稀疏和组内特征级稀疏，带自适应权重
"""
import numpy as np
from .base import BaseLasso
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

class AdaptiveSparseGroupLasso(BaseLasso):
    """
    自适应稀疏组Lasso回归/分类
    同时施加特征级L1惩罚和组级L2惩罚，两者都带自适应权重，
    可以同时实现组选择和组内特征选择，具有oracle性质

    参考:
    - Simon et al. (2013). "A sparse-group lasso."
    - Adaptive Sparse Group Lasso 论文实现
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, groups=None, fit_intercept=True,
                 standardize=True, max_iter=1000, tol=1e-4, group_weights=None,
                 feature_weights=None, gamma=1.0, family="gaussian"):
        """
        初始化参数
        Parameters:
            alpha: 全局正则化强度
            l1_ratio: L1惩罚占比，0<=l1_ratio<=1
                l1_ratio=0: 退化为Group Lasso
                l1_ratio=1: 退化为Adaptive Lasso
            groups: 分组列表，每个元素是组内特征的索引列表
            fit_intercept: 是否拟合截距
            standardize: 是否标准化特征
            max_iter: 最大迭代次数
            tol: 收敛阈值
            group_weights: 各组的惩罚权重，默认sqrt(组大小)
            feature_weights: 各特征的惩罚权重，默认基于初始OLS估计计算
            gamma: 自适应权重指数，默认1.0
            family: 模型家族，支持"gaussian"(回归)或"binomial"(二分类)
        """
        super().__init__(alpha, fit_intercept, standardize, max_iter, tol, family)
        self.l1_ratio = l1_ratio
        self.groups = groups
        self.group_weights = group_weights
        self.feature_weights = feature_weights
        self.gamma = gamma

        if not CVXPY_AVAILABLE:
            raise ImportError("AdaptiveSparseGroupLasso需要cvxpy库，请运行: pip install cvxpy")

        if not (0 <= l1_ratio <= 1):
            raise ValueError("l1_ratio must be between 0 and 1")

    def fit(self, X, y, sample_weight=None):
        """
        拟合自适应稀疏组Lasso模型
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        n_samples, n_features = X.shape

        # 默认分组：每个特征单独为一组
        if self.groups is None:
            self.groups = [[i] for i in range(n_features)]

        # 预处理
        X_processed, y_processed = self._preprocess(X, y)
        self.X_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y)

        # 计算自适应权重（如果没有提供）
        if self.feature_weights is None:
            # 第一步：拟合初始模型得到系数估计
            if self.family.lower() == "gaussian":
                if X.shape[0] > X.shape[1]:
                    from sklearn.linear_model import LinearRegression
                    ols = LinearRegression(fit_intercept=False)
                    ols.fit(X_processed, y_processed, sample_weight=sample_weight)
                    beta_initial = ols.coef_
                else:
                    from sklearn.linear_model import Ridge
                    ridge = Ridge(alpha=0.1, fit_intercept=False, max_iter=self.max_iter)
                    ridge.fit(X_processed, y_processed, sample_weight=sample_weight)
                    beta_initial = ridge.coef_
            else:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(penalty='l2', C=1.0, fit_intercept=False,
                                      max_iter=self.max_iter, solver='liblinear')
                lr.fit(X_processed, y_processed, sample_weight=sample_weight)
                beta_initial = lr.coef_[0]

            # 计算特征自适应权重
            eps = 1e-10
            self.feature_weights = 1.0 / (np.abs(beta_initial) + eps) ** self.gamma
            self.beta_initial_ = beta_initial

        # 定义优化变量
        beta = cp.Variable(n_features)
        intercept = cp.Variable() if self.fit_intercept else 0.0

        # 损失函数
        if self.family.lower() == "gaussian":
            if sample_weight is None:
                loss = cp.sum_squares(y_processed - (X_processed @ beta + intercept)) / (2 * n_samples)
            else:
                loss = cp.sum(cp.multiply(sample_weight, cp.square(y_processed - (X_processed @ beta + intercept)))) / (2 * n_samples)
        else:
            # 逻辑回归损失
            logits = X_processed @ beta + intercept
            if sample_weight is None:
                loss = cp.sum(cp.logistic(-cp.multiply(2 * y_processed - 1, logits))) / n_samples
            else:
                loss = cp.sum(cp.multiply(sample_weight, cp.logistic(-cp.multiply(2 * y_processed - 1, logits)))) / n_samples

        # 自适应稀疏组惩罚
        l1_penalty = 0.0
        group_penalty = 0.0

        for g_idx, group in enumerate(self.groups):
            group_size = len(group)
            # 组权重
            if self.group_weights is None:
                w_g = np.sqrt(group_size)
            else:
                w_g = self.group_weights[g_idx]

            # 特征级L1惩罚（自适应权重）
            for j in group:
                l1_penalty += self.feature_weights[j] * cp.abs(beta[j])

            # 组级L2惩罚
            group_penalty += w_g * cp.norm2(beta[group])

        penalty = self.alpha * (self.l1_ratio * l1_penalty + (1 - self.l1_ratio) * group_penalty)

        # 目标函数
        objective = cp.Minimize(loss + penalty)

        # 求解
        problem = cp.Problem(objective)
        problem.solve(
            solver=cp.ECOS,
            max_iters=self.max_iter,
            abstol=self.tol,
            verbose=False
        )

        if problem.status != 'optimal':
            import warnings
            warnings.warn(f"优化未完全收敛，状态: {problem.status}")

        coef_ = beta.value
        intercept_ = intercept.value if self.fit_intercept else 0.0

        # 后处理，还原到原始尺度
        self._postprocess(coef_, intercept_)

        # 保存组信息
        self.group_indices_ = self.groups
        self.n_groups_ = len(self.groups)

        return self

    @staticmethod
    def group_features_by_correlation(X, corr_threshold=0.7, max_group_size=20):
        """
        基于相关性自动分组，和GroupLasso一致
        """
        from .group_lasso import GroupLasso
        return GroupLasso.group_features_by_correlation(X, corr_threshold, max_group_size)


class AdaptiveSparseGroupLassoCV(AdaptiveSparseGroupLasso):
    """
    带交叉验证的自适应稀疏组Lasso，自动选择最优alpha和l1_ratio参数
    """
    def __init__(self, alphas=None, l1_ratios=[0.1, 0.5, 0.9], groups=None,
                 fit_intercept=True, standardize=True, max_iter=1000, tol=1e-4,
                 group_weights=None, gamma=1.0, family="gaussian", cv=5,
                 scoring=None, n_jobs=None):
        """
        初始化参数
        Parameters:
            alphas: 待搜索的alpha值列表，默认自动生成
            l1_ratios: 待搜索的l1_ratio值列表，默认[0.1,0.5,0.9]
            groups: 分组列表
            fit_intercept: 是否拟合截距
            standardize: 是否标准化特征
            max_iter: 最大迭代次数
            tol: 收敛阈值
            group_weights: 各组的惩罚权重
            gamma: 自适应权重指数
            family: 模型家族
            cv: 交叉验证折数
            scoring: 评价指标
            n_jobs: 并行作业数
        """
        # 先传一个临时的l1_ratio=0.5，后面会在网格搜索中替换
        super().__init__(
            alpha=None, l1_ratio=0.5, groups=groups, fit_intercept=fit_intercept,
            standardize=standardize, max_iter=max_iter, tol=tol,
            group_weights=group_weights, gamma=gamma, family=family
        )
        self.alphas = alphas
        self.l1_ratios = l1_ratios
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """
        用交叉验证拟合模型，选择最优参数
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
        from sklearn.preprocessing import StandardScaler

        X, y = check_X_y(X, y)

        # 如果groups为None，自动分组
        if self.groups is None:
            self.groups = self.group_features_by_correlation(X)

        # 默认评分指标
        if self.scoring is None:
            if self.family.lower() == "gaussian":
                self.scoring = make_scorer(mean_squared_error, greater_is_better=False)
            else:
                self.scoring = make_scorer(roc_auc_score, needs_proba=True)

        # 默认alpha搜索范围
        if self.alphas is None:
            X_scaled = StandardScaler().fit_transform(X)
            if self.family.lower() == "gaussian":
                y_scaled = StandardScaler().fit_transform(y.reshape(-1,1)).flatten()
                lambda_max = np.max(np.abs(X_scaled.T @ y_scaled)) / len(y)
            else:
                lambda_max = np.max(np.abs(X_scaled.T @ (y - np.mean(y)))) / len(y)
            self.alphas = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_max*1e-4), 50))

        # 参数网格
        param_grid = {
            'alpha': self.alphas,
            'l1_ratio': self.l1_ratios
        }

        # 网格搜索
        grid = GridSearchCV(
            AdaptiveSparseGroupLasso(
                groups=self.groups,
                fit_intercept=self.fit_intercept,
                standardize=self.standardize,
                max_iter=self.max_iter,
                tol=self.tol,
                group_weights=self.group_weights,
                gamma=self.gamma,
                family=self.family
            ),
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=True
        )

        grid.fit(X, y, sample_weight=sample_weight)

        # 保存最优参数和结果
        self.best_params_ = grid.best_params_
        self.best_score_ = grid.best_score_
        self.cv_results_ = grid.cv_results_
        self.alpha = self.best_params_['alpha']
        self.l1_ratio = self.best_params_['l1_ratio']

        # 复制最优模型的属性
        best_model = grid.best_estimator_
        self.coef_ = best_model.coef_
        self.intercept_ = best_model.intercept_
        self.feature_weights = best_model.feature_weights
        self.beta_initial_ = best_model.beta_initial_
        self.group_indices_ = best_model.group_indices_
        self.n_groups_ = best_model.n_groups_
        self.n_features_in_ = best_model.n_features_in_
        self.scaler_X = best_model.scaler_X
        self.scaler_y = best_model.scaler_y

        return self
