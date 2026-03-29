"""
Group Lasso 组Lasso实现
参考: Yuan & Lin (2006). "Model selection and estimation in regression with grouped variables."
适合特征有分组结构的场景，同一组的特征要么都选要么都不选
"""
import numpy as np
from .base import BaseLasso
from sklearn.utils.validation import check_X_y

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

class GroupLasso(BaseLasso):
    """
    组Lasso回归/分类
    对特征组施加L2范数惩罚，实现组级稀疏性
    """
    def __init__(self, alpha=1.0, groups=None, fit_intercept=True, standardize=True,
                 max_iter=1000, tol=1e-4, group_weights=None, family="gaussian"):
        """
        初始化参数
        Parameters:
            alpha: 正则化强度
            groups: 分组列表，每个元素是组内特征的索引列表，例如[[0,1,2], [3,4], [5,6,7,8]]
                   如果为None，每个特征单独为一组，退化为普通Lasso
            fit_intercept: 是否拟合截距
            standardize: 是否标准化特征
            max_iter: 最大迭代次数
            tol: 收敛阈值
            group_weights: 每组的权重，默认每组权重为sqrt(组大小)
            family: 模型家族，支持"gaussian"(回归)或"binomial"(二分类)
        """
        super().__init__(alpha, fit_intercept, standardize, max_iter, tol, family)
        self.groups = groups
        self.group_weights = group_weights

        if not CVXPY_AVAILABLE:
            raise ImportError("GroupLasso需要cvxpy库，请运行: pip install cvxpy")

    def fit(self, X, y, sample_weight=None):
        """
        拟合组Lasso模型
        Parameters:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标向量 (n_samples,)
            sample_weight: 样本权重，可选
        Returns:
            self: 拟合后的模型
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

        # 定义优化变量
        beta = cp.Variable(n_features)
        intercept = cp.Variable() if self.fit_intercept else 0.0

        # 损失函数
        if self.family.lower() == "gaussian":
            if sample_weight is None:
                loss = cp.sum_squares(y_processed - (X_processed @ beta + intercept)) / (2 * n_samples)
            else:
                loss = cp.sum(cp.multiply(sample_weight, cp.square(y_processed - (X_processed @ beta + intercept)))) / (2 * n_samples)
        else:  # binomial
            # 逻辑回归损失
            logits = X_processed @ beta + intercept
            if sample_weight is None:
                loss = cp.sum(cp.logistic(-cp.multiply(2 * y_processed - 1, logits))) / n_samples
            else:
                loss = cp.sum(cp.multiply(sample_weight, cp.logistic(-cp.multiply(2 * y_processed - 1, logits)))) / n_samples

        # 组L2正则化项
        group_penalty = 0.0
        for g_idx, group in enumerate(self.groups):
            group_size = len(group)
            # 默认组权重为sqrt(组大小)，符合组Lasso标准定义
            if self.group_weights is None:
                w = np.sqrt(group_size)
            else:
                w = self.group_weights[g_idx]
            group_penalty += w * cp.norm2(beta[group])

        penalty = self.alpha * group_penalty

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

    def get_group_importance(self):
        """获取每组的重要性（组内系数的L2范数）"""
        check_is_fitted(self)
        group_importance = []
        for group in self.groups:
            group_importance.append(np.linalg.norm(self.coef_[group]))
        return np.array(group_importance)

    @staticmethod
    def path(X, y, groups=None, alphas=None, n_alphas=100, alpha_min_ratio=0.01, **kwargs):
        """
        计算正则化路径
        Parameters:
            X: 特征矩阵
            y: 目标向量
            groups: 分组结构
            alphas: 正则化参数列表，自动生成
            n_alphas: 正则化路径长度
            alpha_min_ratio: 最小alpha与最大alpha的比值
            **kwargs: 其他参数传给GroupLasso
        Returns:
            alphas: 正则化参数数组
            coefs: 系数矩阵 (n_alphas, n_features)
        """
        if alphas is None:
            # 自动生成alpha路径
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X)
            y_scaled = StandardScaler().fit_transform(y.reshape(-1,1)).flatten()
            lambda_max = np.max(np.abs(X_scaled.T @ y_scaled)) / len(y)
            alphas = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_max * alpha_min_ratio), n_alphas))

        coefs = []
        for alpha in alphas:
            model = GroupLasso(alpha=alpha, groups=groups, **kwargs)
            model.fit(X, y)
            coefs.append(model.coef_)

        return alphas, np.array(coefs)

    @staticmethod
    def group_features_by_correlation(X, corr_threshold=0.7, max_group_size=20):
        """
        基于相关性自动分组
        Parameters:
            X: 特征矩阵
            corr_threshold: 相关系数阈值，超过则分为一组
            max_group_size: 最大组大小
        Returns:
            groups: 分组列表
        """
        corr_matrix = np.corrcoef(X.T)
        n_features = X.shape[1]
        assigned = np.zeros(n_features, dtype=bool)
        groups = []

        for i in range(n_features):
            if assigned[i]:
                continue
            # 找到和i高度相关的未分配特征
            correlated = np.where(
                (~assigned) & (np.abs(corr_matrix[i]) >= corr_threshold)
            )[0]
            # 限制组大小
            if len(correlated) > max_group_size:
                correlated = correlated[:max_group_size]
            groups.append(correlated.tolist())
            assigned[correlated] = True

        return groups


class GroupLassoCV(GroupLasso):
    """
    带交叉验证的组Lasso，自动选择最优alpha参数
    """
    def __init__(self, alphas=None, groups=None, fit_intercept=True, standardize=True,
                 max_iter=1000, tol=1e-4, group_weights=None, family="gaussian",
                 cv=5, scoring=None, n_jobs=None, use_1se=True):
        """
        初始化参数
        Parameters:
            alphas: 待搜索的alpha值列表，默认自动生成
            groups: 分组列表
            fit_intercept: 是否拟合截距
            standardize: 是否标准化特征
            max_iter: 最大迭代次数
            tol: 收敛阈值
            group_weights: 每组的权重
            family: 模型家族
            cv: 交叉验证折数
            scoring: 评价指标，默认：回归用neg_mean_squared_error，分类用roc_auc
            n_jobs: 并行作业数
            use_1se: 是否使用1-SE规则选择更保守的模型（更稀疏），默认True
        """
        super().__init__(
            alpha=None, groups=groups, fit_intercept=fit_intercept,
            standardize=standardize, max_iter=max_iter, tol=tol,
            group_weights=group_weights, family=family
        )
        self.alphas = alphas
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.use_1se = use_1se

    def fit(self, X, y, sample_weight=None, cv_splits=None):
        """
        用交叉验证拟合模型，选择最优参数

        Parameters
        ----------
        X : array-like
            Training data
        y : array-like
            Target values
        sample_weight : array-like, optional
            Sample weights
        cv_splits : list of tuples, optional
            Pre-generated CV splits (list of (train_idx, val_idx) tuples).
            If provided, uses these splits instead of creating new KFold.
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
        from sklearn.preprocessing import StandardScaler

        X, y = check_X_y(X, y)

        # 保存用于后续 1-SE 计算
        self._X_for_cv = X
        self._y_for_cv = y

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
            # 生成alpha搜索范围
            X_scaled = StandardScaler().fit_transform(X)
            if self.family.lower() == "gaussian":
                y_scaled = StandardScaler().fit_transform(y.reshape(-1,1)).flatten()
                lambda_max = np.max(np.abs(X_scaled.T @ y_scaled)) / len(y)
            else:
                lambda_max = np.max(np.abs(X_scaled.T @ (y - np.mean(y)))) / len(y)
            self.alphas = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_max*1e-4), 100))

        # 参数网格
        param_grid = {'alpha': self.alphas}

        # 网格搜索
        cv_used = cv_splits if cv_splits is not None else self.cv
        grid = GridSearchCV(
            GroupLasso(
                groups=self.groups,
                fit_intercept=self.fit_intercept,
                standardize=self.standardize,
                max_iter=self.max_iter,
                tol=self.tol,
                group_weights=self.group_weights,
                family=self.family
            ),
            param_grid=param_grid,
            cv=cv_used,
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

        # 1-SE 规则：选择最稀疏的模型，其分数在 best - 1*std 范围内
        if self.use_1se:
            best_alpha_1se = self._select_1se_alpha()
            if hasattr(self, 'verbose') and self.verbose:
                print(f"[GroupLassoCV] Best alpha (CV): {self.alpha:.6f}")
                print(f"[GroupLassoCV] 1-SE alpha:      {best_alpha_1se:.6f}")
            self.alpha = best_alpha_1se
            # 重新拟合最终模型
            final_model = GroupLasso(
                alpha=self.alpha, groups=self.groups,
                fit_intercept=self.fit_intercept, standardize=self.standardize,
                max_iter=self.max_iter, tol=self.tol,
                group_weights=self.group_weights, family=self.family
            )
            final_model.fit(X, y, sample_weight=sample_weight)
            self.coef_ = final_model.coef_
            self.intercept_ = final_model.intercept_
            self.group_indices_ = final_model.group_indices_
            self.n_groups_ = final_model.n_groups_
            self.n_features_in_ = final_model.n_features_in_
            self.scaler_X = final_model.scaler_X
            self.scaler_y = final_model.scaler_y
        else:
            # 复制最优模型的属性
            best_model = grid.best_estimator_
            self.coef_ = best_model.coef_
            self.intercept_ = best_model.intercept_
            self.group_indices_ = best_model.group_indices_
            self.n_groups_ = best_model.n_groups_
            self.n_features_in_ = best_model.n_features_in_
            self.scaler_X = best_model.scaler_X
            self.scaler_y = best_model.scaler_y

        return self

    def _select_1se_alpha(self):
        """
        1-SE 规则：选择最简洁的模型，其分数在 best - 1*SE 范围内

        在候选中选择非零系数数量最少的（最稀疏的）

        Returns:
            best_alpha_1se: 通过1-SE检验的最稀疏模型
        """
        cv_results = self.cv_results_
        mean_score = cv_results['mean_test_score']
        std_score = cv_results['std_test_score']
        alphas = cv_results['param_alpha']

        # 找出最佳分数
        best_score = np.max(mean_score)

        # 严格使用标准误 SE = std / sqrt(K)
        K = self.cv
        se_score = std_score / np.sqrt(K)

        # 计算阈值：best_score - 1*SE (score 越大越好，所以用减号)
        threshold = best_score - se_score

        # 找出所有通过 1-SE 检验的候选
        candidates_mask = mean_score >= threshold
        candidate_indices = np.where(candidates_mask)[0]

        if len(candidate_indices) == 0:
            return self.best_params_['alpha']

        # 计算每个候选的非零系数数量，选择最稀疏的
        n_nonzero_list = []
        for idx in candidate_indices:
            alpha = alphas[idx]
            # 临时创建模型获取系数
            model = GroupLasso(
                alpha=alpha, groups=self.groups,
                fit_intercept=self.fit_intercept, standardize=self.standardize,
                max_iter=self.max_iter, tol=self.tol,
                group_weights=self.group_weights, family=self.family
            )
            model.fit(self._X_for_cv, self._y_for_cv)
            n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
            n_nonzero_list.append(n_nonzero)

        # 选择非零系数最少的（最稀疏的）
        best_candidate_idx = candidate_indices[np.argmin(n_nonzero_list)]
        return float(alphas[best_candidate_idx])