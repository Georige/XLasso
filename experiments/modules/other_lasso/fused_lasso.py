"""
Fused Lasso 融合Lasso实现
参考: Tibshirani et al. (2005). "Sparsity and smoothness via the fused lasso."
适合特征有顺序/空间/时间相关性的场景
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

class FusedLasso(BaseLasso):
    """
    融合Lasso回归/分类
    同时对系数本身和相邻系数的差异施加L1惩罚，适合有序特征场景

    参考标准实现: https://github.com/jaredhuling/fusedlasso
    """
    def __init__(self, alpha=1.0, lambda_fused=None, fit_intercept=True, standardize=True,
                 max_iter=1000, tol=1e-4, family="gaussian", D=None):
        """
        初始化参数
        Parameters:
            alpha: 系数L1惩罚强度（也称为lambda1）
            lambda_fused: 融合惩罚强度（也称为lambda2），默认等于alpha
            fit_intercept: 是否拟合截距
            standardize: 是否标准化特征
            max_iter: 最大迭代次数
            tol: 收敛阈值
            family: 模型家族，支持"gaussian"(回归)或"binomial"(二分类)
            D: 自定义差异矩阵，默认是一阶差分矩阵（相邻系数差异）
               传入自定义矩阵可以支持任意结构的融合惩罚
        """
        super().__init__(alpha, fit_intercept, standardize, max_iter, tol, family)
        self.lambda_fused = lambda_fused if lambda_fused is not None else alpha
        self.D = D

        if not CVXPY_AVAILABLE:
            raise ImportError("FusedLasso需要cvxpy库，请运行: pip install cvxpy")

    def fit(self, X, y, sample_weight=None):
        """
        拟合融合Lasso模型
        Parameters:
            X: 特征矩阵 (n_samples, n_features)，特征按顺序排列
            y: 目标向量 (n_samples,)
            sample_weight: 样本权重，可选
        Returns:
            self: 拟合后的模型
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        n_samples, n_features = X.shape

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

        # 正则化项
        l1_penalty = self.alpha * cp.norm1(beta)

        # 融合惩罚
        if self.D is None:
            # 默认一阶差分：相邻系数差异
            fused_penalty = self.lambda_fused * cp.norm1(cp.diff(beta))
        else:
            # 自定义差异矩阵
            fused_penalty = self.lambda_fused * cp.norm1(self.D @ beta)

        penalty = l1_penalty + fused_penalty

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

        return self

    @staticmethod
    def path(X, y, alphas=None, n_alphas=100, alpha_min_ratio=0.01, lambda_fused_ratio=1.0, **kwargs):
        """
        计算正则化路径
        Parameters:
            X: 特征矩阵
            y: 目标向量
            alphas: 正则化参数列表，自动生成
            n_alphas: 正则化路径长度
            alpha_min_ratio: 最小alpha与最大alpha的比值
            lambda_fused_ratio: 融合惩罚与系数惩罚的比例，默认1.0
            **kwargs: 其他参数传给FusedLasso
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
            model = FusedLasso(
                alpha=alpha,
                lambda_fused=alpha * lambda_fused_ratio,
                **kwargs
            )
            model.fit(X, y)
            coefs.append(model.coef_)

        return alphas, np.array(coefs)


class FusedLassoCV(FusedLasso):
    """
    带交叉验证的融合Lasso，自动选择最优alpha和lambda_fused参数
    """
    def __init__(self, alphas=None, lambda_fused_ratios=[0.1, 0.5, 1.0, 2.0, 5.0],
                 fit_intercept=True, standardize=True, max_iter=1000, tol=1e-4,
                 family="gaussian", D=None, cv=5, scoring=None, n_jobs=None):
        """
        初始化参数
        Parameters:
            alphas: 待搜索的alpha值列表，默认自动生成
            lambda_fused_ratios: lambda_fused/alpha的比例列表，默认[0.1,0.5,1.0,2.0,5.0]
            fit_intercept: 是否拟合截距
            standardize: 是否标准化特征
            max_iter: 最大迭代次数
            tol: 收敛阈值
            family: 模型家族
            D: 自定义差异矩阵
            cv: 交叉验证折数
            scoring: 评价指标，默认：回归用neg_mean_squared_error，分类用roc_auc
            n_jobs: 并行作业数
        """
        super().__init__(
            alpha=None, lambda_fused=None, fit_intercept=fit_intercept,
            standardize=standardize, max_iter=max_iter, tol=tol,
            family=family, D=D
        )
        self.alphas = alphas
        self.lambda_fused_ratios = lambda_fused_ratios
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

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

        X, y = check_X_y(X, y)

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
            self.alphas = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_max*1e-4), 50))

        # 参数网格
        param_grid = []
        for alpha in self.alphas:
            for ratio in self.lambda_fused_ratios:
                param_grid.append({
                    'alpha': [alpha],
                    'lambda_fused': [alpha * ratio]
                })

        # 网格搜索
        cv_used = cv_splits if cv_splits is not None else self.cv
        grid = GridSearchCV(
            FusedLasso(
                fit_intercept=self.fit_intercept,
                standardize=self.standardize,
                max_iter=self.max_iter,
                tol=self.tol,
                family=self.family,
                D=self.D
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
        self.lambda_fused = self.best_params_['lambda_fused']

        # 复制最优模型的属性
        best_model = grid.best_estimator_
        self.coef_ = best_model.coef_
        self.intercept_ = best_model.intercept_
        self.n_features_in_ = best_model.n_features_in_
        self.scaler_X = best_model.scaler_X
        self.scaler_y = best_model.scaler_y

        return self