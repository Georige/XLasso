"""
Adaptive Lasso 自适应Lasso实现
参考: Zou (2006). "The adaptive lasso and its oracle properties."
"""
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import check_cv
from sklearn.base import clone
from .base import BaseLasso

class AdaptiveLasso(BaseLasso):
    """
    自适应Lasso回归/分类
    对不同特征的L1惩罚施加不同权重，权重与初始估计的系数绝对值成反比

    参考标准实现: https://github.com/ErikHartman/adalasso
    """
    def __init__(self, alpha=1.0, gamma=1.0, fit_intercept=True, standardize=True,
                 max_iter=1000, tol=1e-4, method='lasso', family="gaussian",
                 initial_estimator=None):
        """
        初始化参数
        Parameters:
            alpha: 全局正则化强度
            gamma: 权重指数，推荐值1.0或2.0
            fit_intercept: 是否拟合截距
            standardize: 是否标准化特征
            max_iter: 最大迭代次数
            tol: 收敛阈值
            method: 求解方法，'lasso'(坐标下降) 或 'cd'(坐标下降，同lasso)
            family: 模型家族，支持"gaussian"(回归)或"binomial"(二分类)
            initial_estimator: 用于计算初始系数的估计器，默认:
                - 回归任务：n>p时用OLS，n<p时用Ridge(alpha=0.1)
                - 分类任务：用L2正则化LogisticRegression
        """
        super().__init__(alpha, fit_intercept, standardize, max_iter, tol, family)
        self.gamma = gamma
        self.method = method
        self.initial_estimator = initial_estimator

    def fit(self, X, y, sample_weight=None):
        """
        拟合自适应Lasso模型
        Parameters:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标向量 (n_samples,)
            sample_weight: 样本权重，可选
        Returns:
            self: 拟合后的模型
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # 预处理
        X_processed, y_processed = self._preprocess(X, y)
        self.X_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y)

        # 第一步：拟合初始模型得到初始系数
        if self.initial_estimator is not None:
            # 使用用户自定义的初始估计器
            estimator = clone(self.initial_estimator)
            estimator.fit(X_processed, y_processed, sample_weight=sample_weight)
            if hasattr(estimator, 'coef_'):
                beta_ols = estimator.coef_.flatten()
            else:
                raise ValueError("Initial estimator must have a coef_ attribute")
        else:
            # 使用默认初始估计器
            if self.family.lower() == "gaussian":
                if X.shape[0] > X.shape[1]:
                    # 样本量大于特征数，使用OLS
                    ols = LinearRegression(fit_intercept=False)
                    ols.fit(X_processed, y_processed, sample_weight=sample_weight)
                    beta_ols = ols.coef_
                else:
                    # 高维场景，使用岭回归得到初始系数
                    ridge = Ridge(alpha=0.1, fit_intercept=False, max_iter=self.max_iter)
                    ridge.fit(X_processed, y_processed, sample_weight=sample_weight)
                    beta_ols = ridge.coef_
            else:  # binomial
                # 二分类使用L2正则化逻辑回归得到初始系数
                lr = LogisticRegression(penalty='l2', C=1.0, fit_intercept=False,
                                      max_iter=self.max_iter, solver='liblinear')
                lr.fit(X_processed, y_processed, sample_weight=sample_weight)
                beta_ols = lr.coef_[0]

        # 计算自适应权重: w_j = 1 / |beta_ols_j|^gamma
        eps = 1e-10
        abs_beta = np.abs(beta_ols)
        abs_beta_safe = np.clip(abs_beta, eps, None)  # Prevent near-zero
        weights = 1.0 / abs_beta_safe ** self.gamma

        # 第二步：带权重的L1正则化回归
        # 对特征进行加权，等价于在目标函数中加入权重
        X_weighted = X_processed / weights[np.newaxis, :]

        if self.family.lower() == "gaussian":
            model = Lasso(
                alpha=self.alpha,
                fit_intercept=False,
                max_iter=self.max_iter,
                tol=self.tol
            )
            model.fit(X_weighted, y_processed, sample_weight=sample_weight)
            coef_scaled = model.coef_
            intercept_ = 0.0
        else:  # binomial
            model = LogisticRegression(
                penalty='l1',
                C=1.0/self.alpha if self.alpha > 0 else 1e10,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                solver='liblinear'
            )
            model.fit(X_weighted, y_processed, sample_weight=sample_weight)
            coef_scaled = model.coef_[0]
            intercept_ = model.intercept_[0] if self.fit_intercept else 0.0

        # 还原系数
        coef_ = coef_scaled / weights

        # 后处理，还原到原始尺度
        self._postprocess(coef_, intercept_)

        # 保存中间结果
        self.beta_initial_ = beta_ols
        self.weights_ = weights

        return self


class AdaptiveLassoCV(AdaptiveLasso):
    """
    带交叉验证的自适应Lasso，自动选择最优alpha和gamma参数
    """
    def __init__(self, alphas=None, gammas=[1.0, 2.0], fit_intercept=True,
                 standardize=True, max_iter=1000, tol=1e-4, method='lasso',
                 family="gaussian", initial_estimator=None, cv=5,
                 scoring=None, n_jobs=None, use_1se=True):
        """
        初始化参数
        Parameters:
            alphas: 待搜索的alpha值列表，默认自动生成
            gammas: 待搜索的gamma值列表，默认[1.0, 2.0]
            fit_intercept: 是否拟合截距
            standardize: 是否标准化特征
            max_iter: 最大迭代次数
            tol: 收敛阈值
            method: 求解方法
            family: 模型家族
            initial_estimator: 初始估计器
            cv: 交叉验证折数
            scoring: 评价指标，默认：回归用neg_mean_squared_error，分类用roc_auc
            n_jobs: 并行作业数
            use_1se: 是否使用1-SE规则选择更保守的模型（更稀疏），默认True
        """
        super().__init__(
            alpha=None, gamma=None, fit_intercept=fit_intercept,
            standardize=standardize, max_iter=max_iter, tol=tol,
            method=method, family=family, initial_estimator=initial_estimator
        )
        self.alphas = alphas
        self.gammas = gammas
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
        from sklearn.metrics import get_scorer

        X, y = check_X_y(X, y)

        # 保存用于后续 1-SE 计算
        self._X_for_cv = X
        self._y_for_cv = y

        # 默认评分指标
        from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
        if self.scoring is None:
            if self.family.lower() == "gaussian":
                self.scoring = make_scorer(mean_squared_error, greater_is_better=False)
            else:
                self.scoring = make_scorer(roc_auc_score, needs_proba=True)

        # 默认alpha搜索范围
        if self.alphas is None:
            # 生成alpha搜索范围，和sklearn LassoCV保持一致
            X_scaled = StandardScaler().fit_transform(X)
            if self.family.lower() == "gaussian":
                y_scaled = StandardScaler().fit_transform(y.reshape(-1,1)).flatten()
                lambda_max = np.max(np.abs(X_scaled.T @ y_scaled)) / len(y)
            else:
                lambda_max = np.max(np.abs(X_scaled.T @ (y - np.mean(y)))) / len(y)
            self.alphas = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_max*1e-4), 100))

        # 参数网格
        param_grid = {
            'alpha': self.alphas,
            'gamma': self.gammas
        }

        # 预计算初始beta以避免重复Ridge fits
        X_processed = self._preprocess(X, y)[0]
        if self.initial_estimator is not None:
            estimator = clone(self.initial_estimator)
            estimator.fit(X_processed, y, sample_weight=sample_weight)
            beta_ols = estimator.coef_.flatten()
        else:
            if self.family.lower() == "gaussian":
                if X.shape[0] > X.shape[1]:
                    ols = LinearRegression(fit_intercept=False)
                    ols.fit(X_processed, y, sample_weight=sample_weight)
                    beta_ols = ols.coef_
                else:
                    ridge = Ridge(alpha=0.1, fit_intercept=False, max_iter=self.max_iter)
                    ridge.fit(X_processed, y, sample_weight=sample_weight)
                    beta_ols = ridge.coef_
            else:
                lr = LogisticRegression(penalty='l2', C=1.0, fit_intercept=False,
                                      max_iter=self.max_iter, solver='liblinear')
                lr.fit(X_processed, y, sample_weight=sample_weight)
                beta_ols = lr.coef_[0]

        # 网格搜索
        cv_used = cv_splits if cv_splits is not None else self.cv
        grid = GridSearchCV(
            AdaptiveLasso(
                fit_intercept=self.fit_intercept,
                standardize=self.standardize,
                max_iter=self.max_iter,
                tol=self.tol,
                method=self.method,
                family=self.family,
                initial_estimator=self.initial_estimator
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
        self.gamma = self.best_params_['gamma']

        # 1-SE 规则：选择最稀疏的模型，其分数在 best - 1*std 范围内
        if self.use_1se:
            best_alpha_1se, best_gamma_1se = self._select_1se_params()
            if hasattr(self, 'verbose') and self.verbose:
                print(f"[AdaptiveLassoCV] Best (alpha, gamma): ({self.alpha:.6f}, {self.gamma})")
                print(f"[AdaptiveLassoCV] 1-SE (alpha, gamma): ({best_alpha_1se:.6f}, {best_gamma_1se})")
            self.alpha = best_alpha_1se
            self.gamma = best_gamma_1se
            # 重新拟合最终模型
            final_model = AdaptiveLasso(
                alpha=self.alpha, gamma=self.gamma,
                fit_intercept=self.fit_intercept, standardize=self.standardize,
                max_iter=self.max_iter, tol=self.tol,
                method=self.method, family=self.family,
                initial_estimator=self.initial_estimator
            )
            final_model.fit(X, y, sample_weight=sample_weight)
            self.coef_ = final_model.coef_
            self.intercept_ = final_model.intercept_
            self.beta_initial_ = final_model.beta_initial_
            self.weights_ = final_model.weights_
            self.n_features_in_ = final_model.n_features_in_
            self.scaler_X = final_model.scaler_X
            self.scaler_y = final_model.scaler_y
        else:
            # 复制最优模型的属性
            best_model = grid.best_estimator_
            self.coef_ = best_model.coef_
            self.intercept_ = best_model.intercept_
            self.beta_initial_ = best_model.beta_initial_
            self.weights_ = best_model.weights_
            self.n_features_in_ = best_model.n_features_in_
            self.scaler_X = best_model.scaler_X
            self.scaler_y = best_model.scaler_y

        return self

    def _select_1se_params(self):
        """
        1-SE 规则：选择最简洁的模型，其分数在 best - 1*SE 范围内

        在候选中选择非零系数数量最少的（最稀疏的）

        Returns:
            tuple: (best_alpha_1se, best_gamma_1se)
        """
        cv_results = self.cv_results_
        mean_score = cv_results['mean_test_score']
        std_score = cv_results['std_test_score']
        alphas = cv_results['param_alpha']
        gammas = cv_results['param_gamma']

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
            # 如果没有候选，返回原始最优
            return self.best_params_['alpha'], self.best_params_['gamma']

        # 计算每个候选的非零系数数量，选择最稀疏的
        n_nonzero_list = []
        for idx in candidate_indices:
            alpha = alphas[idx]
            gamma = gammas[idx]
            # 临时创建模型获取系数
            model = AdaptiveLasso(
                alpha=alpha, gamma=gamma,
                fit_intercept=self.fit_intercept, standardize=self.standardize,
                max_iter=self.max_iter, tol=self.tol,
                method=self.method, family=self.family,
                initial_estimator=self.initial_estimator
            )
            model.fit(self._X_for_cv, self._y_for_cv)
            n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
            n_nonzero_list.append(n_nonzero)

        # 选择非零系数最少的（最稀疏的）
        best_candidate_idx = candidate_indices[np.argmin(n_nonzero_list)]
        return alphas[best_candidate_idx], gammas[best_candidate_idx]

    @staticmethod
    def path(X, y, alphas=None, n_alphas=100, alpha_min_ratio=0.01, gamma=1.0, **kwargs):
        """
        计算正则化路径
        Parameters:
            X: 特征矩阵
            y: 目标向量
            alphas: 正则化参数列表，自动生成
            n_alphas: 正则化路径长度
            alpha_min_ratio: 最小alpha与最大alpha的比值
            gamma: 自适应权重指数
            **kwargs: 其他参数传给AdaptiveLasso
        Returns:
            alphas: 正则化参数数组
            coefs: 系数矩阵 (n_alphas, n_features)
        """
        if alphas is None:
            # 自动生成alpha路径
            X_scaled = StandardScaler().fit_transform(X)
            y_scaled = StandardScaler().fit_transform(y.reshape(-1,1)).flatten()
            lambda_max = np.max(np.abs(X_scaled.T @ y_scaled)) / len(y)
            alphas = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_max * alpha_min_ratio), n_alphas))

        coefs = []
        for alpha in alphas:
            model = AdaptiveLasso(alpha=alpha, gamma=gamma, **kwargs)
            model.fit(X, y)
            coefs.append(model.coef_)

        return alphas, np.array(coefs)
