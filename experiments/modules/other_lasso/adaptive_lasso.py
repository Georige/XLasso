"""
Adaptive Lasso 自适应Lasso实现
参考: Zou (2006). "The adaptive lasso and its oracle properties."
"""
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge, lasso_path
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import check_cv
from sklearn.base import clone
from joblib import Parallel, delayed
from .base import BaseLasso


class AdaptiveLasso(BaseLasso):
    """
    自适应Lasso回归/分类
    对不同特征的L1惩罚施加不同权重，权重与初始估计的系数绝对值成反比

    参考标准实现: https://github.com/ErikHartman/adalasso
    """
    def __init__(self, alpha=1.0, gamma=1.0, fit_intercept=True, standardize=True,
                 max_iter=5000, tol=1e-4, method='lasso', family="gaussian",
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
    严格隔离CV：每个fold独立计算初始估计和alpha搜索，避免验证集泄露

    参考 Zou (2006) 的自适应Lasso oracle 性质实现
    """
    def __init__(self, alphas=None, gammas=[1.0, 2.0], fit_intercept=True,
                 standardize=True, max_iter=5000, tol=1e-4, method='lasso',
                 family="gaussian", initial_estimator=None, cv=5,
                 scoring=None, n_jobs=None, use_1se=True, alpha_min_ratio=1e-4,
                 random_state=42):
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
            alpha_min_ratio: alpha搜索最小值与最大值的比例
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
        self.alpha_min_ratio = alpha_min_ratio
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, cv_splits=None):
        """
        用交叉验证拟合模型，选择最优参数（严格隔离版）

        每个fold独立计算：
        1. 标准化统计量（仅用训练折）
        2. 初始beta_ols（仅用训练折）
        3. 自适应权重（仅用训练折系数）
        4. alpha搜索路径（仅用训练折）

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
            This ensures all algorithms use the same CV splits for fair comparison.
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error, roc_auc_score

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # 默认评分指标
        if self.scoring is None:
            if self.family.lower() == "gaussian":
                self.scoring = 'neg_mean_squared_error'
            else:
                self.scoring = 'roc_auc'

        # 使用提供的cv_splits或创建新的KFold
        if cv_splits is not None:
            n_folds = len(cv_splits)
            splits = cv_splits
        else:
            n_folds = self.cv
            kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            splits = list(kfold.split(X))

        # 预分配结果矩阵
        n_gamma = len(self.gammas)
        if self.alphas is None:
            n_alpha = 100
        else:
            n_alpha = len(self.alphas)

        error_matrix = np.full((n_gamma, n_alpha, n_folds), np.inf)
        nselected_matrix = np.zeros((n_gamma, n_alpha, n_folds))

        eps = 1e-10
        n_jobs = self.n_jobs if self.n_jobs is not None else -1

        # ============================================================
        # 阶段一：K折严格内部寻优（严格隔离，无泄露）
        # 外层 fold 串行，内层 gamma 并行
        # ============================================================
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_tr_raw, X_va_raw = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            # 每个fold独立标准化
            if self.standardize:
                scaler_fold = StandardScaler()
                X_tr = scaler_fold.fit_transform(X_tr_raw)
                X_va = scaler_fold.transform(X_va_raw)
            else:
                X_tr, X_va = X_tr_raw, X_va_raw

            # 处理样本权重
            if sample_weight is not None:
                sw_tr = np.asarray(sample_weight)[train_idx]
            else:
                sw_tr = None

            # 第一步：在训练折上计算初始beta_ols（严格隔离）
            if self.initial_estimator is not None:
                estimator = clone(self.initial_estimator)
                estimator.fit(X_tr, y_tr, sample_weight=sw_tr)
                beta_ols = estimator.coef_.flatten()
            else:
                if self.family.lower() == "gaussian":
                    if X_tr.shape[0] > X_tr.shape[1]:
                        ols = LinearRegression(fit_intercept=False)
                        ols.fit(X_tr, y_tr, sample_weight=sw_tr)
                        beta_ols = ols.coef_
                    else:
                        ridge = Ridge(alpha=0.1, fit_intercept=False, max_iter=self.max_iter)
                        ridge.fit(X_tr, y_tr, sample_weight=sw_tr)
                        beta_ols = ridge.coef_
                else:
                    lr = LogisticRegression(penalty='l2', C=1.0, fit_intercept=False,
                                          max_iter=self.max_iter, solver='liblinear')
                    lr.fit(X_tr, y_tr, sample_weight=sw_tr)
                    beta_ols = lr.coef_[0]

            # 计算自适应权重
            abs_beta = np.clip(np.abs(beta_ols), eps, None)

            # 内层 gamma 并行
            def _eval_gamma(gamma_idx, gamma):
                weights = 1.0 / (abs_beta ** gamma)

                # 空间重构
                X_adaptive_tr = X_tr / weights
                X_adaptive_va = X_va / weights

                # alpha搜索路径
                if self.alphas is None:
                    if self.family.lower() == "gaussian":
                        lambda_max = np.max(np.abs(X_adaptive_tr.T @ y_tr)) / len(y_tr)
                    else:
                        lambda_max = np.max(np.abs(X_adaptive_tr.T @ (y_tr - np.mean(y_tr)))) / len(y_tr)
                    alphas = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_max * self.alpha_min_ratio), n_alpha))[::-1]
                else:
                    alphas = self.alphas

                fold_gamma_errors = np.full(n_alpha, np.inf)
                fold_gamma_nselected = np.zeros(n_alpha)

                if self.family.lower() == "gaussian":
                    # lasso_path 不支持 sample_weight，有权重时回退到逐个 Lasso
                    if sw_tr is None:
                        # 用 lasso_path 一次算出全部 alpha 系数路径（向量化加速）
                        # 在 transformed space 直接预测（与 AFL-CV-EN 一致）
                        _, coefs_path, _ = lasso_path(
                            X_adaptive_tr, y_tr,
                            alphas=alphas,
                            max_iter=self.max_iter,
                            tol=self.tol,
                        )
                        preds = X_adaptive_va @ coefs_path
                        mse_path = np.mean((y_va[:, np.newaxis] - preds) ** 2, axis=0)
                        fold_gamma_errors[:] = mse_path
                        fold_gamma_nselected[:] = np.sum(coefs_path != 0, axis=0)
                    else:
                        # 有 sample_weight：逐个 Lasso（可正确处理权重）
                        for alpha_idx, alpha in enumerate(alphas):
                            model = Lasso(alpha=alpha, fit_intercept=self.fit_intercept,
                                         max_iter=self.max_iter, tol=self.tol)
                            model.fit(X_adaptive_tr, y_tr, sample_weight=sw_tr)
                            coef_scaled = model.coef_
                            intercept_ = model.intercept_ if self.fit_intercept else 0.0
                            coef_ = coef_scaled / weights
                            y_pred_va = X_va @ coef_ + intercept_
                            mse = np.mean((y_va - y_pred_va) ** 2)
                            fold_gamma_errors[alpha_idx] = mse
                            fold_gamma_nselected[alpha_idx] = np.sum(coef_scaled != 0)
                else:
                    # 分类：用 warm_start 构建系数路径，向量化替代逐个独立拟合
                    # alphas 从大到小（强→弱正则化），对应 C 从小到大
                    # warm_start=True：每个 fit 从上一个解出发继续优化
                    alphas_desc = alphas[::-1]  # strong reg → weak reg
                    Cs_asc = 1.0 / alphas_desc  # small C → large C (weak reg → strong reg)
                    n_Cs = len(Cs_asc)

                    fold_gamma_errors = np.full(n_alpha, np.inf)
                    fold_gamma_nselected = np.zeros(n_alpha)

                    coefs_current = np.zeros(X_adaptive_tr.shape[1])
                    intercept_current = np.array([0.0])

                    for c_idx, C in enumerate(Cs_asc):
                        lr = LogisticRegression(
                            penalty='l1', C=C,
                            fit_intercept=self.fit_intercept,
                            max_iter=self.max_iter, tol=self.tol,
                            solver='liblinear',
                            warm_start=True,
                            random_state=self.random_state
                        )
                        # 从上一个解热启动
                        lr.coef_ = coefs_current.reshape(1, -1)
                        lr.intercept_ = intercept_current
                        lr.fit(X_adaptive_tr, y_tr, sample_weight=sw_tr)
                        coefs_current = lr.coef_.ravel()
                        intercept_current = lr.intercept_

                        # 还原系数（原始空间）
                        coef_scaled = coefs_current
                        intercept_ = intercept_current[0] if self.fit_intercept else 0.0
                        coef_ = coef_scaled / weights
                        z_va = X_va @ coef_ + intercept_
                        y_pred_va = 1 / (1 + np.exp(-np.clip(z_va, -30, 30)))

                        # 在结果矩阵中正确位置（alphas_desc 逆序对应 alphas 正序）
                        alpha_idx = n_alpha - 1 - c_idx

                        if self.scoring == 'neg_mean_squared_error':
                            mse = np.mean((y_va - y_pred_va) ** 2)
                            fold_gamma_errors[alpha_idx] = mse
                        elif self.scoring == 'roc_auc' and self.family.lower() == "binomial":
                            try:
                                auc = roc_auc_score(y_va, y_pred_va)
                                fold_gamma_errors[alpha_idx] = -auc
                            except:
                                fold_gamma_errors[alpha_idx] = np.inf

                        fold_gamma_nselected[alpha_idx] = np.sum(coef_scaled != 0)

                return gamma_idx, fold_gamma_errors, fold_gamma_nselected

            gamma_results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_eval_gamma)(gamma_idx, gamma)
                for gamma_idx, gamma in enumerate(self.gammas)
            )

            fold_errors = np.full((n_gamma, n_alpha), np.inf)
            fold_nselected = np.zeros((n_gamma, n_alpha))
            for gamma_idx, gamma_errors, gamma_nselected in gamma_results:
                fold_errors[gamma_idx, :] = gamma_errors
                fold_nselected[gamma_idx, :] = gamma_nselected

            error_matrix[:, :, fold_idx] = fold_errors
            nselected_matrix[:, :, fold_idx] = fold_nselected

        # ============================================================
        # 阶段二：选拔最优参数（1-SE法则）
        # ============================================================
        mean_error = np.mean(error_matrix, axis=2)  # (n_gamma, n_alpha)
        std_error = np.std(error_matrix, axis=2) / np.sqrt(n_folds)  # SE of mean
        mean_nselected = np.mean(nselected_matrix, axis=2)  # (n_gamma, n_alpha)

        # 对于MSE类指标（neg_mean_squared_error），越小越好
        if self.scoring == 'neg_mean_squared_error':
            best_score = np.min(mean_error)
            best_score_idx = np.unravel_index(np.argmin(mean_error), mean_error.shape)
            threshold = best_score + std_error[best_score_idx]
            # 找所有MSE <= threshold的候选
            candidates_mask = mean_error <= threshold
        else:
            # 对于roc_auc，越大越好
            best_score = np.max(mean_error)
            best_score_idx = np.unravel_index(np.argmax(mean_error), mean_error.shape)
            threshold = best_score - std_error[best_score_idx]
            # 找所有score >= threshold的候选
            candidates_mask = mean_error >= threshold

        if not np.any(candidates_mask):
            if hasattr(self, 'verbose') and self.verbose:
                print("[AdaptiveLassoCV] Warning: No candidates within 1-SE, using standard min/max")
            best_gamma_idx, best_alpha_idx = best_score_idx
        else:
            # 在候选中选择n_selected最少的（最稀疏的）
            masked_nselected = np.where(candidates_mask, mean_nselected, np.inf)
            best_flat_idx = np.argmin(masked_nselected)
            best_gamma_idx, best_alpha_idx = np.unravel_index(best_flat_idx, mean_error.shape)

        self.best_gamma_ = self.gammas[best_gamma_idx]

        # 重建该gamma对应的alpha搜索路径以获取best_alpha
        if self.alphas is None:
            X_for_rebuild = X[splits[0][0]]
            y_for_rebuild = y[splits[0][0]]
            if self.standardize:
                scaler_rebuild = StandardScaler()
                X_for_rebuild = scaler_rebuild.fit_transform(X_for_rebuild)
            # 重新计算beta_ols和weights
            if X_for_rebuild.shape[0] > X_for_rebuild.shape[1]:
                ols = LinearRegression(fit_intercept=False)
                ols.fit(X_for_rebuild, y_for_rebuild)
                beta_rebuild = ols.coef_
            else:
                ridge = Ridge(alpha=0.1, fit_intercept=False)
                ridge.fit(X_for_rebuild, y_for_rebuild)
                beta_rebuild = ridge.coef_
            abs_beta_rebuild = np.clip(np.abs(beta_rebuild), eps, None)
            weights_rebuild = 1.0 / (abs_beta_rebuild ** self.best_gamma_)
            X_adaptive_rebuild = X_for_rebuild / weights_rebuild
            lambda_max_rebuild = np.max(np.abs(X_adaptive_rebuild.T @ y_for_rebuild)) / len(y_for_rebuild)
            alphas_final = np.exp(np.linspace(np.log(lambda_max_rebuild), np.log(lambda_max_rebuild * self.alpha_min_ratio), n_alpha))[::-1]
            self.best_alpha_ = alphas_final[best_alpha_idx]
        else:
            self.best_alpha_ = self.alphas[best_alpha_idx]

        best_cv_score = mean_error[best_gamma_idx, best_alpha_idx]
        best_cv_nselected = mean_nselected[best_gamma_idx, best_alpha_idx]
        self.cv_score_ = -best_cv_score if self.scoring == 'neg_mean_squared_error' else best_cv_score
        self.cv_std_ = std_error[best_gamma_idx, best_alpha_idx]

        if hasattr(self, 'verbose') and self.verbose:
            print(f"\n[AdaptiveLassoCV] Selected: gamma={self.best_gamma_}, alpha={self.best_alpha_:.6f}")
            print(f"[AdaptiveLassoCV] CV Score={best_cv_score:.6f}, n_selected~{best_cv_nselected:.1f}")

        # ============================================================
        # 阶段三：全量数据终极拟合
        # ============================================================
        # 用全量数据fit scaler（阶段一的CV中已避免泄露，此处可接受）
        if self.standardize:
            self.scaler_X = StandardScaler()
            X_for_final = self.scaler_X.fit_transform(X)
        else:
            X_for_final = X

        # 计算初始beta_ols（全量数据，用于最终模型）
        if self.initial_estimator is not None:
            estimator_final = clone(self.initial_estimator)
            estimator_final.fit(X_for_final, y, sample_weight=sample_weight)
            beta_ols_final = estimator_final.coef_.flatten()
        else:
            if self.family.lower() == "gaussian":
                if X_for_final.shape[0] > X_for_final.shape[1]:
                    ols_final = LinearRegression(fit_intercept=False)
                    ols_final.fit(X_for_final, y, sample_weight=sample_weight)
                    beta_ols_final = ols_final.coef_
                else:
                    ridge_final = Ridge(alpha=0.1, fit_intercept=False)
                    ridge_final.fit(X_for_final, y, sample_weight=sample_weight)
                    beta_ols_final = ridge_final.coef_
            else:
                lr_final = LogisticRegression(penalty='l2', C=1.0, fit_intercept=False,
                                            max_iter=self.max_iter, solver='liblinear')
                lr_final.fit(X_for_final, y, sample_weight=sample_weight)
                beta_ols_final = lr_final.coef_[0]

        # 计算自适应权重
        abs_beta_final = np.clip(np.abs(beta_ols_final), eps, None)
        weights_final = 1.0 / (abs_beta_final ** self.best_gamma_)

        # 空间重构
        X_adaptive_final = X_for_final / weights_final

        # 拟合最终模型
        if self.family.lower() == "gaussian":
            model_final = Lasso(alpha=self.best_alpha_, fit_intercept=self.fit_intercept,
                               max_iter=self.max_iter, tol=self.tol)
            model_final.fit(X_adaptive_final, y, sample_weight=sample_weight)
            coef_scaled_final = model_final.coef_
            intercept_final = model_final.intercept_ if self.fit_intercept else 0.0
        else:
            model_final = LogisticRegression(
                penalty='l1', C=1.0/self.best_alpha_ if self.best_alpha_ > 0 else 1e10,
                fit_intercept=self.fit_intercept, max_iter=self.max_iter,
                tol=self.tol, solver='liblinear'
            )
            model_final.fit(X_adaptive_final, y, sample_weight=sample_weight)
            coef_scaled_final = model_final.coef_[0]
            intercept_final = model_final.intercept_[0] if self.fit_intercept else 0.0

        # 还原系数
        self.coef_ = coef_scaled_final / weights_final
        self.beta_initial_ = beta_ols_final
        self.weights_ = weights_final

        # 逆标准化
        if self.standardize:
            self.coef_ = self.coef_ / self.scaler_X.scale_
            if self.fit_intercept:
                if self.family.lower() == "gaussian":
                    self.intercept_ = np.mean(y) - np.sum(self.coef_ * self.scaler_X.mean_)
                else:
                    self.intercept_ = intercept_final - np.sum(self.coef_ * self.scaler_X.mean_)
            self.scaler_y = None
        else:
            self.intercept_ = intercept_final

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

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
