"""
API层：NLasso主类
整合所有模块实现完整算法流程
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted
from ..base import NLassoRegressor as BaseNLassoRegressor
from ..base import NLassoClassifier as BaseNLassoClassifier
from ..first_stage import construct_X_loo, calculate_asymmetric_weights
from ..group_module import group_variables, OrthogonalDecomposer, reconstruct_coefficients
from ..second_stage import cd_solve


class NLasso(BaseNLassoRegressor):
    """
    NLasso 主类（回归版本）
    完整实现paper中提出的三阶段算法：
    1. 第一阶段：强Ridge + LOO引导矩阵 + 自适应权重
    2. 组处理：高相关变量分组与正交分解
    3. 第二阶段：非对称Lasso求解
    """

    def _fit_first_stage(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        第一阶段实现：强Ridge回归 + LOO矩阵构造 + 权重计算
        """
        # 构造LOO引导矩阵，得到全样本Ridge系数
        X_loo, beta_ridge = construct_X_loo(
            X=X,
            y=y,
            lambda_ridge=self.lambda_ridge,
            task_type=self.task_type_,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose
        )

        # 计算非对称惩罚权重
        weights = calculate_asymmetric_weights(
            beta_ridge=beta_ridge,
            gamma=self.gamma,
            s=self.s
        )

        self.beta_ridge_ = beta_ridge
        self.X_loo_ = X_loo
        self.weights_ = weights

        return beta_ridge, X_loo, weights

    def _fit_group_module(self, X: np.ndarray, X_loo: np.ndarray) -> tuple:
        """
        组处理模块实现：分组 + 正交分解 + 变换LOO矩阵
        """
        n, p = X.shape

        # 1. 相关性分组
        self.groups_, self.corr_matrix_ = group_variables(
            X=X,
            threshold=self.group_threshold,
            min_group_size=self.group_min_size,
            max_group_size=self.group_max_size,
            verbose=self.verbose
        )

        # 2. 对每个组拟合正交分解器，变换LOO矩阵
        self.decomposers_ = []
        X_loo_transformed_parts = []

        for group in self.groups_:
            X_group_loo = X_loo[:, group]
            k = len(group)

            if k == 1:
                # 单变量组无需分解，直接保留
                decomposer = OrthogonalDecomposer()
                decomposer.fit(X_group_loo)
                X_transformed = decomposer.transform(X_group_loo)
            else:
                # 高相关组进行正交分解
                decomposer = OrthogonalDecomposer(random_state=self.random_state)
                X_transformed = decomposer.fit_transform(X_group_loo)

            self.decomposers_.append(decomposer)
            X_loo_transformed_parts.append(X_transformed)

        # 3. 拼接所有变换后的分量，得到最终的X_loo_transformed
        X_loo_transformed = np.column_stack(X_loo_transformed_parts)

        # 保存变换信息用于系数还原
        self.transform_info_ = {
            'groups': self.groups_,
            'decomposers': self.decomposers_,
            'p_original': p
        }

        return X_loo_transformed, self.transform_info_

    def _fit_second_stage(self, X_loo_transformed: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        第二阶段实现：非对称Lasso坐标下降求解
        实现组变换后的权重映射逻辑（paper 3.4.4节）
        """
        # 1. 权重映射：将原始特征权重映射到变换后的特征空间
        transformed_weights = []
        for group, decomposer in zip(self.groups_, self.decomposers_):
            k = len(group)
            if k == 1:
                # 单变量组：直接复用原始权重
                w = weights[group[0]]
                transformed_weights.append(w.reshape(1, 2))
            else:
                # 多变量组：
                m = decomposer.m_  # 共同趋势分量数
                group_weights = weights[group]  # 组内原始特征权重 (k, 2)

                # a. 共同趋势分量权重：取组内最小权重（惩罚最轻，优先保留共同信号）
                trend_weight = np.min(group_weights, axis=0)
                trend_weights = np.tile(trend_weight, (m, 1))  # (m, 2)

                # b. 细节分量权重：与原始特征权重保持一致
                detail_weights = group_weights  # (k, 2)

                # 合并组内权重
                group_transformed_weights = np.vstack([trend_weights, detail_weights])
                transformed_weights.append(group_transformed_weights)

        # 拼接所有组的权重，得到变换后特征的权重矩阵 (p_transformed, 2)
        weights_transformed = np.vstack(transformed_weights)
        self.weights_transformed_ = weights_transformed

        # Lambda参数处理
        if self.lambda_ is not None:
            # 用户指定lambda，直接使用
            lambdas = [self.lambda_]
        else:
            # 自动生成lambda搜索路径：对数空间从lambda_max到lambda_min
            if self.lambda_path is not None:
                lambdas = self.lambda_path
            else:
                # 计算lambda_max：所有特征的最大相关性
                XtY = np.abs(X_loo_transformed.T @ y) / len(y)
                lambda_max = np.max(XtY) / np.min(weights_transformed)
                lambda_min = lambda_max * 1e-4
                lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), self.n_lambda)[::-1]

        # 求解每个lambda的解，选择最优（目前用最小目标函数值，后续加入CV）
        best_obj = np.inf
        best_theta = None
        best_lambda = None
        best_n_iter = None

        for i, lam in enumerate(lambdas):
            if self.verbose and len(lambdas) > 1:
                print(f"[Stage 3] Solving lambda {i+1}/{len(lambdas)}: {lam:.6f}")

            theta, obj, n_iter = cd_solve(
                X=X_loo_transformed,
                y=y,
                weights=weights_transformed,
                lambda_=lam,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose and len(lambdas) == 1
            )

            # 安全检查：如果obj是NaN，尝试用之前的解或者初始化为零
            if np.isnan(obj):
                if best_theta is not None:
                    theta = best_theta.copy()
                    obj = best_obj
                else:
                    theta = np.zeros(X_loo_transformed.shape[1], dtype=np.float64)
                    obj = np.sum(y ** 2) / (2 * len(y))  # 零系数的目标值

            # 更新最优解
            if best_theta is None or obj < best_obj:
                best_obj = obj
                best_theta = theta
                best_lambda = lam
                best_n_iter = n_iter

        # 双重保险：确保返回有效的theta
        if best_theta is None:
            best_theta = np.zeros(X_loo_transformed.shape[1], dtype=np.float64)
            best_lambda = lambdas[0] if len(lambdas) > 0 else 0.1
            best_obj = np.sum(y ** 2) / (2 * len(y))
            best_n_iter = 0

        self.lambda_ = best_lambda
        self.obj_ = best_obj
        self.n_iter_ = best_n_iter
        self.lambda_path_ = lambdas
        theta = best_theta

        self.theta_ = theta
        return theta

    def _reconstruct_coefficients(self, theta: np.ndarray, transform_info: dict) -> np.ndarray:
        """
        系数还原实现：从变换空间映射回原始特征空间
        根据论文：最终系数 = beta_ridge * theta（element-wise乘积）
        """
        beta_transformed = reconstruct_coefficients(
            theta_transformed=theta,
            groups=transform_info['groups'],
            decomposers=transform_info['decomposers'],
            p_original=transform_info['p_original'],
            group_truncation_threshold=self.group_truncation_threshold,
            epsilon=1e-6
        )

        # 按照UniLasso/NLasso论文：最终系数是 beta_ridge * theta（逐元素相乘）
        beta_original = self.beta_ridge_ * beta_transformed

        return beta_original


class NLassoClassifier(NLasso):
    """
    NLasso 分类版本，复用NLasso回归版本的实现
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_type_ = 'classification'
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        # 分类任务额外处理类别
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("NLassoClassifier currently only supports binary classification")
        # 将y转为0/1编码
        y_continuous = (y == self.classes_[1]).astype(np.float64)
        return super().fit(X, y_continuous, sample_weight)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率（二分类）
        Returns: (n_samples, 2) -> [P(0), P(1)]
        """
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, 'is_fitted_')
        # 直接调用NLasso的predict方法，不要用super()避免递归
        z = NLasso.predict(self, X)
        # 数值稳定sigmoid
        proba_1 = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - proba_1, proba_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        分类任务默认用准确率评分
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)
