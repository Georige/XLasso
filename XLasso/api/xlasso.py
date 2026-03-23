"""
API层：XLasso主类
整合所有模块实现完整算法流程
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted
from ..base import XLassoRegressor as BaseXLassoRegressor
from ..base import XLassoClassifier as BaseXLassoClassifier
from ..first_stage import construct_X_loo, calculate_asymmetric_weights
from ..group_module import group_variables, OrthogonalDecomposer, reconstruct_coefficients
from ..second_stage import cd_solve


class XLasso(BaseXLassoRegressor):
    """
    XLasso 主类（回归版本）
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

        # TODO: 支持lambda路径搜索与交叉验证选择最优lambda
        # 目前默认使用lambda=0.1，后续会加入CV自动调参
        self.lambda_ = 0.1

        theta, self.obj_, self.n_iter_ = cd_solve(
            X=X_loo_transformed,
            y=y,
            weights=weights_transformed,
            lambda_=self.lambda_,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose
        )

        self.theta_ = theta
        return theta

    def _reconstruct_coefficients(self, theta: np.ndarray, transform_info: dict) -> np.ndarray:
        """
        系数还原实现：从变换空间映射回原始特征空间
        """
        beta_original = reconstruct_coefficients(
            theta_transformed=theta,
            groups=transform_info['groups'],
            decomposers=transform_info['decomposers'],
            p_original=transform_info['p_original'],
            group_truncation_threshold=0.5,
            epsilon=1e-6
        )

        return beta_original


class XLassoClassifier(BaseXLassoClassifier):
    """
    XLasso 分类版本
    """
    def _fit_first_stage(self, X: np.ndarray, y: np.ndarray) -> tuple:
        # 分类任务第一阶段：二分类y转为0/1连续值
        y_continuous = y.astype(np.float64)
        return super()._fit_first_stage(X, y_continuous)

    def _fit_group_module(self, X: np.ndarray, X_loo: np.ndarray) -> tuple:
        return super()._fit_group_module(X, X_loo)

    def _fit_second_stage(self, X_loo_transformed: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return super()._fit_second_stage(X_loo_transformed, y, weights)

    def _reconstruct_coefficients(self, theta: np.ndarray, transform_info: dict) -> np.ndarray:
        return super()._reconstruct_coefficients(theta, transform_info)
