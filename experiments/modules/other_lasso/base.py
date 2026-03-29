"""
Lasso算法公共基类
兼容sklearn接口规范，方便后续对比实验
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler

class BaseLasso(BaseEstimator):
    """Lasso算法基类"""
    def __init__(self, alpha=1.0, fit_intercept=True, standardize=True, max_iter=5000, tol=1e-4, family="gaussian"):
        """
        初始化参数
        Parameters:
            alpha: 正则化强度，越大越稀疏
            fit_intercept: 是否拟合截距
            standardize: 是否标准化特征
            max_iter: 最大迭代次数
            tol: 收敛阈值
            family: 模型家族，支持"gaussian"(回归)或"binomial"(二分类)
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.max_iter = max_iter
        self.tol = tol
        self.family = family  # 保存原始参数，支持sklearn clone
        family_lower = family.lower()
        if family_lower not in ["gaussian", "binomial"]:
            raise ValueError(f"Unsupported family: {family}. Use 'gaussian' or 'binomial'.")

        # 告诉sklearn模型类型，用于评分
        if family_lower == "gaussian":
            self._estimator_type = "regressor"
        else:
            self._estimator_type = "classifier"
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def _preprocess(self, X, y=None):
        """数据预处理：标准化"""
        X = check_array(X)
        if y is not None:
            y = check_array(y, ensure_2d=False)

        if self.standardize:
            X = self.scaler_X.fit_transform(X)
            if y is not None and self.fit_intercept and self.family.lower() == "gaussian":
                # 只有回归任务需要标准化y，二分类不需要
                y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        if self.fit_intercept and not self.standardize and self.family.lower() == "gaussian":
            # 只有回归任务需要中心化y，二分类不需要
            X = X - np.mean(X, axis=0)
            if y is not None:
                y = y - np.mean(y)

        return X, y

    def _postprocess(self, coef_, intercept_=0.0):
        """逆标准化，还原到原始尺度"""
        if self.standardize:
            # 还原系数
            coef_ = coef_ / self.scaler_X.scale_
            # 还原截距
            if self.fit_intercept:
                if self.family.lower() == "gaussian":
                    intercept_ = self.scaler_y.mean_ - np.sum(coef_ * self.scaler_X.mean_)
                else:  # binomial
                    intercept_ = intercept_ - np.sum(coef_ * self.scaler_X.mean_)
        else:
            if self.fit_intercept and self.family.lower() == "gaussian":
                intercept_ = self.y_mean_ - np.sum(coef_ * self.X_mean_)

        self.coef_ = coef_
        self.intercept_ = intercept_
        return coef_, intercept_

    def predict(self, X):
        """预测"""
        check_is_fitted(self)
        X = check_array(X)

        if self.standardize:
            X = self.scaler_X.transform(X)

        z = X @ self.coef_ + self.intercept_

        if self.family.lower() == "binomial":
            # 二分类返回类别标签
            return (z >= 0).astype(int)
        else:
            # 回归返回预测值
            return z

    def predict_proba(self, X):
        """二分类预测概率，仅支持binomial家族"""
        if self.family != "binomial":
            raise ValueError("predict_proba only available for family='binomial'")

        check_is_fitted(self)
        X = check_array(X)

        if self.standardize:
            X = self.scaler_X.transform(X)

        z = X @ self.coef_ + self.intercept_
        prob = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - prob, prob])

    def score(self, X, y, sample_weight=None):
        """
        返回模型的评分，回归用R²，分类用准确率
        """
        from sklearn.metrics import r2_score, accuracy_score
        y_pred = self.predict(X)
        if self.family.lower() == "gaussian":
            return r2_score(y, y_pred, sample_weight=sample_weight)
        else:
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
