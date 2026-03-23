#!/usr/bin/env python3
"""
NLasso 基础测试用例
验证核心算法在各种场景下的正确性
"""
import sys
sys.path.insert(0, '/home/liangyuneng/XLasso')

import numpy as np
import unittest


class TestNLassoBasic(unittest.TestCase):
    """NLasso 基础功能测试"""

    def setUp(self):
        """测试前设置随机种子保证可复现"""
        np.random.seed(2026)

    def test_imports(self):
        """测试模块导入"""
        from NLasso import NLasso, NLassoClassifier, NLassoCV, NLassoClassifierCV, metrics
        self.assertTrue(NLasso is not None)
        self.assertTrue(NLassoClassifier is not None)
        self.assertTrue(NLassoCV is not None)
        self.assertTrue(NLassoClassifierCV is not None)
        self.assertTrue(metrics is not None)

    def test_regression_simple(self):
        """测试简单回归任务"""
        from NLasso import NLasso

        n, p = 100, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)

        model = NLasso(
            lambda_ridge=10.0,
            lambda_=0.1,
            gamma=0.3,
            max_iter=100,
            verbose=False
        )

        model.fit(X, y)
        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'intercept_'))
        self.assertEqual(model.coef_.shape, (p,))

        y_pred = model.predict(X)
        self.assertEqual(y_pred.shape, (n,))

        score = model.score(X, y)
        self.assertTrue(isinstance(score, float))

    def test_classification_simple(self):
        """测试简单分类任务"""
        from NLasso import NLassoClassifier

        n, p = 100, 10
        X = np.random.randn(n, p)
        y = np.random.randint(0, 2, size=n)

        model = NLassoClassifier(
            lambda_ridge=10.0,
            lambda_=0.1,
            gamma=0.3,
            max_iter=100,
            verbose=False
        )

        model.fit(X, y)
        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'intercept_'))
        self.assertTrue(hasattr(model, 'classes_'))
        self.assertEqual(model.coef_.shape, (p,))

        y_pred = model.predict(X)
        self.assertEqual(y_pred.shape, (n,))

        y_proba = model.predict_proba(X)
        self.assertEqual(y_proba.shape, (n, 2))

        score = model.score(X, y)
        self.assertTrue(isinstance(score, float))

    def test_regression_cv(self):
        """测试NLassoCV回归"""
        from NLasso import NLassoCV

        n, p = 50, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)

        param_grid = {
            'lambda_ridge': [5.0, 10.0],
            'gamma': [0.3],
            'lambda_': [0.1]
        }

        model_cv = NLassoCV(
            param_grid=param_grid,
            cv=2,
            scoring='r2',
            n_jobs=1,
            verbose=False
        )

        model_cv.fit(X, y)
        self.assertTrue(hasattr(model_cv, 'best_params_'))
        self.assertTrue(hasattr(model_cv, 'best_score_'))
        self.assertTrue(hasattr(model_cv, 'coef_'))

    def test_classification_cv(self):
        """测试NLassoClassifierCV分类"""
        from NLasso import NLassoClassifierCV

        n, p = 50, 10
        X = np.random.randn(n, p)
        y = np.random.randint(0, 2, size=n)

        param_grid = {
            'lambda_ridge': [5.0, 10.0],
            'gamma': [0.3],
            'lambda_': [0.1]
        }

        model_cv = NLassoClassifierCV(
            param_grid=param_grid,
            cv=2,
            scoring='accuracy',
            n_jobs=1,
            verbose=False
        )

        model_cv.fit(X, y)
        self.assertTrue(hasattr(model_cv, 'best_params_'))
        self.assertTrue(hasattr(model_cv, 'best_score_'))
        self.assertTrue(hasattr(model_cv, 'coef_'))

    def test_group_truncation(self):
        """测试组感知截断功能"""
        from NLasso import NLasso

        n, p = 100, 20
        # 创建高相关变量组
        X = np.random.randn(n, p)
        # 前4个变量高度相关
        X[:, 1] = X[:, 0] * 0.9 + np.random.randn(n) * 0.1
        X[:, 2] = X[:, 0] * 0.8 + np.random.randn(n) * 0.2
        X[:, 3] = X[:, 0] * 0.7 + np.random.randn(n) * 0.3
        y = X[:, 0] + X[:, 1] + np.random.randn(n) * 0.5

        # 启用组截断
        model_with_trunc = NLasso(
            lambda_ridge=10.0,
            lambda_=0.1,
            gamma=0.3,
            group_threshold=0.6,
            group_truncation_threshold=0.5,
            max_iter=100,
            verbose=False
        )
        model_with_trunc.fit(X, y)

        # 禁用组截断
        model_without_trunc = NLasso(
            lambda_ridge=10.0,
            lambda_=0.1,
            gamma=0.3,
            group_threshold=0.6,
            group_truncation_threshold=0.0,
            max_iter=100,
            verbose=False
        )
        model_without_trunc.fit(X, y)

        # 两种模式都应该能运行
        self.assertTrue(hasattr(model_with_trunc, 'coef_'))
        self.assertTrue(hasattr(model_without_trunc, 'coef_'))

    def test_metrics_module(self):
        """测试指标计算模块"""
        from NLasso import metrics

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.2, 1.8, 3.1, 3.9, 5.2])
        beta_true = np.array([1.0, 0.0, 2.0, 0.0, 0.0])
        beta_pred = np.array([0.9, 0.0, 1.8, 0.1, 0.0])

        mse = metrics.mean_squared_error(y_true, y_pred)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        sparsity = metrics.sparsity(beta_pred)
        tpr = metrics.true_positive_rate(None, beta_pred, beta_true)
        fdr = metrics.false_discovery_rate(None, beta_pred, beta_true)

        self.assertTrue(isinstance(mse, float))
        self.assertTrue(isinstance(mae, float))
        self.assertTrue(isinstance(r2, float))
        self.assertTrue(isinstance(sparsity, float))
        self.assertTrue(isinstance(tpr, float))
        self.assertTrue(isinstance(fdr, float))


class TestNLassoOnSimulatedData(unittest.TestCase):
    """在模拟数据上测试NLasso"""

    def test_sparse_recovery(self):
        """测试稀疏信号恢复能力"""
        from NLasso import NLasso

        np.random.seed(2026)
        n, p = 200, 50
        X = np.random.randn(n, p)

        # 真实系数：前5个非零
        beta_true = np.zeros(p)
        beta_true[:5] = [1.5, -1.2, 2.0, -0.8, 1.0]

        y = X @ beta_true + np.random.randn(n) * 0.5

        model = NLasso(
            lambda_ridge=5.0,
            lambda_=0.05,
            gamma=0.5,
            max_iter=500,
            verbose=False
        )

        model.fit(X, y)

        # 检查是否能恢复大部分信号
        non_zero_in_true = np.sum(beta_true != 0)
        non_zero_in_pred = np.sum(np.abs(model.coef_) > 1e-3)

        # 至少应该选出一些非零系数
        self.assertTrue(non_zero_in_pred > 0)


if __name__ == '__main__':
    print("="*60)
    print("🧪 开始NLasso基础测试")
    print("="*60 + "\n")

    unittest.main(verbosity=2)
