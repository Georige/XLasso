"""
统一接口的 Lasso Benchmark 框架
支持本地实现和 skglm 高性能算法

Benchmark 维度：
- 精度: MSE, MAE, R², F1, 系数误差平方和(SSE), 准确率, AUC, TPR, FDR
- 稀疏性: 非零系数比例
- 运行时间: 训练/预测时间
- 收敛性: 迭代次数/残差下降
"""
import time
import copy
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler


class UnifiedMetrics:
    """统一评估指标计算"""

    @staticmethod
    def mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def r2(y_true, y_pred):
        return r2_score(y_true, y_pred)

    @staticmethod
    def sse(y_true, y_pred, coef_true=None, coef_est=None):
        """Sum of Squared Errors of estimates"""
        if coef_true is not None and coef_est is not None:
            # 系数误差平方和
            return np.sum((coef_true - coef_est) ** 2)
        return np.sum((y_true - y_pred) ** 2)

    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def auc(y_true, y_pred_proba):
        """AUC score, 需要概率预测"""
        try:
            return roc_auc_score(y_true, y_pred_proba)
        except:
            return np.nan

    @staticmethod
    def tpr(y_true, y_pred):
        """True Positive Rate = Recall = TP / (TP + FN)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else np.nan

    @staticmethod
    def fdr(y_true, y_pred):
        """False Discovery Rate = FP / (FP + TP)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fp / (fp + tp) if (fp + tp) > 0 else np.nan

    @staticmethod
    def f1(y_true, y_pred):
        return f1_score(y_true, y_pred, zero_division=0)

    @staticmethod
    def sparsity(coef_):
        """非零系数比例"""
        return np.mean(coef_ != 0)

    @staticmethod
    def n_nonzero(coef_):
        """非零系数个数"""
        return np.sum(coef_ != 0)


class AlgorithmWrapper(BaseEstimator):
    """
    统一算法接口的 Wrapper
    自动识别分类/回归任务，统一输出格式
    """

    def __init__(self, algorithm_class, is_classification=False, **kwargs):
        self.algorithm_class = algorithm_class
        self.is_classification = is_classification
        self.kwargs = kwargs
        self.model_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None

    def fit(self, X, y, **fit_kwargs):
        self.model_ = self.algorithm_class(**self.kwargs)

        # 记录训练前的迭代属性
        if hasattr(self.model_, 'n_iter'):
            self.n_iter_before_ = self.model_.n_iter

        self.model_.fit(X, y, **fit_kwargs)

        # 统一提取 coef_ 和 intercept_
        if hasattr(self.model_, 'coef_'):
            coef = self.model_.coef_
            self.coef_ = coef.flatten() if coef.ndim > 1 else np.asarray(coef)
        if hasattr(self.model_, 'intercept_'):
            intercept = self.model_.intercept_
            if np.isscalar(intercept):
                self.intercept_ = intercept
            elif hasattr(intercept, 'item'):
                self.intercept_ = intercept.item()
            else:
                self.intercept_ = np.asarray(intercept).item()

        # 获取迭代次数
        if hasattr(self.model_, 'n_iter_'):
            self.n_iter_ = self.model_.n_iter_
        elif hasattr(self.model_, 'n_iter'):
            self.n_iter_ = self.model_.n_iter

        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        """如果模型支持概率预测"""
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X)
        elif hasattr(self.model_, 'predict_log_proba'):
            return np.exp(self.model_.predict_log_proba(X))
        else:
            # 退回普通预测
            pred = self.predict(X)
            # 转换为二分类概率形式
            if self.is_classification:
                return np.column_stack([1 - pred, pred])
            return pred

    @property
    def _estimator_type(self):
        return 'classifier' if self.is_classification else 'regressor'


def create_wrapper(algorithm_class, is_classification_or_kwargs=None, is_classification=False, **kwargs):
    """创建算法 Wrapper"""
    class AutoWrapper(AlgorithmWrapper):
        pass
    AutoWrapper.__name__ = f"Wrapper_{algorithm_class.__name__}"
    # 兼容两种调用方式: (class, is_classification, kwargs) 或 (class, kwargs, is_classification)
    if isinstance(is_classification_or_kwargs, bool):
        return AutoWrapper(algorithm_class, is_classification_or_kwargs, **kwargs)
    elif isinstance(is_classification_or_kwargs, dict):
        return AutoWrapper(algorithm_class, is_classification, **is_classification_or_kwargs)
    else:
        return AutoWrapper(algorithm_class, is_classification, **kwargs)


class BenchmarkRunner:
    """Benchmark 运行器"""

    def __init__(self, algorithms: dict, X, y, coef_true=None,
                 metrics=None, cv=5, random_state=42, task_type='auto'):
        """
        Parameters:
            algorithms: dict of {name: (model_or_class, kwargs)} 待评测算法
            X: 特征矩阵
            y: 目标向量
            coef_true: 真实系数向量 (用于计算 SSE)
            metrics: list of metric names
            cv: 交叉验证折数
            random_state: 随机种子
            task_type: 'regression', 'classification', 'auto'
        """
        self.algorithms = algorithms
        self.X = X
        self.y = y
        self.coef_true = coef_true
        self.cv = cv
        self.random_state = random_state
        self.results_ = None

        # 自动识别任务类型
        if task_type == 'auto':
            if len(np.unique(y)) == 2:
                self.task_type = 'classification'
            else:
                self.task_type = 'regression'
        else:
            self.task_type = task_type

        # 默认指标
        if metrics is None:
            if self.task_type == 'regression':
                metrics = ['mse', 'mae', 'r2', 'sse', 'sparsity', 'n_nonzero', 'time', 'n_iter']
            else:
                metrics = ['accuracy', 'auc', 'tpr', 'fdr', 'f1', 'sparsity', 'n_nonzero', 'time', 'n_iter']
        self.metrics = metrics

    def _is_classification(self, name):
        """判断算法是否为分类器"""
        algo = self.algorithms[name]
        if isinstance(algo, tuple):
            return algo[0].__name__.lower().find('logistic') >= 0 or \
                   algo[0].__name__.lower().find('classifier') >= 0
        return False

    def run_single(self, name, model, X_train, y_train, X_test, y_test):
        """运行单个算法"""
        result = {'algorithm': name}

        # 训练计时
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        # 预测
        y_pred = model.predict(X_test)

        # 计算指标
        if 'mse' in self.metrics and self.task_type == 'regression':
            result['mse'] = UnifiedMetrics.mse(y_test, y_pred)
        if 'mae' in self.metrics and self.task_type == 'regression':
            result['mae'] = UnifiedMetrics.mae(y_test, y_pred)
        if 'r2' in self.metrics and self.task_type == 'regression':
            result['r2'] = UnifiedMetrics.r2(y_test, y_pred)
        if 'sse' in self.metrics:
            result['sse'] = UnifiedMetrics.sse(y_test, y_pred,
                                              coef_true=self.coef_true,
                                              coef_est=model.coef_)
        if 'accuracy' in self.metrics and self.task_type == 'classification':
            result['accuracy'] = UnifiedMetrics.accuracy(y_test, y_pred)
        if 'f1' in self.metrics and self.task_type == 'classification':
            result['f1'] = UnifiedMetrics.f1(y_test, y_pred)

        # 概率预测指标
        if self.task_type == 'classification':
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                if 'auc' in self.metrics:
                    result['auc'] = UnifiedMetrics.auc(y_test, y_proba)
                if 'tpr' in self.metrics:
                    result['tpr'] = UnifiedMetrics.tpr(y_test, y_pred)
                if 'fdr' in self.metrics:
                    result['fdr'] = UnifiedMetrics.fdr(y_test, y_pred)
            except:
                pass

        # 稀疏性指标
        if 'sparsity' in self.metrics:
            result['sparsity'] = UnifiedMetrics.sparsity(model.coef_)
        if 'n_nonzero' in self.metrics:
            result['n_nonzero'] = UnifiedMetrics.n_nonzero(model.coef_)

        # 效率指标
        if 'time' in self.metrics:
            result['train_time'] = train_time
        if 'n_iter' in self.metrics and hasattr(model, 'n_iter_'):
            result['n_iter'] = model.n_iter_ or 0
        elif 'n_iter' in self.metrics:
            result['n_iter'] = getattr(model, 'n_iter_', 0)

        return result

    def run_cv(self):
        """交叉验证运行"""
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        all_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.X)):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            for name, algo in self.algorithms.items():
                # 深拷贝模型
                if isinstance(algo, tuple):
                    if len(algo) == 3:
                        algo_class, algo_kwargs, algo_is_clf = algo
                        model = create_wrapper(algo_class, algo_kwargs, algo_is_clf)
                    elif len(algo) == 2:
                        algo_class, algo_kwargs = algo
                        model = create_wrapper(algo_class, algo_kwargs)
                    else:
                        model = copy.deepcopy(algo)
                else:
                    model = copy.deepcopy(algo)

                result = self.run_single(name, model, X_train, y_train, X_test, y_test)
                result['fold'] = fold_idx
                all_results.append(result)

        self.results_ = pd.DataFrame(all_results)
        return self.results_

    def summary(self):
        """生成汇总表格"""
        if self.results_ is None:
            self.run_cv()

        # 按算法分组计算均值和标准差
        summary_cols = [c for c in self.results_.columns if c not in ['algorithm', 'fold']]

        def agg_func(x):
            return pd.Series({
                'mean': x.mean(),
                'std': x.std(),
                'min': x.min(),
                'max': x.max()
            })

        summary = self.results_.groupby('algorithm')[summary_cols].apply(agg_func)
        summary = summary.round(4)

        return summary

    def run(self):
        """运行完整 benchmark 并返回汇总"""
        self.run_cv()
        return self.summary()


def run_lasso_benchmark(X, y, coef_true=None, algorithms=None, cv=5, random_state=42):
    """
    运行 Lasso 算法 Benchmark

    Parameters:
        X: 特征矩阵
        y: 目标向量
        coef_true: 真实系数 (可选)
        algorithms: dict of {name: (class, kwargs, is_classification)} 自定义算法
        cv: 交叉验证折数
        random_state: 随机种子

    Returns:
        DataFrame with summary statistics
    """
    # 默认算法列表
    if algorithms is None:
        from sklearn.linear_model import Lasso as SklearnLasso, LogisticRegression
        from other_lasso import AdaptiveLasso, FusedLasso, GroupLasso

        # 确定任务类型
        task_type = 'classification' if len(np.unique(y)) == 2 else 'regression'
        print(f"Task type: {task_type}")

        algorithms = {}

        if task_type == 'regression':
            # sklearn Lasso
            algorithms['Sklearn_Lasso'] = (SklearnLasso, {'alpha': 1.0, 'max_iter': 1000, 'tol': 1e-4}, False)

            # Adaptive Lasso
            algorithms['Adaptive_Lasso'] = (AdaptiveLasso, {'alpha': 1.0, 'gamma': 1.0, 'max_iter': 1000, 'tol': 1e-4}, False)

            # Fused Lasso (cvxpy)
            algorithms['Fused_Lasso'] = (FusedLasso, {'alpha': 1.0, 'lambda_fused': 1.0, 'max_iter': 1000, 'tol': 1e-4}, False)

            # Group Lasso (cvxpy)
            algorithms['Group_Lasso'] = (GroupLasso, {'alpha': 1.0, 'groups': None, 'max_iter': 1000, 'tol': 1e-4}, False)

            # skglm Lasso
            try:
                import sys
                sys.path.insert(0, 'other_lasso/skglm_benchmark')
                from skglm import Lasso as SkglmLasso
                algorithms['skglm_Lasso'] = (SkglmLasso, {'alpha': 1.0, 'max_iter': 50, 'tol': 1e-4}, False)
            except ImportError as e:
                print(f"Warning: skglm not available ({e})")

        else:  # classification
            # sklearn LogisticRegression
            algorithms['Sklearn_Logistic'] = (LogisticRegression, {'C': 1.0, 'max_iter': 1000, 'tol': 1e-4}, True)

            # Adaptive Logistic (L1 penalized)
            from other_lasso import AdaptiveLasso
            algorithms['Adaptive_Logistic'] = (AdaptiveLasso, {'alpha': 1.0, 'gamma': 1.0, 'max_iter': 1000, 'tol': 1e-4, 'family': 'binomial'}, True)

            # sklearn Lasso for classification
            algorithms['Sklearn_Lasso'] = (SklearnLasso, {'alpha': 1.0, 'max_iter': 1000, 'tol': 1e-4}, True)

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 运行 Benchmark
    runner = BenchmarkRunner(
        algorithms=algorithms,
        X=X_scaled,
        y=y,
        coef_true=coef_true,
        cv=cv,
        random_state=random_state,
        task_type=task_type
    )

    print(f"\nRunning benchmark with {len(algorithms)} algorithms, cv={cv}...")
    summary = runner.run()
    return summary


def run_simulated_benchmark(n_samples=200, n_features=50, n_informative=10,
                             n_groups=5, noise=1.0, random_state=42):
    """
    使用模拟数据运行 Benchmark
    生成稀疏系数用于计算 SSE
    """
    from sklearn.datasets import make_regression

    print("Generating simulated data...")
    X, y, coef_true = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
        coef=True
    )

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"True coef: {n_informative} non-zero out of {n_features}")

    return run_lasso_benchmark(X, y, coef_true=coef_true, cv=5, random_state=random_state)


if __name__ == '__main__':
    import sys
    from pathlib import Path

    # 添加项目根目录到路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    print("=" * 80)
    print("LASSO BENCHMARK - Unified Interface")
    print("=" * 80)

    # 1. 模拟数据 Benchmark
    print("\n" + "=" * 40)
    print("1. Simulated Data Benchmark")
    print("=" * 40)
    results = run_simulated_benchmark(
        n_samples=200, n_features=50, n_informative=10,
        noise=1.0, random_state=42
    )
    print("\n" + "-" * 40)
    print("Results:")
    print("-" * 40)
    print(results.to_string())

    # 2. 糖尿病数据集 Benchmark
    print("\n" + "=" * 40)
    print("2. Diabetes Dataset Benchmark")
    print("=" * 40)
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    results_diabetes = run_lasso_benchmark(X, y, cv=5, random_state=42)
    print("\n" + "-" * 40)
    print("Results:")
    print("-" * 40)
    print(results_diabetes.to_string())

    # 3. 二分类数据集 Benchmark
    print("\n" + "=" * 40)
    print("3. Binary Classification Benchmark")
    print("=" * 40)
    from sklearn.datasets import make_classification

    X_clf, y_clf = make_classification(
        n_samples=200, n_features=50, n_informative=10,
        n_redundant=0, n_clusters_per_class=2,
        random_state=42, flip_y=0.1
    )
    print(f"X shape: {X_clf.shape}, y shape: {y_clf.shape}")
    print(f"Class distribution: {np.bincount(y_clf)}")

    results_clf = run_lasso_benchmark(X_clf, y_clf, cv=5, random_state=42)
    print("\n" + "-" * 40)
    print("Results:")
    print("-" * 40)
    print(results_clf.to_string())
