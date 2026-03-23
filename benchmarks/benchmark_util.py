"""
Benchmark 工具库
包含统一接口、指标计算、结果保存等核心功能
"""
import os
import time
import copy
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, train_test_split
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
    def coef_sse(coef_true, coef_est):
        """拟合系数与真实系数的误差平方和"""
        if coef_true is None or coef_est is None:
            return np.nan
        return np.sum((coef_true - coef_est) ** 2)

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
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tp / (tp + fn) if (tp + fn) > 0 else np.nan
        except:
            return np.nan

    @staticmethod
    def fdr(y_true, y_pred):
        """False Discovery Rate = FP / (FP + TP)"""
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return fp / (fp + tp) if (fp + tp) > 0 else np.nan
        except:
            return np.nan

    @staticmethod
    def f1(y_true, y_pred):
        return f1_score(y_true, y_pred, zero_division=0)

    @staticmethod
    def sparsity(coef_):
        """非零系数比例"""
        return np.mean(coef_ != 0) if coef_ is not None else np.nan

    @staticmethod
    def n_nonzero(coef_):
        """非零系数个数"""
        return np.sum(coef_ != 0) if coef_ is not None else np.nan


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
        self.train_time_ = None

    def fit(self, X, y, **fit_kwargs):
        self.model_ = self.algorithm_class(**self.kwargs)

        # 记录训练时间
        start_time = time.time()
        self.model_.fit(X, y, **fit_kwargs)
        self.train_time_ = time.time() - start_time

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
        start_time = time.time()
        pred = self.model_.predict(X)
        self.predict_time_ = time.time() - start_time
        return pred

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


def create_wrapper(algorithm_class, is_classification=False, **kwargs):
    """创建算法 Wrapper"""
    class AutoWrapper(AlgorithmWrapper):
        pass
    AutoWrapper.__name__ = f"Wrapper_{algorithm_class.__name__}"
    return AutoWrapper(algorithm_class, is_classification, **kwargs)


class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def convert_numpy_types(obj):
    """将字典中的numpy类型转换为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


class ResultSaver:
    """结果保存器"""

    def __init__(self, root_dir="./result", experiment_name=None):
        """
        Parameters:
            root_dir: 结果根目录
            experiment_name: 实验名称，默认用时间戳
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # 生成实验文件夹名
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"
        self.experiment_dir = self.root_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        # 保存实验配置
        self.config = {}

    def save_config(self, config):
        """保存实验配置"""
        self.config = config
        with open(self.experiment_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    def save_algorithm_results(self, algo_name, algo_params, raw_results, summary):
        """
        保存单个算法的结果

        Parameters:
            algo_name: 算法名称
            algo_params: 算法参数字典
            raw_results: 原始结果 DataFrame
            summary: 汇总结果 DataFrame
        """
        # 生成算法文件夹名
        params_str = "_".join([f"{k}={v}" for k, v in algo_params.items() if k != "is_classification"])
        if len(params_str) > 50:
            params_str = params_str[:50] + "..."
        algo_dir_name = f"{algo_name}_{params_str}".replace("/", "_").replace(":", "_")
        algo_dir = self.experiment_dir / algo_dir_name
        algo_dir.mkdir(exist_ok=True)

        # 保存原始结果和汇总
        raw_results.to_csv(algo_dir / "raw.csv", index=False)
        summary.to_csv(algo_dir / "summary.csv")

        # 保存参数配置
        algo_params_converted = convert_numpy_types(algo_params)
        with open(algo_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(algo_params_converted, f, indent=2, ensure_ascii=False)

        return algo_dir

    def save_overall_summary(self, overall_summary):
        """保存所有算法的汇总对比"""
        overall_summary.to_csv(self.experiment_dir / "overall_summary.csv")
        return self.experiment_dir / "overall_summary.csv"


class BenchmarkRunner:
    """Benchmark 运行器"""

    def __init__(self, algorithms: dict, X, y, coef_true=None,
                 n_repeats=3, cv=5, random_seed_start=2026,
                 standardize=True, task_type='auto', verbose=True):
        """
        Parameters:
            algorithms: dict of {name: (class, kwargs, is_classification)} 待评测算法
            X: 特征矩阵
            y: 目标向量
            coef_true: 真实系数向量 (用于计算 SSE)
            n_repeats: 重复实验次数
            cv: 交叉验证折数，None表示只做train/test split
            random_seed_start: 随机种子起始值，每次重复+1
            standardize: 是否标准化数据
            task_type: 'regression', 'classification', 'auto'
            verbose: 是否打印日志
        """
        self.algorithms = algorithms
        self.X = X
        self.y = y
        self.coef_true = coef_true
        self.n_repeats = n_repeats
        self.cv = cv
        self.random_seed_start = random_seed_start
        self.standardize = standardize
        self.verbose = verbose
        self.results_ = None

        # 自动识别任务类型
        if task_type == 'auto':
            if len(np.unique(y)) == 2:
                self.task_type = 'classification'
            else:
                self.task_type = 'regression'
        else:
            self.task_type = task_type

        # 确定指标
        if self.task_type == 'regression':
            self.metrics = [
                'mse', 'mae', 'r2', 'coef_sse',
                'sparsity', 'n_nonzero',
                'train_time', 'predict_time', 'n_iter'
            ]
        else:
            self.metrics = [
                'accuracy', 'f1', 'auc', 'tpr', 'fdr', 'coef_sse',
                'sparsity', 'n_nonzero',
                'train_time', 'predict_time', 'n_iter'
            ]

        # 标准化数据
        if self.standardize:
            self.scaler_X = StandardScaler()
            self.X_scaled = self.scaler_X.fit_transform(X)
        else:
            self.X_scaled = X

        # 结果保存器
        self.result_saver = None

    def _run_single_fold(self, name, algo, X_train, y_train, X_test, y_test, random_state):
        """运行单个折的实验"""
        # 创建模型
        if isinstance(algo, tuple):
            if len(algo) == 3:
                algo_class, algo_kwargs, algo_is_clf = algo
                model = create_wrapper(algo_class, algo_is_clf, **algo_kwargs)
            elif len(algo) == 2:
                algo_class, algo_kwargs = algo
                model = create_wrapper(algo_class, **algo_kwargs)
            else:
                model = copy.deepcopy(algo)
        else:
            model = copy.deepcopy(algo)

        # 训练
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        predict_time = getattr(model, 'predict_time_', np.nan)

        # 计算指标
        result = {
            'algorithm': name,
            'random_state': random_state,
        }

        # 回归指标
        if self.task_type == 'regression':
            if 'mse' in self.metrics:
                result['mse'] = UnifiedMetrics.mse(y_test, y_pred)
            if 'mae' in self.metrics:
                result['mae'] = UnifiedMetrics.mae(y_test, y_pred)
            if 'r2' in self.metrics:
                result['r2'] = UnifiedMetrics.r2(y_test, y_pred)

        # 分类指标
        else:
            if 'accuracy' in self.metrics:
                result['accuracy'] = UnifiedMetrics.accuracy(y_test, y_pred)
            if 'f1' in self.metrics:
                result['f1'] = UnifiedMetrics.f1(y_test, y_pred)
            if 'auc' in self.metrics:
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    result['auc'] = UnifiedMetrics.auc(y_test, y_proba)
                except:
                    result['auc'] = np.nan
            if 'tpr' in self.metrics:
                result['tpr'] = UnifiedMetrics.tpr(y_test, y_pred)
            if 'fdr' in self.metrics:
                result['fdr'] = UnifiedMetrics.fdr(y_test, y_pred)

        # 通用指标
        if 'coef_sse' in self.metrics:
            result['coef_sse'] = UnifiedMetrics.coef_sse(self.coef_true, model.coef_)
        if 'sparsity' in self.metrics:
            result['sparsity'] = UnifiedMetrics.sparsity(model.coef_)
        if 'n_nonzero' in self.metrics:
            result['n_nonzero'] = UnifiedMetrics.n_nonzero(model.coef_)
        if 'train_time' in self.metrics:
            result['train_time'] = getattr(model, 'train_time_', np.nan)
        if 'predict_time' in self.metrics:
            result['predict_time'] = predict_time
        if 'n_iter' in self.metrics:
            result['n_iter'] = getattr(model, 'n_iter_', np.nan)

        return result

    def run_single_repeat(self, repeat_idx):
        """运行单次重复实验"""
        random_state = self.random_seed_start + repeat_idx
        all_results = []

        if self.verbose:
            print(f"  Repeat {repeat_idx+1}/{self.n_repeats}, random_seed={random_state}")

        if self.cv is None:
            # 单次 train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled, self.y, test_size=0.2, random_state=random_state
            )

            for name, algo in self.algorithms.items():
                result = self._run_single_fold(name, algo, X_train, y_train, X_test, y_test, random_state)
                result['repeat'] = repeat_idx
                result['fold'] = 0
                all_results.append(result)

        else:
            # 交叉验证
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=random_state)

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.X_scaled)):
                X_train, X_test = self.X_scaled[train_idx], self.X_scaled[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]

                for name, algo in self.algorithms.items():
                    result = self._run_single_fold(name, algo, X_train, y_train, X_test, y_test, random_state)
                    result['repeat'] = repeat_idx
                    result['fold'] = fold_idx
                    all_results.append(result)

        return all_results

    def run(self, save_results=True, experiment_name=None):
        """
        运行完整 Benchmark

        Parameters:
            save_results: 是否保存结果到文件
            experiment_name: 实验名称

        Returns:
            overall_summary: 所有算法的汇总结果
        """
        if self.verbose:
            print(f"Running benchmark: {len(self.algorithms)} algorithms, {self.n_repeats} repeats, cv={self.cv}")
            print(f"Task type: {self.task_type}")
            print(f"Metrics: {', '.join(self.metrics)}")

        all_results = []

        # 运行所有重复
        for repeat_idx in range(self.n_repeats):
            repeat_results = self.run_single_repeat(repeat_idx)
            all_results.extend(repeat_results)

        # 转换为 DataFrame
        self.results_ = pd.DataFrame(all_results)

        # 按算法汇总
        summary_cols = [c for c in self.results_.columns if c not in ['algorithm', 'repeat', 'fold', 'random_state']]
        summary = self.results_.groupby('algorithm')[summary_cols].agg([
            'mean', 'std', 'min', 'max'
        ]).round(4)

        # 保存结果
        if save_results:
            self.result_saver = ResultSaver(experiment_name=experiment_name)

            # 保存全局配置
            config = {
                'n_repeats': self.n_repeats,
                'cv': self.cv,
                'random_seed_start': self.random_seed_start,
                'standardize': self.standardize,
                'task_type': self.task_type,
                'metrics': self.metrics,
                'n_samples': self.X.shape[0],
                'n_features': self.X.shape[1],
                'algorithms': list(self.algorithms.keys()),
            }
            self.result_saver.save_config(config)

            # 保存每个算法的结果
            for algo_name in self.algorithms.keys():
                algo_results = self.results_[self.results_['algorithm'] == algo_name]
                algo_summary = summary.loc[algo_name]

                # 获取算法参数
                algo = self.algorithms[algo_name]
                if isinstance(algo, tuple):
                    if len(algo) == 3:
                        algo_params = algo[1] | {'is_classification': algo[2]}
                    else:
                        algo_params = algo[1]
                else:
                    algo_params = {}

                self.result_saver.save_algorithm_results(
                    algo_name, algo_params, algo_results, algo_summary
                )

            # 保存整体汇总
            overall_path = self.result_saver.save_overall_summary(summary)

            if self.verbose:
                print(f"\nResults saved to: {self.result_saver.experiment_dir}")
                print(f"Overall summary: {overall_path}")

        return summary

    def get_raw_results(self):
        """获取原始结果"""
        return self.results_


# 全局算法列表
def get_algorithm_list(task_type='regression'):
    """获取默认算法列表（全部使用CV自动调参版本）"""
    from sklearn.linear_model import LassoCV, LogisticRegressionCV
    from other_lasso import (
        AdaptiveLassoCV, FusedLassoCV, GroupLassoCV, AdaptiveSparseGroupLassoCV
    )

    algorithms = {}

    if task_type == 'regression':
        # sklearn 系列
        algorithms['Sklearn_LassoCV'] = (LassoCV, {
            'alphas': np.logspace(-4, 1, 50), 'cv': 3,
            'max_iter': 1000, 'tol': 1e-4, 'n_jobs': -1,
            'random_state': 2026
        }, False)

        # 本地CV实现（自动调参）- 暂时注释，后续优化后再开启
        # algorithms['Adaptive_LassoCV'] = (AdaptiveLassoCV, {
        #     'gammas': [0.5, 1.0, 2.0], 'cv': 3,
        #     'max_iter': 1000, 'tol': 1e-4, 'n_jobs': -1
        # }, False)

        # algorithms['Fused_LassoCV'] = (FusedLassoCV, {
        #     'lambda_fused_ratios': [0.1, 0.5, 1.0, 2.0], 'cv': 3,
        #     'max_iter': 1000, 'tol': 1e-4, 'n_jobs': -1
        # }, False)

        # algorithms['Group_LassoCV'] = (GroupLassoCV, {
        #     'cv': 3, 'max_iter': 1000, 'tol': 1e-4, 'n_jobs': -1
        # }, False)

        # algorithms['Adaptive_Sparse_Group_LassoCV'] = (AdaptiveSparseGroupLassoCV, {
        #     'l1_ratios': [0.1, 0.5, 0.9], 'gamma': 1.0, 'cv': 3,
        #     'max_iter': 500, 'tol': 1e-4, 'n_jobs': -1
        # }, False)

        # skglm 系列（暂时保留固定参数版本，skglm的CV使用GeneralizedLinearEstimatorCV）
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / 'other_lasso/skglm_benchmark'))
            from skglm import Lasso as SkglmLasso, ElasticNet, WeightedLasso
            from skglm.cv import GeneralizedLinearEstimatorCV
            from skglm.datafits import Quadratic
            from skglm.penalties import L1

            algorithms['skglm_Lasso'] = (SkglmLasso, {
                'alpha': 0.1, 'max_iter': 50, 'tol': 1e-4
            }, False)

            algorithms['skglm_ElasticNet'] = (ElasticNet, {
                'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 50, 'tol': 1e-4
            }, False)

            algorithms['skglm_WeightedLasso'] = (WeightedLasso, {
                'alpha': 0.1, 'max_iter': 50, 'tol': 1e-4
            }, False)

        except Exception as e:
            warnings.warn(f"skglm 导入失败: {e}")

    else:  # classification
        algorithms['Sklearn_LogisticCV_L1'] = (LogisticRegressionCV, {
            'Cs': np.logspace(-4, 1, 20), 'penalty': 'l1',
            'solver': 'liblinear', 'cv': 3, 'max_iter': 1000,
            'n_jobs': -1, 'random_state': 2026
        }, True)

        # 本地CV实现（自动调参）- 暂时注释，后续优化后再开启
        # algorithms['Adaptive_LogisticCV'] = (AdaptiveLassoCV, {
        #     'gammas': [0.5, 1.0, 2.0], 'family': 'binomial',
        #     'cv': 3, 'max_iter': 1000, 'tol': 1e-4, 'n_jobs': -1
        # }, True)

    return algorithms
