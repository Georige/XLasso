#!/usr/bin/env python3
"""
NLasso 超参数调优工具
支持两阶段调优流程：
- Stage1: 固定lambda，搜索最优结构化参数
- Stage2: 在最优结构化参数下，CV调优lambda
"""
import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from NLasso import NLasso, NLassoClassifier, metrics
from data_generator import (
    generate_experiment1_data,
    generate_experiment2_data,
    generate_experiment3_data,
    generate_experiment4_data,
    generate_experiment5_data,
    generate_experiment6_data,
    generate_experiment7_data
)

# 实验生成器映射
EXPERIMENT_GENERATORS = {
    1: generate_experiment1_data,
    2: generate_experiment2_data,
    3: generate_experiment3_data,
    4: generate_experiment4_data,
    5: generate_experiment5_data,
    6: generate_experiment6_data,
    7: generate_experiment7_data
}

# 实验名称映射
EXPERIMENT_NAMES = {
    1: "高维成对相关稀疏回归",
    2: "AR(1)相关稀疏回归",
    3: "二分类偏移变量选择",
    4: "反符号孪生变量",
    5: "绝对隐身陷阱",
    6: "鸠占鹊巢陷阱",
    7: "自回归符号雪崩"
}

# =============================================================================
# Numpy JSON Encoder
# =============================================================================
class NumpyEncoder(json.JSONEncoder):
    """支持numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
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

# =============================================================================
# 参数搜索空间定义
# =============================================================================
# Stage1: 固定lambda，搜索结构化参数
STAGE1_PARAM_SPACE = {
    'lambda_ridge': [1.0, 5.0, 10.0, 20.0, 50.0],  # 第一阶段强Ridge正则化强度
    'gamma': [0.1, 0.3, 0.5, 0.7, 1.0],  # 权重指数映射陡峭程度
    's': [1.0],  # 全局惩罚缩放因子（固定为1）
    'group_threshold': [0.5, 0.6, 0.7, 0.8, 0.9],  # 相关性分组阈值
    'group_truncation_threshold': [0.0, 0.3, 0.5, 0.7],  # 组感知截断阈值
}

# Stage1固定的lambda值
STAGE1_FIXED_LAMBDA = 0.1

# Stage2: 结构化参数固定，搜索lambda的空间
STAGE2_LAMBDA_SPACE = np.logspace(-4, 1, 50)  # 50个点，从0.0001到10

# =============================================================================
# 结果保存器
# =============================================================================
class TuningResultSaver:
    """
    调优结果保存器
    遵循用户要求的保存结构
    """
    def __init__(self, result_root: str = "tuning/result"):
        self.result_root = Path(result_root)
        self.result_root.mkdir(parents=True, exist_ok=True)

        # 生成时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp_dir = self.result_root / self.timestamp
        self.timestamp_dir.mkdir(exist_ok=True)

        # 存储所有实验的结果
        self.all_results = []

    def _generate_param_string(self, params: Dict[str, Any]) -> str:
        """
        生成参数字符串用于文件夹命名
        格式: key1_value1_key2_value2...
        """
        sorted_keys = sorted(params.keys())
        parts = []
        for key in sorted_keys:
            value = params[key]
            # 格式化数值，避免太长
            if isinstance(value, float):
                value_str = f"{value:.4f}".rstrip('0').rstrip('.') if '.' in f"{value:.4f}" else f"{value:.4f}"
            else:
                value_str = str(value)
            parts.append(f"{key}_{value_str}")
        return "_".join(parts)

    def _generate_param_hash(self, params: Dict[str, Any]) -> str:
        """生成参数hash作为文件夹名的补充（防止过长）"""
        param_str = json.dumps(convert_numpy_types(params), sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]

    def save_experiment_result(
        self,
        experiment_id: int,
        params: Dict[str, Any],
        raw_results: List[Dict[str, Any]],
        summary_stats: Dict[str, Any],
        stage: int = 1
    ) -> Path:
        """
        保存单个参数配置的实验结果

        Args:
            experiment_id: 实验ID (1-7)
            params: 该次实验的参数配置
            raw_results: 每次重复试验的原始结果列表
            summary_stats: 汇总统计结果
            stage: 调优阶段 (1 or 2)

        Returns:
            保存结果的文件夹路径
        """
        # 实验名称
        exp_name = EXPERIMENT_NAMES[experiment_id]

        # 生成实验文件夹名
        param_str = self._generate_param_string(params)
        param_hash = self._generate_param_hash(params)
        stage_prefix = f"stage{stage}"
        exp_folder_name = f"{stage_prefix}_{exp_name}_{param_str}_{param_hash}"
        # 防止文件夹名过长，截断param_str
        if len(exp_folder_name) > 150:
            exp_folder_name = f"{stage_prefix}_{exp_name}_{param_hash}"

        exp_dir = self.timestamp_dir / exp_folder_name
        exp_dir.mkdir(exist_ok=True)

        # 1. 保存raw.csv - 每次重复试验记录
        df_raw = pd.DataFrame(raw_results)
        df_raw.to_csv(exp_dir / "raw.csv", index=False)

        # 2. 保存summary.csv - 重复指标平均值
        df_summary = pd.DataFrame([summary_stats])
        df_summary.to_csv(exp_dir / "summary.csv", index=False)

        # 3. 保存params.json - 实验所用参数
        params_to_save = {
            'experiment_id': experiment_id,
            'experiment_name': exp_name,
            'stage': stage,
            'timestamp': self.timestamp,
            'params': params,
            'n_repeats': len(raw_results)
        }
        with open(exp_dir / "params.json", 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(params_to_save), f, indent=2, cls=NumpyEncoder)

        # 保存到总结果列表
        self.all_results.append({
            'experiment_id': experiment_id,
            'experiment_name': exp_name,
            'stage': stage,
            'exp_dir': str(exp_dir),
            **params,
            **summary_stats
        })

        return exp_dir

    def save_overall_summary(self) -> Tuple[Path, Path]:
        """
        保存整个调优实验的总体汇总
        包含所有不同参数的平均指标，以及最优模型排名

        Returns:
            (overall_summary_path, ranking_path)
        """
        if not self.all_results:
            raise ValueError("没有结果可保存，请先运行实验")

        df_overall = pd.DataFrame(self.all_results)

        # 保存总体汇总
        overall_path = self.timestamp_dir / "overall_summary.csv"
        df_overall.to_csv(overall_path, index=False)

        # 生成最优模型排名（按MSE升序或准确率降序，FDR升序）
        ranking_dfs = []
        for exp_id in df_overall['experiment_id'].unique():
            df_exp = df_overall[df_overall['experiment_id'] == exp_id].copy()

            if 'mse_mean' in df_exp.columns:
                # 回归任务：优先按MSE升序，再按FDR升序
                df_exp = df_exp.sort_values(by=['mse_mean', 'fdr_mean'], ascending=[True, True])
            elif 'accuracy_mean' in df_exp.columns:
                # 分类任务：优先按准确率降序，再按FDR升序
                df_exp = df_exp.sort_values(by=['accuracy_mean', 'fdr_mean'], ascending=[False, True])

            df_exp['rank'] = range(1, len(df_exp) + 1)
            ranking_dfs.append(df_exp)

        df_ranking = pd.concat(ranking_dfs, ignore_index=True)
        ranking_path = self.timestamp_dir / "ranking.csv"
        df_ranking.to_csv(ranking_path, index=False)

        return overall_path, ranking_path

    def get_timestamp_dir(self) -> Path:
        """获取当前时间戳目录"""
        return self.timestamp_dir

# =============================================================================
# 参数组合生成器
# =============================================================================
def generate_param_combinations(param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    生成参数网格的所有组合

    Args:
        param_space: 参数空间字典，key是参数名，value是候选值列表

    Returns:
        所有参数组合的列表
    """
    import itertools
    keys = list(param_space.keys())
    values = list(param_space.values())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]

# =============================================================================
# 单组参数配置的调优实验运行
# =============================================================================
def run_single_tuning_experiment(
    experiment_id: int,
    params: Dict[str, Any],
    n_repeats: int = 3,
    seed_start: int = 2026,
    family: str = "gaussian",
    verbose: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    运行单个参数配置的调优实验

    Args:
        experiment_id: 实验ID (1-7)
        params: NLasso参数字典
        n_repeats: 重复次数
        seed_start: 起始随机种子
        family: 任务类型 ("gaussian" 回归 or "binomial" 分类)
        verbose: 是否打印详细信息

    Returns:
        (raw_results, summary_stats)
    """
    raw_results = []
    generator = EXPERIMENT_GENERATORS[experiment_id]

    for repeat in range(n_repeats):
        seed = seed_start + repeat
        if verbose:
            print(f"  Repeat {repeat+1}/{n_repeats}, seed={seed}")

        # 生成数据
        np.random.seed(seed)
        if family == "gaussian":
            X, y, beta_true = generator(family='gaussian')
            is_classification = False
        else:
            X, y, beta_true = generator(family='binomial')
            is_classification = True

        # 训练模型
        import time
        start_time = time.time()

        if is_classification:
            model = NLassoClassifier(**params)
        else:
            model = NLasso(**params)

        model.fit(X, y)
        train_time = time.time() - start_time

        # 预测
        start_time = time.time()
        y_pred = model.predict(X)
        predict_time = time.time() - start_time

        # 计算指标
        if is_classification:
            y_score = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
            exp_metrics = metrics.calculate_classification_metrics(
                y_true=y,
                y_pred=y_pred,
                y_score=y_score,
                coef=model.coef_,
                beta_true=beta_true,
                train_time=train_time,
                predict_time=predict_time,
                n_iter=getattr(model, 'n_iter_', None)
            )
        else:
            exp_metrics = metrics.calculate_regression_metrics(
                y_true=y,
                y_pred=y_pred,
                coef=model.coef_,
                beta_true=beta_true,
                train_time=train_time,
                predict_time=predict_time,
                n_iter=getattr(model, 'n_iter_', None)
            )

        # 保存原始结果
        raw_result = {
            'repeat': repeat + 1,
            'seed': seed,
            **exp_metrics
        }
        raw_results.append(raw_result)

    # 计算汇总统计
    summary_stats = {}
    df_raw = pd.DataFrame(raw_results)

    for col in df_raw.columns:
        if col in ['repeat', 'seed']:
            continue
        if pd.api.types.is_numeric_dtype(df_raw[col]):
            summary_stats[f"{col}_mean"] = df_raw[col].mean()
            summary_stats[f"{col}_std"] = df_raw[col].std()
            summary_stats[f"{col}_min"] = df_raw[col].min()
            summary_stats[f"{col}_max"] = df_raw[col].max()

    return raw_results, summary_stats
