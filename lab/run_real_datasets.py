#!/usr/bin/env python
"""
运行真实数据集实验
支持乳腺癌、Arcene、Gisette、Dorothea数据集
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from experiment_utils import (
    EXPERIMENT_CONFIG, ALGORITHMS, run_algorithm, save_results
)
from other_lasso import GroupLasso

def load_breast_cancer_data():
    """加载乳腺癌数据集"""
    data = load_breast_cancer()
    return data.data, data.target, None

def load_arcene_data():
    """加载Arcene数据集（如果存在）"""
    # 这里后续补充实际加载逻辑，现在先用占位符
    raise NotImplementedError("Arcene数据集请手动下载后补充加载逻辑")

def load_gisette_data():
    """加载Gisette数据集"""
    raise NotImplementedError("Gisette数据集请手动下载后补充加载逻辑")

def load_dorothea_data():
    """加载Dorothea数据集"""
    raise NotImplementedError("Dorothea数据集请手动下载后补充加载逻辑")

DATASETS = {
    'breast_cancer': ('乳腺癌数据集', load_breast_cancer_data, 'classification'),
    'arcene': ('Arcene质谱数据', load_arcene_data, 'classification'),
    'gisette': ('Gisette手写数字', load_gisette_data, 'classification'),
    'dorothea': ('Dorothea药物发现', load_dorothea_data, 'classification'),
}

def parse_args():
    parser = argparse.ArgumentParser(description='运行真实数据集实验')
    parser.add_argument('--dataset', '-d', type=str, default='breast_cancer',
                        choices=list(DATASETS.keys()) + ['all'],
                        help='要运行的数据集')
    parser.add_argument('--n-repeats', '-n', type=int, default=EXPERIMENT_CONFIG['n_repeats'],
                        help='实验重复次数')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式，只运行2次重复')
    return parser.parse_args()

def run_dataset_experiment(dataset_id, args):
    """运行单个数据集实验"""
    dataset_name, load_func, task_type = DATASETS[dataset_id]
    print(f"\n" + "="*100)
    print(f"开始运行数据集: {dataset_name}")
    print(f"任务类型: {task_type}, 重复次数: {args.n_repeats}")
    print("="*100)

    try:
        X, y, beta_true = load_func()
    except NotImplementedError as e:
        print(f"⚠️ 跳过 {dataset_name}: {str(e)}")
        return []

    results = []
    family = 'binomial' if task_type == 'classification' else 'gaussian'

    for repeat in range(args.n_repeats):
        print(f"\n🔄 第 {repeat+1}/{args.n_repeats} 次重复")
        np.random.seed(EXPERIMENT_CONFIG['random_state'] + repeat)

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=EXPERIMENT_CONFIG['test_size'],
            random_state=EXPERIMENT_CONFIG['random_state'] + repeat,
            stratify=y if task_type == 'classification' else None
        )

        # 标准化
        if EXPERIMENT_CONFIG['standardize']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # 自动分组
        groups = GroupLasso.group_features_by_correlation(X_train, corr_threshold=0.7)
        print(f"自动分组: {len(groups)}个组")

        # 运行所有算法
        for alg_name in ALGORITHMS:
            alg_config = ALGORITHMS[alg_name]
            if task_type == 'classification' and 'classification' not in alg_config['task_type']:
                continue
            if task_type == 'regression' and 'regression' not in alg_config['task_type']:
                continue

            start_time = time.time()
            metrics = run_algorithm(
                alg_name, X_train, y_train, X_test, y_test,
                family=family, groups=groups
            )
            run_time = time.time() - start_time

            if metrics['success']:
                result_row = {
                    '算法': alg_name,
                    '重复次数': repeat + 1,
                    '运行时间(s)': run_time,
                    **metrics
                }
                results.append(result_row)
                print(f"✅ {alg_name}: 准确率={metrics.get('accuracy', 'N/A'):.4f}, AUC={metrics.get('auc', 'N/A'):.4f}, 耗时={run_time:.2f}s")
            else:
                print(f"❌ {alg_name}: 失败")

    # 保存结果
    if len(results) > 0:
        config = {
            'dataset': dataset_name,
            'task_type': task_type,
            'n_repeats': args.n_repeats,
            'debug': args.debug,
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }
        save_results(results, f'real_{dataset_id}', config=config)
    else:
        print("⚠️ 没有成功的实验结果")

    return results

def main():
    args = parse_args()

    # 调试模式
    if args.debug:
        args.n_repeats = min(args.n_repeats, 2)
        print(f"🐞 调试模式: 重复次数={args.n_repeats}")

    # 确定要运行的数据集
    if args.dataset == 'all':
        run_datasets = list(DATASETS.keys())
    else:
        run_datasets = [args.dataset]

    # 运行所有数据集
    for dataset_id in run_datasets:
        run_dataset_experiment(dataset_id, args)

    print("\n" + "="*100)
    print("所有真实数据集实验完成!")
    print("="*100)

if __name__ == "__main__":
    main()
