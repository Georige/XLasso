#!/usr/bin/env python
"""
超参数调优实验模板
用于优化XLasso系列算法的关键超参数
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from experiment_utils import (
    EXPERIMENT_CONFIG, ALGORITHMS,
    generate_experiment1_data, generate_experiment2_data, generate_experiment3_data, generate_experiment4_data,
    run_algorithm, save_results
)
from other_lasso import GroupLasso

# 待调优的超参数网格（基于新权重公式✅）
PARAM_GRID = {
    'XLasso-Soft': {
        'k': [0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
        'lmda_min_ratio': [1e-5, 1e-4, 1e-3, 1e-2],
        'lmda_scale': [0.1, 0.3, 0.5, 1.0, 2.0],
    },
    'XLasso-GroupDecomp': {
        'k': [0.8, 1.0, 1.5],
        'lmda_min_ratio': [1e-4, 1e-3],
        'lmda_scale': [0.3, 0.5, 1.0],
        'group_corr_threshold': [0.6, 0.7, 0.8],
        'enable_group_aware_filter': [True, False],
    },
    'XLasso-Full': {
        'k': [0.8, 1.0, 1.5],
        'lmda_min_ratio': [1e-4, 1e-3],
        'lmda_scale': [0.3, 0.5, 1.0],
        'group_corr_threshold': [0.6, 0.7, 0.8],
        'enable_group_aware_filter': [True],
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='超参数调优实验')
    parser.add_argument('--algorithm', '-a', type=str, required=True,
                        choices=PARAM_GRID.keys(),
                        help='要调优的算法')
    parser.add_argument('--experiment', '-e', type=str, default='exp2',
                        choices=['exp1', 'exp2', 'exp3', 'exp4-2'],
                        help='调优使用的实验场景')
    parser.add_argument('--n-repeats', '-n', type=int, default=3,
                        help='每个参数组合的重复次数')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式，只运行少量参数组合')
    return parser.parse_args()

def get_data_generator(exp_id):
    """获取数据生成函数"""
    generators = {
        'exp1': ('高维成对相关稀疏回归', generate_experiment1_data, 'gaussian'),
        'exp2': ('AR(1)相关稀疏回归', generate_experiment2_data, 'gaussian'),
        'exp3': ('二分类偏移变量选择', generate_experiment3_data, 'binomial'),
        'exp4-2': ('降维打击场景', generate_experiment4_data, 'binomial'),
    }
    return generators[exp_id]

def generate_param_combinations(param_grid):
    """生成所有参数组合"""
    from itertools import product
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))

def main():
    args = parse_args()
    alg_name = args.algorithm
    exp_id = args.experiment

    exp_name, data_generator, family = get_data_generator(exp_id)
    param_grid = PARAM_GRID[alg_name]

    # 调试模式简化参数网格
    if args.debug:
        for k in param_grid:
            param_grid[k] = param_grid[k][:2]
        args.n_repeats = min(args.n_repeats, 2)
        print(f"🐞 调试模式: 参数网格简化为 {param_grid}")

    print(f"🚀 开始超参数调优: {alg_name}")
    print(f"📊 实验场景: {exp_name}")
    print(f"🔢 参数网格: {param_grid}")
    print(f"🔄 重复次数: {args.n_repeats}")

    all_results = []
    param_combinations = list(generate_param_combinations(param_grid))
    total_runs = len(param_combinations) * args.n_repeats
    print(f"📈 总运行次数: {total_runs}")

    # 获取算法基础配置
    base_config = ALGORITHMS[alg_name]
    base_params = base_config['params'].copy()

    for run_idx, params in enumerate(param_combinations):
        print(f"\n{'='*80}")
        print(f"🔍 参数组合 {run_idx+1}/{len(param_combinations)}: {params}")
        print(f"{'='*80}")

        # 合并参数
        current_params = base_params.copy()
        current_params.update(params)

        for repeat in range(args.n_repeats):
            print(f"\n🔄 第 {repeat+1}/{args.n_repeats} 次重复")
            np.random.seed(EXPERIMENT_CONFIG['random_state'] + repeat)

            # 生成数据
            X, y, beta_true = data_generator(
                n=300, p=500 if not args.debug else 100, sigma=1.0, family=family
            )

            # 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=EXPERIMENT_CONFIG['test_size'],
                random_state=EXPERIMENT_CONFIG['random_state'] + repeat
            )

            # 标准化
            if EXPERIMENT_CONFIG['standardize']:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # 自动分组
            groups = GroupLasso.group_features_by_correlation(X_train, corr_threshold=0.7)

            # 运行算法
            metrics = run_algorithm(
                alg_name, X_train, y_train, X_test, y_test,
                family=family, groups=groups, beta_true=beta_true
            )

            if metrics['success']:
                result_row = {
                    'algorithm': alg_name,
                    'experiment': exp_name,
                    'repeat': repeat + 1,
                    **params,
                    **metrics
                }
                all_results.append(result_row)

                # 打印结果
                if family == 'gaussian':
                    print(f"✅ 完成: MSE={metrics.get('mse', 'N/A'):.4f}, F1={metrics.get('f1', 'N/A'):.4f}")
                else:
                    print(f"✅ 完成: 准确率={metrics.get('accuracy', 'N/A'):.4f}, AUC={metrics.get('auc', 'N/A'):.4f}, F1={metrics.get('f1', 'N/A'):.4f}")
            else:
                print(f"❌ 失败")

    # 保存结果
    if all_results:
        config = {
            'algorithm': alg_name,
            'experiment': exp_name,
            'param_grid': param_grid,
            'n_repeats': args.n_repeats,
            'debug': args.debug
        }
        save_results(all_results, f'tuning_{alg_name}_{exp_id}', config=config)
        print(f"\n🎉 调优完成! 结果已保存")

        # 打印最优参数
        print("\n🏆 最优参数 (按F1分数排序):")
        import pandas as pd
        df = pd.DataFrame(all_results)
        for params, group in df.groupby(list(param_grid.keys())):
            avg_f1 = group['f1'].mean()
            if family == 'gaussian':
                avg_metric = group['mse'].mean()
                print(f"参数{params}: 平均F1={avg_f1:.4f}, 平均MSE={avg_metric:.4f}")
            else:
                avg_acc = group['accuracy'].mean()
                print(f"参数{params}: 平均F1={avg_f1:.4f}, 平均准确率={avg_acc:.4f}")

if __name__ == "__main__":
    main()
