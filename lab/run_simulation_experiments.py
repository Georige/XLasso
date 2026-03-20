#!/usr/bin/env python
"""
运行模拟实验
支持实验1、实验2、实验4三个经典场景
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from experiment_utils import (
    EXPERIMENT_CONFIG, ALGORITHMS,
    generate_experiment1_data, generate_experiment2_data, generate_experiment4_data,
    run_algorithm, save_results
)
from other_lasso import GroupLasso

def parse_args():
    parser = argparse.ArgumentParser(description='运行模拟实验')
    parser.add_argument('--experiment', '-e', type=str, default='all',
                        choices=['exp1', 'exp2', 'exp4', 'all'],
                        help='要运行的实验')
    parser.add_argument('--n-repeats', '-n', type=int, default=EXPERIMENT_CONFIG['n_repeats'],
                        help='实验重复次数')
    parser.add_argument('--family', '-f', type=str, default='all',
                        choices=['gaussian', 'binomial', 'all'],
                        help='任务类型：回归/二分类')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式，只运行2次重复，100个特征')
    return parser.parse_args()

def run_single_experiment(exp_name, data_generator, args):
    """运行单个实验"""
    print(f"\n" + "="*100)
    print(f"开始运行实验: {exp_name}")
    print(f"任务类型: {args.family}, 重复次数: {args.n_repeats}")
    print("="*100)

    results = []
    n_samples = 300
    n_features = 1000 if not args.debug else 100
    sigmas = [0.5, 1.0, 2.5]

    for repeat in range(args.n_repeats):
        print(f"\n🔄 第 {repeat+1}/{args.n_repeats} 次重复")
        np.random.seed(EXPERIMENT_CONFIG['random_state'] + repeat)

        for sigma in sigmas:
            print(f"  噪声水平σ={sigma}...", end="", flush=True)

            # 生成数据
            X, y, beta_true = data_generator(
                n=n_samples, p=n_features, sigma=sigma, family=args.family
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

            # 自动分组（给需要分组的算法用）
            groups = GroupLasso.group_features_by_correlation(X_train, corr_threshold=0.7)

            # 运行所有算法
            for alg_name in ALGORITHMS:
                alg_config = ALGORITHMS[alg_name]
                if args.family == 'gaussian' and 'regression' not in alg_config['task_type']:
                    continue
                if args.family == 'binomial' and 'classification' not in alg_config['task_type']:
                    continue

                start_time = time.time()
                metrics = run_algorithm(
                    alg_name, X_train, y_train, X_test, y_test,
                    family=args.family, groups=groups
                )
                run_time = time.time() - start_time

                if metrics['success']:
                    result_row = {
                        '算法': alg_name,
                        '重复次数': repeat + 1,
                        '噪声σ': sigma,
                        '运行时间(s)': run_time,
                        **metrics
                    }
                    results.append(result_row)
                    print(f".", end="", flush=True)
                else:
                    print(f"x", end="", flush=True)

            print(" 完成")

    # 保存结果
    if len(results) > 0:
        config = {
            'experiment': exp_name,
            'family': args.family,
            'n_repeats': args.n_repeats,
            'debug': args.debug,
            'n_samples': n_samples,
            'n_features': n_features,
            'sigmas': sigmas
        }
        save_results(results, f'sim_{exp_name}_{args.family}', config=config)
    else:
        print("⚠️ 没有成功的实验结果")

    return results

def main():
    args = parse_args()

    # 调整调试模式参数
    if args.debug:
        args.n_repeats = min(args.n_repeats, 2)
        print(f"🐞 调试模式: 重复次数={args.n_repeats}, 特征数=100")

    # 实验配置
    experiments = {
        'exp1': ('高维成对相关稀疏回归', generate_experiment1_data),
        'exp2': ('AR(1)相关稀疏回归', generate_experiment2_data),
        'exp4': ('反符号孪生变量', generate_experiment4_data),
    }

    # 确定要运行的实验
    if args.experiment == 'all':
        run_exps = list(experiments.keys())
    else:
        run_exps = [args.experiment]

    # 确定任务类型
    if args.family == 'all':
        families = ['gaussian', 'binomial']
    else:
        families = [args.family]

    # 运行所有实验
    for exp_id in run_exps:
        exp_name, data_generator = experiments[exp_id]
        for family in families:
            args.family = family
            run_single_experiment(exp_id, data_generator, args)

    print("\n" + "="*100)
    print("所有模拟实验完成!")
    print("="*100)

if __name__ == "__main__":
    main()
