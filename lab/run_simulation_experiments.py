#!/usr/bin/env python
"""
运行模拟实验
支持实验1~实验7共七个场景（包含3个魔鬼压力测试场景）
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
    generate_experiment1_data, generate_experiment2_data, generate_experiment3_data,
    generate_experiment4_data, generate_experiment5_data, generate_experiment6_data,
    generate_experiment7_data,
    run_algorithm, save_results
)
from other_lasso import GroupLasso

def parse_args():
    parser = argparse.ArgumentParser(description='运行模拟实验')
    parser.add_argument('--experiment', '-e', type=str, default='all',
                        choices=['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6', 'exp7', 'all'],
                        help='要运行的实验')
    parser.add_argument('--n-repeats', '-n', type=int, default=EXPERIMENT_CONFIG['n_repeats'],
                        help='实验重复次数')
    parser.add_argument('--family', '-f', type=str, default='all',
                        choices=['gaussian', 'binomial', 'all'],
                        help='任务类型：回归/二分类')
    parser.add_argument('--algorithms', '-a', type=str, default='all',
                        help='要运行的算法列表，逗号分隔，例如："原始UniLasso,标准Lasso,XLasso-Full"，默认运行所有算法')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式，只运行2次重复，100个特征')
    # -------- 结构参数覆盖（网格搜索用） --------
    parser.add_argument('--fixed-lambda', type=float, default=None,
                        help='固定lambda值，禁用CV，直接用该lambda计算（用于Stage1结构搜索）')
    parser.add_argument('--k', type=float, default=None,
                        help='覆盖XLasso的k(gamma)参数')
    parser.add_argument('--enable-group-decomp', action='store_true',
                        help='启用组分解（覆盖算法配置）')
    parser.add_argument('--no-group-decomp', action='store_true',
                        help='禁用组分解（覆盖算法配置）')
    parser.add_argument('--group-corr-threshold', type=float, default=None,
                        help='覆盖组相关系数阈值')
    parser.add_argument('--no-group-aware-filter', action='store_true',
                        help='禁用组感知过滤（覆盖算法配置）')
    parser.add_argument('--save-root', type=str, default=None,
                        help='覆盖结果保存根目录（用于tune_structure.py网格搜索）')
    return parser.parse_args()

def main():
    args = parse_args()

    # 调整调试模式参数
    if args.debug:
        args.n_repeats = min(args.n_repeats, 2)
        print(f"🐞 调试模式: 重复次数={args.n_repeats}, 特征数=100")

    # 实验配置（按paper.md Section 5.1设置）
    experiments = {
        'exp1': ('高维成对相关稀疏回归', generate_experiment1_data),
        'exp2': ('AR(1)相关稀疏回归', generate_experiment2_data),
        'exp3': ('二分类偏移变量选择', generate_experiment3_data),
        'exp4': ('孪生变量反符号选择', generate_experiment4_data),
        'exp5': ('魔鬼等级1：绝对隐身陷阱', generate_experiment5_data),
        'exp6': ('魔鬼等级2：鸠占鹊巢陷阱', generate_experiment6_data),
        'exp7': ('魔鬼等级3：AR(1)符号雪崩', generate_experiment7_data),
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

    import csv
    from datetime import datetime

    # 创建全局保存目录
    if args.save_root:
        save_root = args.save_root
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = os.path.join(EXPERIMENT_CONFIG['save_dir'], f"simulation_results_{timestamp}")
    os.makedirs(save_root, exist_ok=True)
    print(f"\n📂 所有实验结果将保存到: {save_root}")

    # 过滤要运行的算法
    if args.algorithms.lower() == 'all':
        filtered_algorithms = ALGORITHMS.items()
    else:
        selected_algs = [alg.strip() for alg in args.algorithms.split(',')]
        filtered_algorithms = [(name, config) for name, config in ALGORITHMS.items() if name in selected_algs]
        if not filtered_algorithms:
            print(f"❌ 没有找到指定的算法，请从以下列表中选择: {list(ALGORITHMS.keys())}")
            sys.exit(1)

    print(f"\n📋 将要运行的算法: {[name for name, _ in filtered_algorithms]}")

    # 按算法优先级逐个运行：每个算法单独创建目录和csv文件（按你要求的目录结构）
    for method_idx, (method_name, method_config) in enumerate(filtered_algorithms):
        # -------- 结构参数覆盖（网格搜索用） --------
        # 动态构建算法配置（不修改原始ALGORITHMS）
        alg_class = method_config['class']
        params = method_config['params'].copy()
        task_type = method_config['task_type']

        # 如果是xlasso系列，根据命令行参数覆盖结构参数
        if alg_class in ('xlasso', 'xlasso_cv'):
            # --fixed-lambda: 切换到非CV版本，用固定lambda
            if args.fixed_lambda is not None:
                alg_class = 'xlasso'  # 切换到fit_uni路径
                params['lmda'] = args.fixed_lambda
                # 移除cv相关参数
                params.pop('cv_folds', None)
                params.pop('lmda_min_ratio', None)
                params.pop('n_lmdas', None)

            if args.k is not None:
                params['k'] = args.k
            if args.enable_group_decomp:
                params['enable_group_decomp'] = True
            if args.no_group_decomp:
                params['enable_group_decomp'] = False
                params['enable_group_aware_filter'] = False
            if args.group_corr_threshold is not None:
                params['group_corr_threshold'] = args.group_corr_threshold
            if args.no_group_aware_filter:
                params['enable_group_aware_filter'] = False

            alg_class_overridden = alg_class
        else:
            alg_class_overridden = alg_class

        # 清理算法名称，创建合法文件名
        safe_method_name = method_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        method_dir = os.path.join(save_root, f"{method_idx+1:02d}_{safe_method_name}")
        os.makedirs(method_dir, exist_ok=True)

        # 每个算法单独的csv文件：raw保存每次重复的详细数据，summary保存平均结果
        raw_file = os.path.join(method_dir, f"{safe_method_name}_raw.csv")
        summary_file = os.path.join(method_dir, f"{safe_method_name}_summary.csv")

        # 写入表头
        csv_header = [
            'experiment_id', 'experiment_name', 'family', 'repeat', 'sigma',
            'mse', 'accuracy', 'auc', 'tpr', 'fdr', 'f1', 'n_selected', 'est_error', 'time_seconds'
        ]
        summary_header = [
            'experiment_id', 'experiment_name', 'family', 'sigma',
            'avg_mse', 'avg_accuracy', 'avg_auc', 'avg_tpr', 'avg_fdr', 'avg_f1', 'avg_n_selected', 'avg_est_error', 'avg_time_seconds'
        ]

        with open(raw_file, 'w', newline='') as f:
            csv.writer(f).writerow(csv_header)

        with open(summary_file, 'w', newline='') as f:
            csv.writer(f).writerow(summary_header)

        print(f"\n{'='*120}")
        print(f"🚀 优先级 {method_idx+1} | 开始运行算法: {method_name}")
        print(f"📂 结果目录: {method_dir}")
        print(f"{'='*120}")

        # 运行所有任务类型
        for family in families:
            # 检查任务类型是否支持
            if family == 'gaussian' and 'regression' not in task_type:
                print(f"⚠️  算法 {method_name} 不支持回归任务，跳过")
                continue
            if family == 'binomial' and 'classification' not in task_type:
                print(f"⚠️  算法 {method_name} 不支持分类任务，跳过")
                continue

            # 运行所有实验
            for exp_id in run_exps:
                exp_name, data_generator = experiments[exp_id]
                args.family = family
                n_samples = 300
                sigmas = [0.5, 1.0, 2.5]

                print(f"\n🧪 实验: {exp_name} | 任务类型: {family}")
                experiment_results = []

                # 运行所有重复
                for repeat in range(args.n_repeats):
                    print(f"\n🔄 第 {repeat+1}/{args.n_repeats} 次重复")
                    np.random.seed(EXPERIMENT_CONFIG['random_state'] + repeat)

                    for sigma in sigmas:
                        print(f"  σ={sigma}...", end="", flush=True)

                        # 生成数据（使用各实验默认参数）
                        X, y, beta_true = data_generator(
                            n=n_samples, sigma=sigma, family=args.family
                        )
                        # 调试模式：截断到100个特征
                        if args.debug:
                            X = X[:, :100]
                            beta_true = beta_true[:100]

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

                        start_time = time.time()
                        # 构建覆盖配置（用于网格搜索时传入动态参数）
                        alg_config_override = {
                            'class': alg_class_overridden,
                            'params': params,
                            'task_type': task_type
                        }
                        metrics = run_algorithm(
                            method_name, X_train, y_train, X_test, y_test,
                            family=args.family, groups=groups, beta_true=beta_true, scaler=scaler,
                            alg_config_override=alg_config_override
                        )
                        run_time = time.time() - start_time

                        if metrics['success']:
                            # 立刻写入该算法的raw文件，记录每次重复的详细数据（方便排查异常）
                            mse_val = metrics.get('mse', np.nan)
                            accuracy_val = metrics.get('accuracy', np.nan)
                            auc_val = metrics.get('auc', np.nan)
                            tpr_val = metrics.get('tpr', np.nan)
                            fdr_val = metrics.get('fdr', np.nan)
                            f1_val = metrics.get('f1', np.nan)
                            n_selected_val = metrics.get('n_selected', np.nan)
                            est_error_val = metrics.get('est_error', np.nan)

                            with open(raw_file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    exp_id, exp_name, family, repeat+1, sigma,
                                    mse_val, accuracy_val, auc_val, tpr_val, fdr_val, f1_val, n_selected_val, est_error_val, run_time
                                ])

                            result_row = {
                                'sigma': sigma,
                                'time_seconds': run_time,
                                **metrics
                            }
                            experiment_results.append(result_row)

                            # 格式化显示
                            parts = [" ✅ 完成,"]
                            if family == 'gaussian':
                                mse_str = f"{mse_val:.4f}" if isinstance(mse_val, (int, float)) and not np.isnan(mse_val) else "N/A"
                                parts.append(f"MSE={mse_str}")
                            else:
                                acc_str = f"{accuracy_val:.4f}" if isinstance(accuracy_val, (int, float)) and not np.isnan(accuracy_val) else "N/A"
                                parts.append(f"准确率={acc_str}")
                                auc_str = f"{auc_val:.4f}" if isinstance(auc_val, (int, float)) and not np.isnan(auc_val) else "N/A"
                                parts.append(f"AUC={auc_str}")
                            f1_str = f"{f1_val:.4f}" if isinstance(f1_val, (int, float)) and not np.isnan(f1_val) else "N/A"
                            parts.append(f"F1={f1_str}")
                            parts.append(f"耗时={run_time:.2f}s")
                            print(" ".join(parts), flush=True)
                        else:
                            print(f" ❌ 失败", flush=True)

                # 该实验所有重复完成，计算平均值
                if experiment_results:
                    print(f"\n📊 {exp_name} 平均结果:")
                    for sigma in sigmas:
                        sigma_results = [r for r in experiment_results if r['sigma'] == sigma]
                        if sigma_results:
                            avg_mse = np.nanmean([r.get('mse', np.nan) for r in sigma_results])
                            avg_accuracy = np.nanmean([r.get('accuracy', np.nan) for r in sigma_results])
                            avg_auc = np.nanmean([r.get('auc', np.nan) for r in sigma_results])
                            avg_tpr = np.nanmean([r.get('tpr', np.nan) for r in sigma_results])
                            avg_fdr = np.nanmean([r.get('fdr', np.nan) for r in sigma_results])
                            avg_f1 = np.nanmean([r.get('f1', np.nan) for r in sigma_results])
                            avg_n_selected = np.nanmean([r.get('n_selected', np.nan) for r in sigma_results])
                            avg_est_error = np.nanmean([r.get('est_error', np.nan) for r in sigma_results])
                            avg_time = np.nanmean([r['time_seconds'] for r in sigma_results])

                            # 格式化显示，只显示存在的指标
                            output_parts = [f"σ={sigma}:"]
                            if not np.isnan(avg_mse):
                                output_parts.append(f"MSE={avg_mse:.4f}")
                            if not np.isnan(avg_accuracy):
                                output_parts.append(f"准确率={avg_accuracy:.4f}")
                            if not np.isnan(avg_auc):
                                output_parts.append(f"AUC={avg_auc:.4f}")
                            if not np.isnan(avg_tpr):
                                output_parts.append(f"TPR={avg_tpr:.4f}")
                            if not np.isnan(avg_fdr):
                                output_parts.append(f"FDR={avg_fdr:.4f}")
                            if not np.isnan(avg_f1):
                                output_parts.append(f"F1={avg_f1:.4f}")
                            if not np.isnan(avg_n_selected):
                                output_parts.append(f"选中变量数={avg_n_selected:.1f}")
                            if not np.isnan(avg_est_error):
                                output_parts.append(f"EstError={avg_est_error:.4f}")
                            output_parts.append(f"耗时={avg_time:.2f}s")
                            print("  " + " ".join(output_parts))

                            # 写入该算法的汇总文件，保存平均结果
                            with open(summary_file, 'a', newline='') as f:
                                summary_writer = csv.writer(f)
                                summary_writer.writerow([
                                    exp_id, exp_name, family, sigma,
                                    avg_mse, avg_accuracy, avg_auc, avg_tpr, avg_fdr, avg_f1, avg_n_selected, avg_est_error, avg_time
                                ])

        print(f"\n✅ 算法 {method_name} 所有实验完成!")
        print(f"📊 结果已保存到: {method_dir}")

    print("\n" + "="*120)
    print("🎉 所有模拟实验全部完成!")
    print(f"📂 所有结果已保存到: {save_root}")
    print("="*120)

if __name__ == "__main__":
    main()
