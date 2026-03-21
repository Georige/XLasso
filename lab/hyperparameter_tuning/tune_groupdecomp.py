#!/usr/bin/env python
"""
XLasso v2.3.2 最新算法超参数调优
只包含改进框架里的最新参数，不含兼容旧参数

核心参数：
- k: 权重调节系数，公式 w_j = 0.5 * p_j^k
- lmda_scale: 全局惩罚缩放系数 s

组级扩展参数（v2.1）：
- enable_group_decomp: 是否开启组级正交分解
- group_corr_threshold: 组相关系数阈值 τ
- enable_group_aware_filter: 是否开启组感知过滤
- group_filter_k: 组过滤保留阈值（变量数）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from unilasso.uni_lasso import fit_uni
from experiment_utils import generate_experiment1_data, generate_experiment2_data, generate_experiment3_data

sns.set_style("whitegrid")

def evaluate(y_true, y_pred, beta_true, beta_pred, family='gaussian'):
    """计算评价指标"""
    metrics = {}

    if family == 'gaussian':
        metrics['mse'] = float(np.mean((y_true - y_pred) ** 2))
    else:
        y_pred_label = (y_pred > 0.5).astype(int)
        metrics['accuracy'] = float(np.mean(y_true == y_pred_label))
        from sklearn.metrics import roc_auc_score
        try:
            metrics['auc'] = float(roc_auc_score(y_true, y_pred))
        except:
            metrics['auc'] = 0.5

    true_nonzero = np.abs(beta_true) > 1e-8
    pred_nonzero = np.abs(beta_pred) > 1e-8

    tp = np.sum(true_nonzero & pred_nonzero)
    fp = np.sum(~true_nonzero & pred_nonzero)
    fn = np.sum(true_nonzero & ~pred_nonzero)

    metrics['tpr'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics['fdr'] = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics['f1'] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    metrics['n_selected'] = int(np.sum(pred_nonzero))
    metrics['true_positive'] = int(tp)

    return metrics

def run_single_experiment(params, data_generator, family, n_repeats=3):
    """运行单个参数组合的实验，多次重复取平均"""
    all_metrics = []

    for repeat in range(n_repeats):
        np.random.seed(42 + repeat)
        n = 300
        p = 500
        sigma = 1.0

        X, y, beta_true = data_generator(n=n, p=p, sigma=sigma, family=family)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42 + repeat
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        try:
            result = fit_uni(
                X_train, y_train,
                family=family,
                k=params['k'],
                lmda_scale=params['lmda_scale'],
                lmda_min_ratio=1e-2,
                n_lmdas=50,
                # 组级扩展参数
                enable_group_decomp=params.get('enable_group_decomp', False),
                group_corr_threshold=params.get('group_corr_threshold', 0.7),
                enable_group_aware_filter=params.get('enable_group_aware_filter', False),
                group_filter_k=params.get('group_filter_k', None)
            )

            # 选最优lambda：在正则化路径上找F1最大的点
            best_f1 = -1
            best_metrics = None

            for lmda_idx in range(len(result.lmdas)):
                beta_pred = result.coefs[lmda_idx]
                intercept = result.intercept[lmda_idx]

                if family == 'gaussian':
                    y_pred = X_test @ beta_pred + intercept
                else:
                    y_pred = 1 / (1 + np.exp(- (X_test @ beta_pred + intercept)))

                metrics = evaluate(y_test, y_pred, beta_true, beta_pred, family)
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_metrics = metrics

            if best_metrics is not None:
                all_metrics.append(best_metrics)
        except Exception as e:
            param_str = ', '.join([f'{k}={v}' for k, v in params.items()])
            print(f"❌ 实验失败 ({param_str}, repeat={repeat}): {e}")
            continue

    if not all_metrics:
        return None

    # 平均多次重复的结果
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_metrics

def main():
    # 输出目录
    output_dir = "results/groupdecomp_tuning"
    os.makedirs(output_dir, exist_ok=True)

    # 算法模型配置
    models = [
        ('XLasso-Soft', {
            'enable_group_decomp': False,
            'enable_group_aware_filter': False,
        }),
        ('XLasso-GroupDecomp', {
            'enable_group_decomp': True,
            'enable_group_aware_filter': False,
        }),
        ('XLasso-Full', {
            'enable_group_decomp': True,
            'enable_group_aware_filter': True,
        }),
    ]

    # 核心参数网格（v2.3.2最新公式）
    k_values = [0.3, 0.5, 0.8, 1.0, 1.5]
    lmda_scale_values = [0.2, 0.3, 0.5, 0.8, 1.0]

    # 组级扩展参数网格
    group_corr_threshold_values = [0.6, 0.7, 0.8]
    group_filter_k_values = [1, 2, 3, None]  # None表示使用默认阈值(组大小的50%)

    n_repeats = 3

    # 实验配置
    experiments = [
        ('exp1', '成对相关稀疏回归', 'Pairwise Correlated Sparse Regression', generate_experiment1_data, 'gaussian'),
        ('exp2', 'AR(1)相关稀疏回归', 'AR(1) Correlated Sparse Regression', generate_experiment2_data, 'gaussian'),
        ('exp3', '二分类偏移变量选择', 'Binary Classification Offset Selection', generate_experiment3_data, 'binomial'),
    ]

    all_results = []

    for model_name, model_base_params in models:
        print(f"\n{'='*80}")
        print(f"🚀 开始调优模型: {model_name}")
        print(f"{'='*80}")

        for exp_id, exp_name_cn, exp_name_en, data_gen, family in experiments:
            print(f"\n{'='*80}")
            print(f"📊 实验: {exp_name_cn}")
            print(f"{'='*80}")

            exp_results = []

            # 确定参数网格
            if model_name == 'XLasso-Soft':
                # 只有基础参数
                param_combinations = [
                    {'k': k, 'lmda_scale': ls, **model_base_params}
                    for k in k_values
                    for ls in lmda_scale_values
                ]
            elif model_name == 'XLasso-GroupDecomp':
                # 基础参数 + 组相关阈值
                param_combinations = [
                    {'k': k, 'lmda_scale': ls, 'group_corr_threshold': gct, **model_base_params}
                    for k in k_values
                    for ls in lmda_scale_values
                    for gct in group_corr_threshold_values
                ]
            else:  # XLasso-Full
                # 全部参数
                param_combinations = [
                    {'k': k, 'lmda_scale': ls, 'group_corr_threshold': gct, 'group_filter_k': gfk, **model_base_params}
                    for k in k_values
                    for ls in lmda_scale_values
                    for gct in group_corr_threshold_values
                    for gfk in group_filter_k_values
                ]

            total_runs = len(param_combinations)
            run_idx = 0

            for params in param_combinations:
                run_idx += 1
                param_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                print(f"\n🔍 {run_idx}/{total_runs}: {param_str}")

                metrics = run_single_experiment(params, data_gen, family, n_repeats)
                if metrics is None:
                    continue

                result_row = {
                    'model': model_name,
                    'experiment': exp_id,
                    'experiment_name': exp_name_cn,
                    'experiment_name_en': exp_name_en,
                    **params,
                    **metrics
                }
                exp_results.append(result_row)
                all_results.append(result_row)

                # 打印结果
                if family == 'gaussian':
                    print(f"✅ 完成: F1={metrics['f1']:.4f}, MSE={metrics['mse']:.4f}, TPR={metrics['tpr']:.4f}, FDR={metrics['fdr']:.4f}, 选中={metrics['n_selected']}, 正确={metrics['true_positive']}")
                else:
                    print(f"✅ 完成: F1={metrics['f1']:.4f}, 准确率={metrics['accuracy']:.4f}, TPR={metrics['tpr']:.4f}, FDR={metrics['fdr']:.4f}, 选中={metrics['n_selected']}, 正确={metrics['true_positive']}")

            # 保存单实验结果
            if exp_results:
                exp_df = pd.DataFrame(exp_results)
                exp_df.to_csv(f"{output_dir}/{model_name}_{exp_id}_results.csv", index=False, float_format='%.6f')
                print(f"\n📁 实验结果已保存到: {output_dir}/{model_name}_{exp_id}_results.csv")

                # 生成最优参数
                best_row = exp_df.loc[exp_df['f1'].idxmax()]
                print(f"\n🏆 {exp_name_cn} 最优参数:")
                best_params_str = ', '.join([f'{k}={best_row[k]}' for k in params.keys() if k in best_row])
                print(f"   {best_params_str}")
                print(f"   最优F1 = {best_row['f1']:.4f}")
                if family == 'gaussian':
                    print(f"   对应MSE = {best_row['mse']:.4f}, TPR = {best_row['tpr']:.4f}, FDR = {best_row['fdr']:.4f}, 选中变量={int(best_row['n_selected'])}, 正确变量={int(best_row['true_positive'])}")
                else:
                    print(f"   对应准确率 = {best_row['accuracy']:.4f}, TPR = {best_row['tpr']:.4f}, FDR = {best_row['fdr']:.4f}, 选中变量={int(best_row['n_selected'])}, 正确变量={int(best_row['true_positive'])}")

    # 保存所有结果
    all_df = pd.DataFrame(all_results)
    all_df.to_csv(f"{output_dir}/all_experiments_results.csv", index=False, float_format='%.6f')
    print(f"\n📁 所有实验结果已保存到: {output_dir}/all_experiments_results.csv")

    # 生成汇总最优参数表
    best_params = []
    for model_name in all_df['model'].unique():
        for exp_id in all_df['experiment'].unique():
            exp_df = all_df[(all_df['model'] == model_name) & (all_df['experiment'] == exp_id)]
            if len(exp_df) == 0:
                continue
            best_row = exp_df.loc[exp_df['f1'].idxmax()]
            row_dict = {
                'model': best_row['model'],
                'experiment': best_row['experiment'],
                'experiment_name': best_row['experiment_name'],
                'best_f1': best_row['f1'],
                'tpr': best_row['tpr'],
                'fdr': best_row['fdr'],
                'n_selected': int(best_row['n_selected']),
                'true_positive': int(best_row['true_positive']),
            }
            # 添加最优参数
            for k in ['k', 'lmda_scale', 'group_corr_threshold', 'group_filter_k']:
                if k in best_row:
                    row_dict[f'best_{k}'] = best_row[k]
            # 添加MSE或准确率
            if 'mse' in best_row and not pd.isna(best_row['mse']):
                row_dict['mse'] = best_row['mse']
            if 'accuracy' in best_row and not pd.isna(best_row['accuracy']):
                row_dict['accuracy'] = best_row['accuracy']
            best_params.append(row_dict)

    best_params_df = pd.DataFrame(best_params)
    best_params_df.to_csv(f"{output_dir}/best_parameters.csv", index=False, float_format='%.6f')
    print(f"\n📁 最优参数汇总已保存到: {output_dir}/best_parameters.csv")

    print(f"\n🎉 所有调优实验完成！结果和可视化已保存到 {output_dir} 目录")
    print("\n📊 最优参数汇总:")
    print(best_params_df.to_string(index=False))

if __name__ == "__main__":
    main()
