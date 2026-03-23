#!/usr/bin/env python
"""
核心超参数调优：lmda_scale 和 k 网格搜索
保存所有结果并生成可视化
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
# 移除中文字体配置，使用默认英文字体避免显示异常

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
    metrics['n_selected'] = int(np.sum(pred_nonzero))  # 选中变量总数
    metrics['true_positive'] = int(tp)  # 正确选中的变量数

    return metrics

def run_single_experiment(k, lmda_scale, data_generator, family, n_repeats=3):
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
                k=k,
                lmda_scale=lmda_scale,
                lmda_min_ratio=1e-2,
                n_lmdas=50
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
            print(f"❌ 实验失败 (k={k}, lmda_scale={lmda_scale}, repeat={repeat}): {e}")
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
    output_dir = "results/core_params_tuning"
    os.makedirs(output_dir, exist_ok=True)

    # 参数网格
    k_values = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    lmda_scale_values = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    n_repeats = 3

    # 实验配置 (exp_id, 中文名称, 英文名称, 数据生成函数, 家族)
    experiments = [
        ('exp1', '成对相关稀疏回归', 'Pairwise Correlated Sparse Regression', generate_experiment1_data, 'gaussian'),
        ('exp2', 'AR(1)相关稀疏回归', 'AR(1) Correlated Sparse Regression', generate_experiment2_data, 'gaussian'),
        ('exp3', '二分类偏移变量选择', 'Binary Classification Offset Selection', generate_experiment3_data, 'binomial'),
    ]

    all_results = []

    for exp_id, exp_name_cn, exp_name_en, data_gen, family in experiments:
        print(f"\n{'='*80}")
        print(f"🚀 开始调优实验: {exp_name_cn}")
        print(f"{'='*80}")
        print(f"参数网格: k={k_values}, lmda_scale={lmda_scale_values}")
        print(f"重复次数: {n_repeats}")

        exp_results = []

        total_runs = len(k_values) * len(lmda_scale_values)
        run_idx = 0

        for k in k_values:
            for lmda_scale in lmda_scale_values:
                run_idx += 1
                print(f"\n🔍 {run_idx}/{total_runs}: k={k}, lmda_scale={lmda_scale}")

                metrics = run_single_experiment(k, lmda_scale, data_gen, family, n_repeats)
                if metrics is None:
                    continue

                result_row = {
                    'experiment': exp_id,
                    'experiment_name': exp_name_cn,
                    'experiment_name_en': exp_name_en,
                    'k': k,
                    'lmda_scale': lmda_scale,
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
        exp_df = pd.DataFrame(exp_results)
        exp_df.to_csv(f"{output_dir}/{exp_id}_results.csv", index=False, float_format='%.6f')
        print(f"\n📁 实验结果已保存到: {output_dir}/{exp_id}_results.csv")

        # 生成热力图可视化
        print(f"🎨 生成可视化图表...")
        pivot_f1 = exp_df.pivot(index='k', columns='lmda_scale', values='f1')
        pivot_tpr = exp_df.pivot(index='k', columns='lmda_scale', values='tpr')
        pivot_fdr = exp_df.pivot(index='k', columns='lmda_scale', values='fdr')

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='viridis', ax=axes[0])
        axes[0].set_title(f'{exp_name_en}\nF1 Score', fontsize=14)
        axes[0].set_xlabel('lmda_scale', fontsize=12)
        axes[0].set_ylabel('k (weight exponent)', fontsize=12)

        sns.heatmap(pivot_tpr, annot=True, fmt='.3f', cmap='Blues', ax=axes[1])
        axes[1].set_title(f'{exp_name_en}\nTrue Positive Rate (TPR)', fontsize=14)
        axes[1].set_xlabel('lmda_scale', fontsize=12)
        axes[1].set_ylabel('k (weight exponent)', fontsize=12)

        sns.heatmap(pivot_fdr, annot=True, fmt='.3f', cmap='Reds_r', ax=axes[2])
        axes[2].set_title(f'{exp_name_en}\nFalse Discovery Rate (FDR)', fontsize=14)
        axes[2].set_xlabel('lmda_scale', fontsize=12)
        axes[2].set_ylabel('k (weight exponent)', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{exp_id}_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 生成最优参数
        best_row = exp_df.loc[exp_df['f1'].idxmax()]
        print(f"\n🏆 {exp_name_cn} 最优参数:")
        print(f"   k = {best_row['k']}, lmda_scale = {best_row['lmda_scale']}")
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
    for exp_id in all_df['experiment'].unique():
        exp_df = all_df[all_df['experiment'] == exp_id]
        best_row = exp_df.loc[exp_df['f1'].idxmax()]
        best_params.append({
            'experiment': best_row['experiment'],
            'experiment_name': best_row['experiment_name'],
            'best_k': best_row['k'],
            'best_lmda_scale': best_row['lmda_scale'],
            'best_f1': best_row['f1'],
            'tpr': best_row['tpr'],
            'fdr': best_row['fdr'],
            'n_selected': int(best_row['n_selected']),
            'true_positive': int(best_row['true_positive']),
            **({'mse': best_row['mse']} if 'mse' in best_row else {'accuracy': best_row['accuracy']})
        })

    best_params_df = pd.DataFrame(best_params)
    best_params_df.to_csv(f"{output_dir}/best_parameters.csv", index=False, float_format='%.6f')
    print(f"\n📁 最优参数汇总已保存到: {output_dir}/best_parameters.csv")

    print(f"\n🎉 所有调优实验完成！结果和可视化已保存到 {output_dir} 目录")
    print("\n📊 最优参数汇总:")
    print(best_params_df.to_string(index=False))

if __name__ == "__main__":
    main()
