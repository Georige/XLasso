#!/usr/bin/env python
"""
实验结果可视化工具
自动读取实验目录，生成各算法指标对比图
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from experiment_utils import EXPERIMENT_CONFIG

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

def parse_args():
    parser = argparse.ArgumentParser(description='实验结果可视化')
    parser.add_argument('--result-dir', '-d', type=str, required=True,
                        help='实验结果目录，例如：result/comprehensive_experiments/simulation_results_20260321_133824')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='图片输出目录，默认保存在实验目录下的plots子目录')
    return parser.parse_args()

def load_experiment_results(result_dir):
    """加载所有算法的实验结果"""
    all_results = []

    # 遍历所有算法目录
    for algo_dir in sorted(os.listdir(result_dir)):
        algo_path = os.path.join(result_dir, algo_dir)
        if not os.path.isdir(algo_path):
            continue

        # 查找raw.csv文件
        csv_files = [f for f in os.listdir(algo_path) if f.endswith('_raw.csv')]
        if not csv_files:
            continue

        csv_path = os.path.join(algo_path, csv_files[0])
        df = pd.read_csv(csv_path)

        # 提取算法名称（从目录名）
        algo_name = algo_dir.split('_', 1)[1].replace('_', ' ')
        df['algorithm'] = algo_name

        all_results.append(df)

    if not all_results:
        raise ValueError(f"在 {result_dir} 中没有找到实验结果")

    return pd.concat(all_results, ignore_index=True)

def plot_metric_comparison(df, metric_name, metric_display, ylabel, title, output_path, reverse_y=False):
    """绘制指定指标的对比柱状图"""
    # 统计每个算法在每个sigma下的均值和标准差
    summary = df.groupby(['algorithm', 'sigma'])[metric_name].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(12, 6))

    # 获取唯一的算法和sigma值
    algorithms = sorted(summary['algorithm'].unique())
    sigmas = sorted(summary['sigma'].unique())
    n_algos = len(algorithms)
    n_sigmas = len(sigmas)
    bar_width = 0.8 / n_algos

    # 绘制柱状图
    for i, algo in enumerate(algorithms):
        algo_data = summary[summary['algorithm'] == algo]
        x = np.arange(n_sigmas) + i * bar_width
        plt.bar(x, algo_data['mean'], width=bar_width, label=algo, yerr=algo_data['std'], capsize=3)

    plt.xlabel('噪声水平 σ', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.xticks(np.arange(n_sigmas) + bar_width * (n_algos - 1) / 2, [f'σ={s}' for s in sigmas])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    if reverse_y:
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ 已生成: {output_path}")

def plot_f1_vs_sigma(df, output_path):
    """绘制F1分数随sigma变化的折线图"""
    plt.figure(figsize=(10, 6))

    algorithms = sorted(df['algorithm'].unique())
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo].groupby('sigma')['f1'].agg(['mean', 'std']).reset_index()
        plt.errorbar(algo_data['sigma'], algo_data['mean'], yerr=algo_data['std'], marker='o', label=algo, capsize=3)

    plt.xlabel('噪声水平 σ', fontsize=12)
    plt.ylabel('F1分数', fontsize=12)
    plt.title('不同噪声水平下各算法F1分数对比', fontsize=14, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ 已生成: {output_path}")

def plot_fdr_vs_tpr(df, output_path):
    """绘制FDR-TPR散点图（越靠近左上角越好）"""
    plt.figure(figsize=(8, 8))

    # 统计每个算法的平均FDR和TPR
    summary = df.groupby('algorithm')[['fdr', 'tpr']].mean().reset_index()

    plt.scatter(summary['fdr'], summary['tpr'], s=100, alpha=0.7)

    # 添加算法标签
    for _, row in summary.iterrows():
        plt.annotate(row['algorithm'], (row['fdr'], row['tpr']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.xlabel('假发现率 FDR (越小越好)', fontsize=12)
    plt.ylabel('真阳性率 TPR (越大越好)', fontsize=12)
    plt.title('各算法变量选择性能对比 (FDR vs TPR)', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ 已生成: {output_path}")

def generate_summary_table(df, output_path):
    """生成汇总表格"""
    # 按算法和sigma分组统计
    summary = df.groupby(['algorithm', 'family', 'sigma']).agg({
        'mse': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'auc': ['mean', 'std'],
        'tpr': ['mean', 'std'],
        'fdr': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'n_selected': ['mean', 'std'],
        'time_seconds': ['mean', 'std']
    }).round(4)

    # 保存为csv
    summary.to_csv(output_path, encoding='utf-8-sig')
    print(f"✅ 已生成汇总表格: {output_path}")

    # 同时生成markdown格式
    md_path = output_path.replace('.csv', '.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 实验结果汇总表\n\n")
        f.write(summary.to_markdown())
    print(f"✅ 已生成Markdown汇总表: {md_path}")

def main():
    args = parse_args()

    # 加载结果
    print(f"📂 加载实验结果: {args.result_dir}")
    df = load_experiment_results(args.result_dir)

    # 设置输出目录
    if args.output_dir is None:
        output_dir = os.path.join(args.result_dir, 'plots')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"🖼️  图片将保存到: {output_dir}")

    # 生成汇总表
    generate_summary_table(df, os.path.join(output_dir, 'summary_table.csv'))

    # 按实验类型和任务类型分别绘图
    experiments = df['experiment_id'].unique()
    families = df['family'].unique()

    for exp_id in experiments:
        exp_name = df[df['experiment_id'] == exp_id]['experiment_name'].iloc[0]
        for family in families:
            subset = df[(df['experiment_id'] == exp_id) & (df['family'] == family)]
            if len(subset) == 0:
                continue

            exp_prefix = f"{exp_id}_{family}"
            task_type = "回归" if family == 'gaussian' else "分类"

            print(f"\n📊 生成 {exp_name} ({task_type}) 图表...")

            # 回归任务指标
            if family == 'gaussian':
                # MSE对比
                plot_metric_comparison(
                    subset, 'mse', 'MSE', '均方误差 (越小越好)',
                    f'{exp_name} - 均方误差(MSE)对比',
                    os.path.join(output_dir, f'{exp_prefix}_mse_comparison.png'),
                    reverse_y=True
                )

            # 分类任务指标
            if family == 'binomial':
                # 准确率对比
                plot_metric_comparison(
                    subset, 'accuracy', 'Accuracy', '准确率 (越大越好)',
                    f'{exp_name} - 准确率(Accuracy)对比',
                    os.path.join(output_dir, f'{exp_prefix}_accuracy_comparison.png')
                )

                # AUC对比
                plot_metric_comparison(
                    subset, 'auc', 'AUC', 'AUC (越大越好)',
                    f'{exp_name} - AUC对比',
                    os.path.join(output_dir, f'{exp_prefix}_auc_comparison.png')
                )

            # 通用指标
            # F1对比
            plot_metric_comparison(
                subset, 'f1', 'F1', 'F1分数 (越大越好)',
                f'{exp_name} - F1分数对比',
                os.path.join(output_dir, f'{exp_prefix}_f1_comparison.png')
            )

            # FDR对比
            plot_metric_comparison(
                subset, 'fdr', 'FDR', '假发现率 (越小越好)',
                f'{exp_name} - 假发现率(FDR)对比',
                os.path.join(output_dir, f'{exp_prefix}_fdr_comparison.png'),
                reverse_y=True
            )

            # TPR对比
            plot_metric_comparison(
                subset, 'tpr', 'TPR', '真阳性率 (越大越好)',
                f'{exp_name} - 真阳性率(TPR)对比',
                os.path.join(output_dir, f'{exp_prefix}_tpr_comparison.png')
            )

            # F1随sigma变化折线图
            plot_f1_vs_sigma(
                subset,
                os.path.join(output_dir, f'{exp_prefix}_f1_vs_sigma.png')
            )

            # FDR vs TPR散点图
            plot_fdr_vs_tpr(
                subset,
                os.path.join(output_dir, f'{exp_prefix}_fdr_vs_tpr.png')
            )

    print(f"\n🎉 所有图表生成完成! 共生成 {len(os.listdir(output_dir))} 个文件")
    print(f"📂 结果目录: {output_dir}")

if __name__ == "__main__":
    main()
