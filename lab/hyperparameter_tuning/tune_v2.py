#!/usr/bin/env python
"""
XLasso v2.3.2 全模型超参数网格搜索实验
包含模型：
- 标准Lasso
- UniLasso
- XLasso-Soft
- XLasso-GroupDecomp
- XLasso-Full

结果保存目录：results/full_model_tuning_YYYYMMDD/
"""
import sys
import os
from datetime import datetime
# 添加项目根目录和lab目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
lab_dir = os.path.join(project_root, 'lab')
sys.path.insert(0, project_root)
sys.path.insert(0, lab_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso as SkLasso, LogisticRegression
from unilasso.uni_lasso import fit_uni
from experiment_utils import generate_experiment1_data, generate_experiment2_data, generate_experiment3_data, generate_experiment4_data

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

def fit_lasso(X_train, y_train, X_test, y_test, beta_true, family, alpha):
    """拟合标准Lasso模型"""
    if family == 'gaussian':
        model = SkLasso(alpha=alpha, max_iter=10000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        beta_pred = model.coef_
        intercept = model.intercept_
    else:
        # LogisticRegression with L1: C = 1/alpha
        C = 1.0 / alpha if alpha > 0 else 1e10
        model = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=10000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        beta_pred = model.coef_[0]
        intercept = model.intercept_[0]
    return y_pred, beta_pred, intercept

def fit_unilasso(X_train, y_train, X_test, y_test, beta_true, family, lmda_scale):
    """拟合UniLasso模型（k=1e9等价于非负硬约束）"""
    result = fit_uni(
        X_train, y_train,
        family=family,
        k=1e9,
        lmda_scale=lmda_scale,
        lmda_min_ratio=1e-2,
        n_lmdas=50,
        enable_group_decomp=False,
        enable_group_aware_filter=False
    )
    # 在正则化路径上找F1最大的点
    best_f1 = -1
    best_y_pred = None
    best_beta_pred = None
    best_intercept = None

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
            best_y_pred = y_pred
            best_beta_pred = beta_pred
            best_intercept = intercept

    return best_y_pred, best_beta_pred, best_intercept

def fit_xlasso_soft(X_train, y_train, X_test, y_test, beta_true, family, k, lmda_scale):
    """拟合XLasso-Soft模型"""
    result = fit_uni(
        X_train, y_train,
        family=family,
        k=k,
        lmda_scale=lmda_scale,
        lmda_min_ratio=1e-2,
        n_lmdas=50,
        enable_group_decomp=False,
        enable_group_aware_filter=False
    )
    # 在正则化路径上找F1最大的点
    best_f1 = -1
    best_y_pred = None
    best_beta_pred = None
    best_intercept = None

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
            best_y_pred = y_pred
            best_beta_pred = beta_pred
            best_intercept = intercept

    return best_y_pred, best_beta_pred, best_intercept

def fit_xlasso_groupdecomp(X_train, y_train, X_test, y_test, beta_true, family, k, lmda_scale, group_corr_threshold):
    """拟合XLasso-GroupDecomp模型"""
    result = fit_uni(
        X_train, y_train,
        family=family,
        k=k,
        lmda_scale=lmda_scale,
        lmda_min_ratio=1e-2,
        n_lmdas=50,
        enable_group_decomp=True,
        group_corr_threshold=group_corr_threshold,
        enable_group_aware_filter=False
    )
    # 在正则化路径上找F1最大的点
    best_f1 = -1
    best_y_pred = None
    best_beta_pred = None
    best_intercept = None

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
            best_y_pred = y_pred
            best_beta_pred = beta_pred
            best_intercept = intercept

    return best_y_pred, best_beta_pred, best_intercept

def fit_xlasso_full(X_train, y_train, X_test, y_test, beta_true, family, k, lmda_scale, group_corr_threshold, group_filter_k):
    """拟合XLasso-Full模型"""
    result = fit_uni(
        X_train, y_train,
        family=family,
        k=k,
        lmda_scale=lmda_scale,
        lmda_min_ratio=1e-2,
        n_lmdas=50,
        enable_group_decomp=True,
        group_corr_threshold=group_corr_threshold,
        enable_group_aware_filter=True,
        group_filter_k=group_filter_k
    )
    # 在正则化路径上找F1最大的点
    best_f1 = -1
    best_y_pred = None
    best_beta_pred = None
    best_intercept = None

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
            best_y_pred = y_pred
            best_beta_pred = beta_pred
            best_intercept = intercept

    return best_y_pred, best_beta_pred, best_intercept

def run_single_experiment(model_name, model_params, data_generator, family, n_repeats=3):
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
            if model_name == 'Lasso':
                y_pred, beta_pred, _ = fit_lasso(X_train, y_train, X_test, y_test, beta_true, family, **model_params)
            elif model_name == 'UniLasso':
                y_pred, beta_pred, _ = fit_unilasso(X_train, y_train, X_test, y_test, beta_true, family, **model_params)
            elif model_name == 'XLasso-Soft':
                y_pred, beta_pred, _ = fit_xlasso_soft(X_train, y_train, X_test, y_test, beta_true, family, **model_params)
            elif model_name == 'XLasso-GroupDecomp':
                y_pred, beta_pred, _ = fit_xlasso_groupdecomp(X_train, y_train, X_test, y_test, beta_true, family, **model_params)
            elif model_name == 'XLasso-Full':
                y_pred, beta_pred, _ = fit_xlasso_full(X_train, y_train, X_test, y_test, beta_true, family, **model_params)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            metrics = evaluate(y_test, y_pred, beta_true, beta_pred, family)
            all_metrics.append(metrics)
        except Exception as e:
            param_str = ', '.join([f'{k}={v}' for k, v in model_params.items()])
            print(f"❌ 实验失败 ({model_name}, {param_str}, repeat={repeat}): {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_metrics:
        return None

    # 平均多次重复的结果
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_metrics

def get_model_param_grid(model_name, quick_test=False):
    """获取各模型的参数网格"""
    if quick_test:
        # 快速测试用的少量参数
        if model_name == 'Lasso':
            return [{'alpha': a} for a in [0.01, 0.1]]
        elif model_name == 'UniLasso':
            return [{'lmda_scale': ls} for ls in [0.3, 0.5]]
        elif model_name == 'XLasso-Soft':
            return [{'k': k, 'lmda_scale': ls}
                    for k in [0.3, 0.5]
                    for ls in [0.3, 0.5]]
        elif model_name == 'XLasso-GroupDecomp':
            return [{'k': k, 'lmda_scale': ls, 'group_corr_threshold': gct}
                    for k in [0.3, 0.5]
                    for ls in [0.3, 0.5]
                    for gct in [0.7]]
        elif model_name == 'XLasso-Full':
            return [{'k': k, 'lmda_scale': ls, 'group_corr_threshold': gct, 'group_filter_k': gfk}
                    for k in [0.3]
                    for ls in [0.5]
                    for gct in [0.7]
                    for gfk in [1, None]]
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # 完整参数网格
    if model_name == 'Lasso':
        return [{'alpha': a} for a in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]]
    elif model_name == 'UniLasso':
        return [{'lmda_scale': ls} for ls in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]]
    elif model_name == 'XLasso-Soft':
        return [{'k': k, 'lmda_scale': ls}
                for k in [0.3, 0.5, 0.8, 1.0, 1.5]
                for ls in [0.2, 0.3, 0.5, 0.8, 1.0]]
    elif model_name == 'XLasso-GroupDecomp':
        return [{'k': k, 'lmda_scale': ls, 'group_corr_threshold': gct}
                for k in [0.3, 0.5, 0.8, 1.0]
                for ls in [0.2, 0.3, 0.5, 0.8]
                for gct in [0.6, 0.7, 0.8]]
    elif model_name == 'XLasso-Full':
        return [{'k': k, 'lmda_scale': ls, 'group_corr_threshold': gct, 'group_filter_k': gfk}
                for k in [0.3, 0.5, 0.8]
                for ls in [0.2, 0.3, 0.5, 0.8]
                for gct in [0.6, 0.7, 0.8]
                for gfk in [1, 2, None]]
    else:
        raise ValueError(f"Unknown model: {model_name}")

def plot_model_comparison(all_df, output_dir):
    """绘制模型对比图"""
    models = ['Lasso', 'UniLasso', 'XLasso-Soft', 'XLasso-GroupDecomp', 'XLasso-Full']
    experiments = all_df['experiment'].unique()
    n_exps = len(experiments)

    # 1. F1对比柱状图（2行2列布局）
    n_rows = 2 if n_exps > 3 else 1
    n_cols = 2 if n_exps > 3 else n_exps
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12 if n_exps > 3 else 6))
    axes_flat = axes.flatten() if n_exps > 3 else [axes] if n_exps == 1 else axes

    for idx, exp_id in enumerate(experiments):
        exp_df = all_df[all_df['experiment'] == exp_id]
        best_per_model = []
        for model in models:
            model_df = exp_df[exp_df['model'] == model]
            if len(model_df) > 0:
                best_row = model_df.loc[model_df['f1'].idxmax()]
                best_per_model.append({'model': model, 'f1': best_row['f1']})
        plot_df = pd.DataFrame(best_per_model)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        ax = axes_flat[idx]
        sns.barplot(data=plot_df, x='model', y='f1', ax=ax, palette=colors, hue='model', legend=False)
        ax.set_title(f'{exp_id}\nBest F1 Score', fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylim(0, max(plot_df['f1']) * 1.2 if len(plot_df) > 0 else 1)
        for i, row in plot_df.iterrows():
            ax.text(i, row['f1'] + (max(plot_df['f1']) * 0.02 if len(plot_df) > 0 else 0.02),
                          f'{row["f1"]:.3f}', ha='center', fontsize=10)

    # 隐藏多余的子图
    for idx in range(n_exps, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_f1.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. TPR-FDR散点图（2行2列布局）
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12 if n_exps > 3 else 6))
    axes_flat = axes.flatten() if n_exps > 3 else [axes] if n_exps == 1 else axes

    for idx, exp_id in enumerate(experiments):
        exp_df = all_df[all_df['experiment'] == exp_id]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        ax = axes_flat[idx]
        for model_idx, model in enumerate(models):
            model_df = exp_df[exp_df['model'] == model]
            if len(model_df) > 0:
                ax.scatter(model_df['fdr'], model_df['tpr'], label=model,
                                  color=colors[model_idx], alpha=0.6, s=50)
                # 标记最优F1点
                best_row = model_df.loc[model_df['f1'].idxmax()]
                ax.scatter(best_row['fdr'], best_row['tpr'], color=colors[model_idx],
                                  s=150, edgecolors='black', linewidths=2, zorder=10)
        ax.set_title(f'{exp_id}\nTPR vs FDR', fontsize=14)
        ax.set_xlabel('False Discovery Rate (FDR)', fontsize=12)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

    # 隐藏多余的子图
    for idx in range(n_exps, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_tpr_fdr.png', dpi=150, bbox_inches='tight')
    plt.close()

def main(quick_test=False):
    # 创建带日期的输出目录
    date_str = datetime.now().strftime('%Y%m%d')
    suffix = "_quick" if quick_test else ""
    output_dir = f"results/full_model_tuning_{date_str}{suffix}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 结果将保存到: {output_dir}")
    if quick_test:
        print("⚡ 快速测试模式：使用少量参数")

    # 模型配置
    models = [
        'Lasso',
        'UniLasso',
        'XLasso-Soft',
        'XLasso-GroupDecomp',
        'XLasso-Full',
    ]

    n_repeats = 1 if quick_test else 3

    # 实验配置
    experiments = [
        ('exp1', '成对相关稀疏回归', 'Pairwise Correlated Sparse Regression', generate_experiment1_data, 'gaussian'),
        ('exp2', 'AR(1)相关稀疏回归', 'AR(1) Correlated Sparse Regression', generate_experiment2_data, 'gaussian'),
        ('exp3', '二分类偏移变量选择', 'Binary Classification Offset Selection', generate_experiment3_data, 'binomial'),
        ('exp4', '孪生变量反符号选择', 'Twin Variables Opposite Sign Selection', generate_experiment4_data, 'gaussian'),
    ]

    all_results = []

    for model_name in models:
        print(f"\n{'='*80}")
        print(f"🚀 开始调优模型: {model_name}")
        print(f"{'='*80}")

        param_grid = get_model_param_grid(model_name, quick_test=quick_test)
        print(f"参数组合数: {len(param_grid)}")

        for exp_id, exp_name_cn, exp_name_en, data_gen, family in experiments:
            print(f"\n{'='*80}")
            print(f"📊 实验: {exp_name_cn}")
            print(f"{'='*80}")

            total_runs = len(param_grid)
            run_idx = 0

            for params in param_grid:
                run_idx += 1
                param_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                print(f"\n🔍 {run_idx}/{total_runs}: {param_str}")

                metrics = run_single_experiment(model_name, params, data_gen, family, n_repeats)
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
                all_results.append(result_row)

                # 打印结果
                if family == 'gaussian':
                    print(f"✅ 完成: F1={metrics['f1']:.4f}, MSE={metrics['mse']:.4f}, TPR={metrics['tpr']:.4f}, FDR={metrics['fdr']:.4f}, 选中={metrics['n_selected']}, 正确={metrics['true_positive']}")
                else:
                    print(f"✅ 完成: F1={metrics['f1']:.4f}, 准确率={metrics['accuracy']:.4f}, TPR={metrics['tpr']:.4f}, FDR={metrics['fdr']:.4f}, 选中={metrics['n_selected']}, 正确={metrics['true_positive']}")

    # 保存所有结果
    all_df = pd.DataFrame(all_results)
    all_df.to_csv(f"{output_dir}/all_experiments_results.csv", index=False, float_format='%.6f')
    print(f"\n📁 所有实验结果已保存到: {output_dir}/all_experiments_results.csv")

    # 生成汇总最优参数表
    best_params = []
    for model_name in models:
        for exp_id, exp_name_cn, exp_name_en, _, _ in experiments:
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
            for k in ['alpha', 'k', 'lmda_scale', 'group_corr_threshold', 'group_filter_k']:
                if k in best_row and not pd.isna(best_row[k]):
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

    # 生成可视化对比图
    print(f"\n🎨 生成模型对比可视化...")
    plot_model_comparison(all_df, output_dir)

    print(f"\n🎉 所有调优实验完成！结果和可视化已保存到 {output_dir} 目录")
    print("\n📊 最优参数汇总:")
    print(best_params_df.to_string(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Full model tuning')
    parser.add_argument('--quick', action='store_true', help='Quick test mode with few parameters')
    args = parser.parse_args()
    main(quick_test=args.quick)
