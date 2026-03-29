"""
Benchmark Visualization Script
Generated for: adaptive_flipped_lasso_exp2_benchmark__20260326_231735
"""

import sys
sys.path.insert(0, '/home/lili/lyn/clear/NLasso/XLasso/experiments')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from viz import plot_metric_compare, plot_bar_chart
import warnings
warnings.filterwarnings('ignore')

# Load data
work_dir = '/home/lili/lyn/clear/NLasso/XLasso/experiments/analysis/exp2_bechmark/adaptive_flipped_lasso_exp2_benchmark__20260326_231735'
summary = pd.read_csv(f'{work_dir}/summary_by_sigma_model.csv')

print("Available columns:", summary.columns.tolist())
print("Models:", summary['model'].unique())
print("Sigma values:", summary['sigma'].unique())
print()

# Set style
plt.style.use('/home/lili/lyn/clear/NLasso/XLasso/experiments/viz/themes/paper.mplstyle')

# 1. F1 Score (up) and MSE (down) - diverging bar chart with shared x-axis
fig, ax = plt.subplots(figsize=(12, 7))

# Normalize MSE to [0, 1] scale for visualization
pivot_f1 = summary.pivot(index='sigma', columns='model', values='f1_mean')
pivot_mse_raw = summary.pivot(index='sigma', columns='model', values='mse_mean')
pivot_mse = pivot_mse_raw / pivot_mse_raw.max().max()  # normalize to [0, 1]

n_sigma = len(pivot_f1.index)
models = pivot_f1.columns.tolist()
n_models = len(models)
bar_width = 0.8 / n_models
x_pos = np.arange(n_sigma)

colors = plt.cm.Set2(np.linspace(0, 1, n_models))

for i, model in enumerate(models):
    offset = (i - n_models / 2 + 0.5) * bar_width
    # F1 bars (upward)
    ax.bar(x_pos + offset, pivot_f1[model].values, bar_width,
           color=colors[i], label=model, alpha=0.9,
           edgecolor='black', linewidth=0.5)
    # MSE bars (downward, using negative values)
    ax.bar(x_pos + offset, -pivot_mse[model].values, bar_width,
           color=colors[i], alpha=0.5,
           edgecolor='black', linewidth=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels([f'σ={s}\nSNR={summary[summary["sigma"]==s]["snr"].iloc[0]}' for s in pivot_f1.index], fontsize=10)
ax.set_ylabel('')
ax.set_xlabel('')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Add axis labels on the left side
ax.text(-0.05, 0.7, 'F1', transform=ax.transAxes, fontsize=13,
        va='center', ha='center', fontweight='bold', rotation=90, color='darkblue')
ax.text(-0.05, 0.3, 'MSE normalized', transform=ax.transAxes, fontsize=13,
        va='center', ha='center', fontweight='bold', rotation=90, color='darkgreen')


# Custom legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:n_models], labels[:n_models],
          loc='upper right',
          ncol=1, fontsize=9)

ax.set_ylim([-1.2, 1.2])
# Fix y-axis ticks to show positive values for the lower half
yticks = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks(yticks)
ax.set_yticklabels([f'{abs(y):.1f}' for y in yticks])
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{work_dir}/f1_mse_combined.png', dpi=150, bbox_inches='tight')
plt.show()

# 3. R² comparison by sigma (using same color scheme)
fig, ax = plt.subplots(figsize=(10, 6))

pivot_r2 = summary.pivot(index='sigma', columns='model', values='r2_mean')
pivot_r2_std = summary.pivot(index='sigma', columns='model', values='r2_std')

n_sigma = len(pivot_r2.index)
models = pivot_r2.columns.tolist()
n_models = len(models)
bar_width = 0.8 / n_models
x_pos = np.arange(n_sigma)

colors = plt.cm.Set2(np.linspace(0, 1, n_models))

for i, model in enumerate(models):
    offset = (i - n_models / 2 + 0.5) * bar_width
    ax.bar(x_pos + offset, pivot_r2[model].values, bar_width,
           color=colors[i], label=model, alpha=0.9,
           edgecolor='black', linewidth=0.5,
           yerr=pivot_r2_std[model].values, capsize=3)

ax.set_xticks(x_pos)
ax.set_xticklabels([f'σ={s}\nSNR={summary[summary["sigma"]==s]["snr"].iloc[0]}' for s in pivot_r2.index], fontsize=10)
ax.set_xlabel('')
ax.set_ylabel('R²', fontsize=14)
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim([0, 1.15])
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{work_dir}/r2_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 4. FDR comparison by sigma
fig, ax = plt.subplots(figsize=(10, 6))
pivot_fdr = summary.pivot(index='sigma', columns='model', values='fdr_mean')
pivot_fdr.plot(kind='bar', ax=ax, yerr=summary.pivot(index='sigma', columns='model', values='fdr_std'))
ax.set_xlabel('Sigma')
ax.set_ylabel('FDR')
ax.set_title('False Discovery Rate by Sigma')
ax.legend(title='Model', loc='upper right')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{work_dir}/fdr_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 5. TPR comparison by sigma
fig, ax = plt.subplots(figsize=(10, 6))
pivot_tpr = summary.pivot(index='sigma', columns='model', values='tpr_mean')
pivot_tpr.plot(kind='bar', ax=ax, yerr=summary.pivot(index='sigma', columns='model', values='tpr_std'))
ax.set_xlabel('Sigma')
ax.set_ylabel('TPR')
ax.set_title('True Positive Rate by Sigma')
ax.legend(title='Model', loc='upper right')
ax.set_ylim([0, 1.1])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{work_dir}/tpr_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 6. Scatter plots from raw data (individual files)
raw = pd.read_csv(f'{work_dir}/raw.csv')

metrics = ['f1', 'mse', 'r2', 'fdr', 'tpr', 'precision']
titles = ['F1 Score', 'MSE', 'R²', 'FDR', 'TPR', 'Precision']
colors_raw = plt.cm.Set2(np.linspace(0, 1, len(raw['model'].unique())))
model_color_map = dict(zip(raw['model'].unique(), colors_raw))

np.random.seed(42)
for metric, title in zip(metrics, titles):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in raw['model'].unique():
        data = raw[raw['model'] == model].copy()
        jitter = np.random.uniform(-0.08, 0.08, len(data))
        ax.scatter(data['sigma'] + jitter, data[metric], c=[model_color_map[model]], label=model, alpha=0.7, s=30)
    ax.set_xlabel('Sigma', fontsize=11)
    ax.set_ylabel(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{work_dir}/scatter_{metric}.png', dpi=150, bbox_inches='tight')
    plt.show()

# 6. Aggregate metrics across all sigma (mean values)
print("=== Aggregate Performance ===")
agg_metrics = summary.groupby('model').agg({
    'f1_mean': 'mean',
    'mse_mean': 'mean',
    'r2_mean': 'mean',
    'fdr_mean': 'mean',
    'tpr_mean': 'mean'
}).round(4)
print(agg_metrics)

# 7. F1 and MSE vs params (exp1 data)
exp1_path = '/home/lili/lyn/clear/NLasso/XLasso/experiments/analysis/exp1/adaptive_flipped_lasso_exp1/adaptive_flipped_lasso_exp1_raw.csv'
exp1 = pd.read_csv(exp1_path)

print("Exp1 columns:", exp1.columns.tolist())
print("Lambda ridge values:", sorted(exp1['params_lambda_ridge'].unique()))
print("Lambda values:", sorted(exp1['params_lambda_'].unique()))
print("Gamma values:", sorted(exp1['params_gamma'].unique()))

# Color scheme
gamma_colors = plt.cm.Set2(np.linspace(0, 1, len(exp1['params_gamma'].unique())))
gamma_color_map = dict(zip(sorted(exp1['params_gamma'].unique()), gamma_colors))
ridge_colors = plt.cm.Set2(np.linspace(0, 1, len(exp1['params_lambda_ridge'].unique())))
ridge_color_map = dict(zip(sorted(exp1['params_lambda_ridge'].unique()), ridge_colors))

# F1 vs lambda_ridge
fig, ax = plt.subplots(figsize=(8, 5))
for gamma in sorted(exp1['params_gamma'].unique()):
    data = exp1[exp1['params_gamma'] == gamma]
    means = data.groupby('params_lambda_ridge')['f1'].mean()
    ax.plot(means.index, means.values, label=f'gamma={gamma}', marker='o', color=gamma_color_map[gamma])
ax.set_xlabel('Lambda Ridge', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 vs Lambda Ridge', fontsize=13)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{work_dir}/f1_vs_lambda_ridge.png', dpi=150, bbox_inches='tight')
plt.show()

# F1 vs lambda_
fig, ax = plt.subplots(figsize=(8, 5))
for gamma in sorted(exp1['params_gamma'].unique()):
    data = exp1[exp1['params_gamma'] == gamma]
    means = data.groupby('params_lambda_')['f1'].mean()
    ax.plot(means.index, means.values, label=f'gamma={gamma}', marker='o', color=gamma_color_map[gamma])
ax.set_xlabel('Lambda', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 vs Lambda', fontsize=13)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig(f'{work_dir}/f1_vs_lambda.png', dpi=150, bbox_inches='tight')
plt.show()

# F1 vs gamma
fig, ax = plt.subplots(figsize=(8, 5))
for lambda_ridge in sorted(exp1['params_lambda_ridge'].unique()):
    data = exp1[exp1['params_lambda_ridge'] == lambda_ridge]
    means = data.groupby('params_gamma')['f1'].mean()
    ax.plot(means.index, means.values, label=f'lambda_ridge={lambda_ridge}', marker='o', color=ridge_color_map[lambda_ridge])
ax.set_xlabel('Gamma', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 vs Gamma', fontsize=13)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{work_dir}/f1_vs_gamma.png', dpi=150, bbox_inches='tight')
plt.show()

# MSE vs lambda_ridge
fig, ax = plt.subplots(figsize=(8, 5))
for gamma in sorted(exp1['params_gamma'].unique()):
    data = exp1[exp1['params_gamma'] == gamma]
    means = data.groupby('params_lambda_ridge')['mse'].mean()
    ax.plot(means.index, means.values, label=f'gamma={gamma}', marker='o', color=gamma_color_map[gamma])
ax.set_xlabel('Lambda Ridge', fontsize=12)
ax.set_ylabel('MSE', fontsize=12)
ax.set_title('MSE vs Lambda Ridge', fontsize=13)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{work_dir}/mse_vs_lambda_ridge.png', dpi=150, bbox_inches='tight')
plt.show()

# MSE vs lambda_
fig, ax = plt.subplots(figsize=(8, 5))
for gamma in sorted(exp1['params_gamma'].unique()):
    data = exp1[exp1['params_gamma'] == gamma]
    means = data.groupby('params_lambda_')['mse'].mean()
    ax.plot(means.index, means.values, label=f'gamma={gamma}', marker='o', color=gamma_color_map[gamma])
ax.set_xlabel('Lambda', fontsize=12)
ax.set_ylabel('MSE', fontsize=12)
ax.set_title('MSE vs Lambda', fontsize=13)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig(f'{work_dir}/mse_vs_lambda.png', dpi=150, bbox_inches='tight')
plt.show()

# MSE vs gamma
fig, ax = plt.subplots(figsize=(8, 5))
for lambda_ridge in sorted(exp1['params_lambda_ridge'].unique()):
    data = exp1[exp1['params_lambda_ridge'] == lambda_ridge]
    means = data.groupby('params_gamma')['mse'].mean()
    ax.plot(means.index, means.values, label=f'lambda_ridge={lambda_ridge}', marker='o', color=ridge_color_map[lambda_ridge])
ax.set_xlabel('Gamma', fontsize=12)
ax.set_ylabel('MSE', fontsize=12)
ax.set_title('MSE vs Gamma', fontsize=13)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{work_dir}/mse_vs_gamma.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nExp1 Visualizations saved to: {work_dir}/")
print("Files: f1_vs_params.png, mse_vs_params.png")