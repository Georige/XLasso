"""
综合实验工具函数
包含所有公共的实验配置、数据生成、指标计算、结果保存等功能
"""
import os
import sys
import time
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, accuracy_score, roc_auc_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from unilasso.uni_lasso import fit_uni, cv_uni
from other_lasso import (
    AdaptiveLassoCV, GroupLassoCV, FusedLassoCV,
    AdaptiveSparseGroupLassoCV
)
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from lab.preprocessing import GroupPreprocessor

# ------------------------------------------------------------------------------
# 实验配置
# ------------------------------------------------------------------------------
EXPERIMENT_CONFIG = {
    'random_state': 2026,  # 基准随机种子，重复n次对应种子为2026~2026+n-1
    'n_repeats': 3,  # 每个实验重复次数，优化后减少到3次加速运行
    'test_size': 0.3,
    'cv_folds': 3,  # 交叉验证折数
    'standardize': True,
    'n_jobs': -1,  # 并行数，-1使用所有CPU
    'save_dir': 'result/comprehensive_experiments/',  # 结果保存目录
    'version': 'v2.2.0',
    'author': 'XLasso Team'
}

# 算法配置（按用户指定优先级排序：UniLasso → Lasso → XLasso全系 → Adaptive Lasso → Fused Lasso → Group Lasso → Adaptive Sparse Group Lasso）
# 调参原则：按场景固定结构参数(k/enable_group_decomp等)，仅动态选择λ
ALGORITHMS = {
    # 优先级1: 原始UniLasso
    '原始UniLasso': {
        'class': 'xlasso',
        'params': {
            'adaptive_weighting': False,
            'enable_group_decomp': False,
        },
        'task_type': ['classification', 'regression']
    },

    # 优先级2: 标准Lasso
    '标准Lasso': {
        'class': LogisticRegressionCV,
        'params': {
            'penalty': 'l1',
            'cv': EXPERIMENT_CONFIG['cv_folds'],
            'max_iter': 1000,
            'solver': 'liblinear',
            'n_jobs': EXPERIMENT_CONFIG['n_jobs'],
            'random_state': EXPERIMENT_CONFIG['random_state']
        },
        'task_type': ['classification', 'regression']  # 回归用LassoCV
    },

    # 优先级3: XLasso全系（结构参数固定，仅λ通过cv_uni做3折交叉验证）
    # 调参原则：按场景固定结构参数(k/enable_group_decomp等)，仅动态选择λ
    'XLasso-Soft': {
        'class': 'xlasso_cv',
        'params': {
            'enable_group_decomp': False,
            'k': 1.0,                   # 结构参数：中等非对称惩罚强度
            'lmda_min_ratio': 1e-2,
            'n_lmdas': 100,
            'cv_folds': EXPERIMENT_CONFIG['cv_folds'],
        },
        'task_type': ['classification', 'regression']
    },
    'XLasso-Soft-γ0.5': {
        'class': 'xlasso_cv',
        'params': {
            'enable_group_decomp': False,
            'k': 0.5,                   # 结构参数：弱非对称，接近对称Lasso
            'lmda_min_ratio': 1e-2,
            'n_lmdas': 100,
            'cv_folds': EXPERIMENT_CONFIG['cv_folds'],
        },
        'task_type': ['classification', 'regression']
    },
    'XLasso-Soft-γ2.0': {
        'class': 'xlasso_cv',
        'params': {
            'enable_group_decomp': False,
            'k': 2.0,                   # 结构参数：强非对称，对负系数更敏感
            'lmda_min_ratio': 1e-2,
            'n_lmdas': 100,
            'cv_folds': EXPERIMENT_CONFIG['cv_folds'],
        },
        'task_type': ['classification', 'regression']
    },
    'XLasso-GroupDecomp': {
        'class': 'xlasso_cv',
        'params': {
            'enable_group_decomp': True,
            'group_corr_threshold': 0.7,
            'enable_group_aware_filter': False,
            'k': 1.0,                   # 结构参数：启用组级正交分解
            'lmda_min_ratio': 1e-2,
            'n_lmdas': 100,
            'cv_folds': EXPERIMENT_CONFIG['cv_folds'],
        },
        'task_type': ['classification', 'regression']
    },
    'XLasso-Full': {
        'class': 'xlasso_cv',
        'params': {
            'enable_group_decomp': True,
            'group_corr_threshold': 0.7,
            'enable_group_aware_filter': True,
            'k': 1.0,                   # 结构参数：启用组分解+感知过滤
            'lmda_min_ratio': 1e-2,
            'n_lmdas': 100,
            'cv_folds': EXPERIMENT_CONFIG['cv_folds'],
        },
        'task_type': ['classification', 'regression']
    },

    # XLasso with Pre-Decomposition (前置正交分解预处理)
    'XLasso-PreDecomp': {
        'class': 'xlasso_predecomp',
        'params': {
            'corr_threshold': 0.5,        # 相关性阈值
            'max_group_size': 20,
            'min_explained_variance_ratio': 0.7,
            'k': 1.0,                   # gamma for adaptive weighting
            'lmda_min_ratio': 1e-2,
            'n_lmdas': 100,
            'cv_folds': EXPERIMENT_CONFIG['cv_folds'],
        },
        'task_type': ['classification', 'regression']
    },

    # 优先级4: Adaptive Lasso
    'Adaptive Lasso': {
        'class': AdaptiveLassoCV,
        'params': {
            'gammas': [0.5, 1.0, 2.0],
            'cv': EXPERIMENT_CONFIG['cv_folds'],
            'n_jobs': EXPERIMENT_CONFIG['n_jobs'],
            'max_iter': 1000,
        },
        'task_type': ['classification', 'regression']
    },

    # 优先级5: Fused Lasso
    'Fused Lasso': {
        'class': FusedLassoCV,
        'params': {
            'lambda_fused_ratios': [0.1, 0.5, 1.0, 2.0],
            'cv': EXPERIMENT_CONFIG['cv_folds'],
            'n_jobs': EXPERIMENT_CONFIG['n_jobs'],
            'max_iter': 1000,
        },
        'task_type': ['classification', 'regression']
    },

    # 优先级6: Group Lasso
    'Group Lasso': {
        'class': GroupLassoCV,
        'params': {
            'cv': EXPERIMENT_CONFIG['cv_folds'],
            'n_jobs': EXPERIMENT_CONFIG['n_jobs'],
            'max_iter': 1000,
        },
        'need_groups': True,  # 需要自动分组
        'task_type': ['classification', 'regression']
    },

    # 优先级7: Adaptive Sparse Group Lasso
    'Adaptive Sparse Group Lasso': {
        'class': AdaptiveSparseGroupLassoCV,
        'params': {
            'l1_ratios': [0.1, 0.5, 0.9],
            'cv': EXPERIMENT_CONFIG['cv_folds'],
            'n_jobs': EXPERIMENT_CONFIG['n_jobs'],
            'max_iter': 500,
        },
        'need_groups': True,
        'task_type': ['classification', 'regression']
    },

    # 辅助基准
    '逻辑回归(L2)': {
        'class': LogisticRegressionCV,
        'params': {
            'penalty': 'l2',
            'cv': EXPERIMENT_CONFIG['cv_folds'],
            'max_iter': 1000,
            'solver': 'liblinear',
            'n_jobs': EXPERIMENT_CONFIG['n_jobs'],
            'random_state': EXPERIMENT_CONFIG['random_state']
        },
        'task_type': ['classification']
    },
}

# ------------------------------------------------------------------------------
# 数据生成函数
# ------------------------------------------------------------------------------
def generate_experiment1_data(n=300, p=500, sigma=1.0, family='gaussian'):
    """实验1：高维成对相关稀疏回归 (paper.md Section 5.1.1)
    n=300, p=500, X~N(0,Σ) with Σ_ij=0.5, first 20 variables β=1.0, σ=1.0
    """
    # 协方差矩阵：成对相关0.5
    cov = np.ones((p, p)) * 0.5
    np.fill_diagonal(cov, 1.0)
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 真实系数：前20个变量 β=1.0，其余480个为0
    beta_true = np.zeros(p)
    beta_true[:20] = 1.0

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:  # binomial
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true

def generate_experiment2_data(n=300, p=500, sigma=1.0, rho=0.8, family='gaussian'):
    """实验2：AR(1)相关稀疏回归 (paper.md Section 5.1.2)
    n=300, p=500, X~N(0,Σ) with Σ_ij=0.8^|i-j|, odd-indexed first 20 variables β=1.0, σ=1.0
    """
    # AR(1)协方差矩阵
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = rho ** abs(i - j)
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 真实系数：奇数索引的前20个变量 β=1.0 (j=1,3,5,...,39)，其余480个为0
    beta_true = np.zeros(p)
    beta_true[1:40:2] = 1.0

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true

def generate_experiment3_data(n=300, p=500, sigma=1.0, rho=0.8, family='binomial'):
    """实验3：二分类偏移变量选择 (paper.md Section 5.1.3)
    n=300, p=500, X~N(0,Σ) with AR(1) ρ=0.8, first 20 variables β=1.0
    y=1 samples have offset 0.6 on first 20 variables, ~150 each class
    """
    # AR(1)相关性结构
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = rho ** abs(i - j)
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 真实系数：前20个变量 β=1.0，其余480个为0
    beta_true = np.zeros(p)
    beta_true[:20] = 1.0

    if family == 'binomial':
        # 二分类：基于系数生成y，再对y=1样本施加偏移
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
        # 确保类别平衡：约150个y=1，约150个y=0
        y1_idx = y == 1
        if np.sum(y1_idx) > 160:
            # 如果y=1过多，随机降采样
            drop_idx = np.random.choice(np.where(y1_idx)[0], int(np.sum(y1_idx) - 150), replace=False)
            y[drop_idx] = 0
        elif np.sum(y1_idx) < 140:
            # 如果y=1过少，随机上采样一些y=1
            add_idx = np.random.choice(np.where(y == 0)[0], int(150 - np.sum(y1_idx)), replace=False)
            y[add_idx] = 1
        # 对y=1的样本，前20个变量增加偏移量0.6
        y1_idx = y == 1
        X[y1_idx, :20] += 0.6
    else:
        # 回归任务
        y = X @ beta_true + np.random.randn(n) * sigma

    return X, y, beta_true


def generate_experiment4_data(n=300, p=1000, sigma=1.0, rho=0.85, family='gaussian'):
    """实验4：反符号孪生变量 (paper.md Section 5.1.4)
    n=300, p=1000, 10 pairs twin variables with ρ=0.85
    β_{2t-1}=2.0, β_{2t}=-2.5, remaining 980 variables are noise
    """
    X = np.random.randn(n, p)
    beta_true = np.zeros(p)

    # 生成10对反符号孪生变量
    for i in range(10):
        common = np.random.randn(n)
        X[:, 2*i] = common * np.sqrt(rho) + np.random.randn(n) * np.sqrt(1-rho)
        X[:, 2*i+1] = -common * np.sqrt(rho) + np.random.randn(n) * np.sqrt(1-rho)
        beta_true[2*i] = 2.0
        beta_true[2*i+1] = -2.5

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true

def generate_experiment5_data(n=300, p=1000, sigma=1.0, rho=0.8, family='gaussian'):
    """实验5 魔鬼等级1：绝对隐身陷阱 Perfect Masking (paper.md Section 5.1.5)
    n=300, p=1000, 10 pairs twin variables with ρ=0.8
    β_{2t-1}=2.0, β_{2t}=-2.5, remaining 980 variables are independent noise
    核心挑战：通过精确设计X与y的关系使真实信号的边际相关性归零
    """
    X = np.random.randn(n, p)
    beta_true = np.zeros(p)

    # 生成10对孪生变量（与exp4相同结构）
    for i in range(10):
        common = np.random.randn(n)
        X[:, 2*i] = common * np.sqrt(rho) + np.random.randn(n) * np.sqrt(1-rho)
        X[:, 2*i+1] = -common * np.sqrt(rho) + np.random.randn(n) * np.sqrt(1-rho)
        beta_true[2*i] = 2.0
        beta_true[2*i+1] = -2.5

    # 完美掩藏：构造X使每个孪生变量对的边际效应归零
    # 原理：y = β₁X₁ + β₂X₂ + ...，若X₁,X₂共享common factor，
    # 则通过在构造X时让其与y正交，可以使单变量回归系数→0
    # 实际实现：在生成X后，对每个孪生变量对施以与y精确抵消的调整
    y_base = X @ beta_true + np.random.randn(n) * sigma

    # 对每个孪生变量对，旋转使边际相关归零
    for i in range(10):
        # 计算X_{2i}和X_{2i+1}在y_base上的边际系数
        # 通过Householder变换旋转这对变量，使其与y_base的协方差归一化
        # 实现：找与[y_base, X_{2i}, X_{2i+1}]正交的方向
        v1 = X[:, 2*i].copy()
        v2 = X[:, 2*i+1].copy()

        # 使v1与y正交（边际相关归零）
        proj_y = np.dot(v1, y_base) / (np.dot(y_base, y_base) + 1e-10)
        v1_ortho = v1 - proj_y * y_base
        v1_ortho = v1_ortho / (np.linalg.norm(v1_ortho) + 1e-10)

        # 对v2同样处理，使正交于y_base
        proj_y2 = np.dot(v2, y_base) / (np.dot(y_base, y_base) + 1e-10)
        v2_ortho = v2 - proj_y2 * y_base
        v2_ortho = v2_ortho / (np.linalg.norm(v2_ortho) + 1e-10)

        X[:, 2*i] = v1_ortho * np.std(X[:, 2*i]) + np.mean(X[:, 2*i])
        X[:, 2*i+1] = v2_ortho * np.std(X[:, 2*i+1]) + np.mean(X[:, 2*i+1])

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true

def generate_experiment6_data(n=300, p=500, sigma=1.0, rho=0.8, family='gaussian'):
    """实验6 魔鬼等级2：鸠占鹊巢陷阱 The Decoy Trap (paper.md Section 5.1.6)
    n=300, p=500, 每3变量一组共5组
    真信号: X1,X2,X4,X5,X7,X8,X13,X14,X16,X17 (每组前2个)，β=1.0
    噪声诱饵: X3,X6,X9,X15,X18 (每组第3个)，与同组真信号ρ=0.8
    剩余482个独立噪声变量，β=0
    """
    X = np.random.randn(n, p)
    beta_true = np.zeros(p)

    # 5组三变量结构 (0-indexed: 对应paper中的1-indexed)
    # 组0: X0,X1,X2 | 组1: X3,X4,X5 | 组2: X6,X7,X8
    # 组3: X12,X13,X14 | 组4: X15,X16,X17
    # 真信号: 每组前2个变量 (a,b)；噪声诱饵: 第3个变量 (c)
    groups = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (12, 13, 14), (15, 16, 17)]

    for a, b, c in groups:
        # a,b为真信号(独立), c为噪声诱饵(与a,b相关)
        common_ab = np.random.randn(n) * np.sqrt(0.5)
        indep_a = np.random.randn(n) * np.sqrt(0.5)
        indep_b = np.random.randn(n) * np.sqrt(0.5)
        indep_c = np.random.randn(n) * np.sqrt(1 - rho)

        X[:, a] = common_ab + indep_a
        X[:, b] = common_ab + indep_b
        # c与a,b均相关ρ
        common_ac = np.random.randn(n) * np.sqrt(rho)
        X[:, c] = common_ac + indep_c

        beta_true[a] = 1.0
        beta_true[b] = 1.0
        # beta_true[c] = 0 (保持为0)

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true

def generate_experiment7_data(n=300, p=500, sigma=1.0, rho=0.9, family='gaussian'):
    """实验7 魔鬼等级3：自回归符号雪崩 AR(1) Sign Avalanche (paper.md Section 5.1.7)
    n=300, p=500, 前20变量AR(1)链条 Σ_ij=0.9^|i-j|
    β_j = (-1)^(j+1) * 2.0 * 0.9^((j-1)/2), j=1..20
    例如: β1=2.0, β2=-1.8, β3=1.62, β4=-1.458, ...
    剩余480个变量β=0
    """
    # AR(1)协方差矩阵
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = rho ** abs(i - j)
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 真实系数：前20个变量正负交替衰减
    beta_true = np.zeros(p)
    for j in range(1, 21):
        beta_true[j-1] = ((-1) ** (j + 1)) * 2.0 * (0.9 ** ((j - 1) / 2))

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true

# ------------------------------------------------------------------------------
# 指标计算
# ------------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred, y_prob=None, beta_true=None, beta_pred=None, task_type='regression'):
    """计算实验指标"""
    metrics = {}

    if task_type == 'regression':
        metrics['mse'] = mean_squared_error(y_true, y_pred)
    else:  # classification
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        metrics['f1'] = f1_score(y_true, y_pred)

    # 变量选择指标（如果有真实系数）
    if beta_true is not None and beta_pred is not None:
        beta_true_nonzero = np.abs(beta_true) > 1e-8
        beta_pred_nonzero = np.abs(beta_pred) > 1e-8

        tp = np.sum(beta_true_nonzero & beta_pred_nonzero)
        fp = np.sum(~beta_true_nonzero & beta_pred_nonzero)
        fn = np.sum(beta_true_nonzero & ~beta_pred_nonzero)

        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['fdr'] = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['f1_score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        metrics['n_selected'] = np.sum(beta_pred_nonzero)

    return metrics

# ------------------------------------------------------------------------------
# 算法运行函数
# ------------------------------------------------------------------------------
def run_algorithm(alg_name, X_train, y_train, X_test, y_test, family='gaussian', groups=None, beta_true=None, scaler=None, alg_config_override=None):
    """运行单个算法

    Args:
        alg_config_override: 可选，传入配置字典覆盖ALGORITHMS中的默认配置，
                            用于网格搜索时动态传入不同参数组合
    """
    try:
        alg_config = alg_config_override if alg_config_override is not None else ALGORITHMS[alg_name]
        params = alg_config['params'].copy()
        params['family'] = family

        if alg_config['class'] == 'xlasso':
            # XLasso算法（固定lambda或取路径中间值）
            # lmda（单值，网格搜索用）映射为lmdas供fit_uni使用
            fixed_lmda = params.pop('lmda', None)
            if fixed_lmda is not None:
                params['lmdas'] = fixed_lmda
            # k参数映射为gamma（fit_uni用gamma而非k作为权重指数）
            if 'k' in params:
                gamma_val = params.pop('k')
                params['gamma'] = gamma_val
                params['adaptive_weighting'] = True
            result = fit_uni(X_train, y_train, **params)
            # 单lambda时coefs可能为1D或2D，需要统一处理
            if len(result.lmdas) == 1:
                coef = result.coefs.squeeze()
                intercept = result.intercept.squeeze() if isinstance(result.intercept, np.ndarray) else result.intercept
            else:
                best_idx = len(result.lmdas) // 2
                coef = result.coefs[best_idx]
                intercept = result.intercept[best_idx]

            z = X_test @ coef + intercept
            if family == 'gaussian':
                y_pred = z
                y_prob = None
            else:
                y_prob = 1 / (1 + np.exp(-z))
                y_pred = (y_prob >= 0.5).astype(int)

        elif alg_config['class'] == 'xlasso_cv':
            # XLasso算法（使用cv_uni做lambda交叉验证）
            cv_folds = params.pop('cv_folds')
            params.pop('lmda_scale', None)  # cv_uni不支持lmda_scale，直接丢弃
            # k参数映射为gamma（cv_uni用gamma而非k命名）
            gamma_val = params.pop('k', 1.0)
            params['gamma'] = gamma_val
            # gamma只在adaptive_weighting=True时生效，需要显式开启
            params['adaptive_weighting'] = True
            result = cv_uni(X_train, y_train, n_folds=cv_folds, **params)
            coef = result.coefs[result.best_idx].squeeze()
            intercept = result.intercept[result.best_idx].squeeze()

            z = X_test @ coef + intercept
            if family == 'gaussian':
                y_pred = z
                y_prob = None
            else:
                y_prob = 1 / (1 + np.exp(-z))
                y_pred = (y_prob >= 0.5).astype(int)

        elif alg_config['class'] == 'xlasso_predecomp':
            # XLasso with Pre-Decomposition (前置正交分解预处理)
            cv_folds = params.pop('cv_folds')
            corr_threshold = params.pop('corr_threshold', 0.5)
            max_group_size = params.pop('max_group_size', 20)
            min_explained_variance_ratio = params.pop('min_explained_variance_ratio', 0.7)
            params.pop('lmda_scale', None)

            # Create and fit preprocessor on X_train
            preprocessor = GroupPreprocessor(
                corr_threshold=corr_threshold,
                max_group_size=max_group_size,
                min_explained_variance_ratio=min_explained_variance_ratio,
            )
            X_train_trans = preprocessor.fit_transform(X_train)

            # k参数映射为gamma
            gamma_val = params.pop('k', 1.0)
            params['gamma'] = gamma_val
            params['adaptive_weighting'] = True

            # Run cv_uni on transformed data
            result = cv_uni(X_train_trans, y_train, n_folds=cv_folds, **params)
            coef_trans = result.coefs[result.best_idx].squeeze()
            intercept = result.intercept[result.best_idx].squeeze()

            # Inverse transform coefficients to original space
            coef = preprocessor.inverse_transform_coef(coef_trans)

            # Prediction using original X_test (coefficients are now in original space)
            z = X_test @ coef + intercept
            if family == 'gaussian':
                y_pred = z
                y_prob = None
            else:
                y_prob = 1 / (1 + np.exp(-z))
                y_pred = (y_prob >= 0.5).astype(int)

        else:
            # 其他算法
            params_copy = params.copy()
            # sklearn的算法不支持family参数
            if alg_name in ['逻辑回归(L2)', '标准Lasso']:
                params_copy.pop('family', None)

            if family == 'gaussian' and alg_name == '标准Lasso':
                # 回归用LassoCV
                from sklearn.linear_model import LassoCV
                model = LassoCV(
                    cv=EXPERIMENT_CONFIG['cv_folds'],
                    n_jobs=EXPERIMENT_CONFIG['n_jobs'],
                    max_iter=1000,
                    random_state=EXPERIMENT_CONFIG['random_state']
                )
            elif family == 'binomial' and alg_name == '标准Lasso':
                # 分类用LogisticRegressionCV，l1正则
                from sklearn.linear_model import LogisticRegressionCV
                model = LogisticRegressionCV(
                    penalty='l1',
                    cv=EXPERIMENT_CONFIG['cv_folds'],
                    max_iter=1000,
                    solver='liblinear',
                    n_jobs=EXPERIMENT_CONFIG['n_jobs'],
                    random_state=EXPERIMENT_CONFIG['random_state']
                )
            elif alg_name == '逻辑回归(L2)':
                # 逻辑回归
                from sklearn.linear_model import LogisticRegressionCV
                model = LogisticRegressionCV(
                    penalty='l2',
                    cv=EXPERIMENT_CONFIG['cv_folds'],
                    max_iter=1000,
                    solver='liblinear',
                    n_jobs=EXPERIMENT_CONFIG['n_jobs'],
                    random_state=EXPERIMENT_CONFIG['random_state']
                )
            else:
                if 'need_groups' in alg_config and alg_config['need_groups']:
                    params_copy['groups'] = groups
                model = alg_config['class'](**params_copy)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if hasattr(model, 'predict_proba') and family == 'binomial':
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = None

            coef = model.coef_
            if y_prob is not None and family == 'binomial' and alg_name not in ['逻辑回归(L2)', '标准Lasso']:
                # 二分类阈值修正
                y_pred = (y_prob >= 0.5).astype(int)

        # 计算指标
        metrics = calculate_metrics(y_test, y_pred, y_prob, task_type='regression' if family == 'gaussian' else 'classification')
        metrics['n_selected'] = np.sum(np.abs(coef) > 1e-8)

        # 计算变量选择指标和估计误差（如果有真实系数）
        if beta_true is not None:
            beta_true_nonzero = np.abs(beta_true) > 1e-8
            beta_pred_nonzero = np.abs(coef) > 1e-8
            tp = np.sum(beta_true_nonzero & beta_pred_nonzero)
            fp = np.sum(~beta_true_nonzero & beta_pred_nonzero)
            fn = np.sum(beta_true_nonzero & ~beta_pred_nonzero)

            metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['fdr'] = fp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

            # 估计误差：coef来自标准化数据，需要反标准化后与原始beta_true比较
            if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                coef_original = coef / scaler.scale_
                metrics['est_error'] = mean_squared_error(beta_true, coef_original)
            else:
                metrics['est_error'] = mean_squared_error(beta_true, coef)

        metrics['success'] = True

        return metrics

    except Exception as e:
        print(f"❌ 算法 {alg_name} 运行失败: {str(e)}")
        return {'success': False, 'error': str(e)}

# ------------------------------------------------------------------------------
# 结果保存
# ------------------------------------------------------------------------------
def get_experiment_save_path(experiment_name):
    """生成带时间戳的实验保存路径"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        EXPERIMENT_CONFIG['save_dir'],
        f"{experiment_name}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir, timestamp

def save_results(results, experiment_name, config=None):
    """
    保存实验结果，包含完整元数据
    Args:
        results: 实验结果列表
        experiment_name: 实验名称
        config: 实验配置字典，可选
    Returns:
        df_raw, df_summary, save_dir
    """
    # 生成保存路径
    save_dir, timestamp = get_experiment_save_path(experiment_name)
    start_time = time.time()

    # 1. 保存原始结果
    df_raw = pd.DataFrame(results)
    raw_path = os.path.join(save_dir, f'results_raw.csv')
    df_raw.to_csv(raw_path, index=False, encoding='utf-8-sig')

    # 2. 保存汇总结果
    df_summary = None
    if len(results) > 0:
        # 提取指标列
        metrics_cols = [col for col in results[0].keys()
                       if col not in ['算法', '重复次数', 'success', 'error', '噪声σ', '数据集']]
        group_cols = ['算法']
        if '噪声σ' in df_raw.columns:
            group_cols.append('噪声σ')
        if '数据集' in df_raw.columns:
            group_cols.append('数据集')

        # 按分组统计均值和标准差
        summary = []
        for _, group in df_raw[df_raw['success']].groupby(group_cols):
            if len(group) == 0:
                continue
            row = {col: group.iloc[0][col] for col in group_cols}
            row['样本量'] = len(group)
            row['成功率'] = len(group) / len(df_raw[df_raw['算法'] == row['算法']])

            for col in metrics_cols:
                if col in group.columns:
                    values = group[col].dropna()
                    if len(values) > 0:
                        row[f'{col}_mean'] = np.mean(values)
                        row[f'{col}_std'] = np.std(values)
                        row[f'{col}_min'] = np.min(values)
                        row[f'{col}_max'] = np.max(values)

            summary.append(row)

        df_summary = pd.DataFrame(summary)
        summary_path = os.path.join(save_dir, f'results_summary.csv')
        df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')

    # 3. 保存实验元数据
    metadata = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'datetime': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'version': EXPERIMENT_CONFIG['version'],
        'config': EXPERIMENT_CONFIG.copy(),
        'user_config': config if config is not None else {},
        'total_runtime_seconds': time.time() - start_time,
        'total_experiments': len(results),
        'successful_experiments': sum(1 for r in results if r['success']),
        'failed_experiments': sum(1 for r in results if not r['success']),
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform,
            'n_cpus': os.cpu_count()
        }
    }

    metadata_path = os.path.join(save_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    # 4. 生成README说明文件
    readme_content = f"""# 实验结果: {experiment_name}
📅 实验时间: {metadata['datetime']}
🏷️ 版本: {metadata['version']}
⏱️ 总运行时间: {metadata['total_runtime_seconds']:.2f}秒
📊 总实验数: {metadata['total_experiments']} | ✅ 成功: {metadata['successful_experiments']} | ❌ 失败: {metadata['failed_experiments']}

## 实验配置
```json
{json.dumps(metadata['config'], indent=2, ensure_ascii=False)}
```

## 文件说明
- `results_raw.csv`: 所有实验的原始结果，包含每次重复的详细数据
- `results_summary.csv`: 按算法分组的汇总统计，包含均值、标准差、最值
- `metadata.json`: 完整元数据，包含实验配置、系统信息、运行时间等
- `comparison_*.png`: 对比图（如果开启绘图）

## 算法列表
{chr(10).join([f"- {alg}" for alg in df_raw['算法'].unique()]) if len(results) > 0 else "无"}

## 指标说明
### 回归任务
- MSE: 均方误差（越小越好）
- TPR: 真阳性率（越大越好）
- FDR: 假发现率（越小越好）
- F1: F1分数（越大越好）
- n_selected: 选中特征数

### 分类任务
- accuracy: 准确率（越大越好）
- AUC: ROC曲线下面积（越大越好）
- F1: F1分数（越大越好）
- n_selected: 选中特征数
"""
    readme_path = os.path.join(save_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    # 5. 打印结果摘要
    print(f"\n💾 实验结果已保存到: {save_dir}")
    print(f"   📄 原始数据: results_raw.csv ({len(df_raw)}条记录)")
    if df_summary is not None:
        print(f"   📊 汇总数据: results_summary.csv ({len(df_summary)}条分组记录)")
    print(f"   ℹ️  元数据: metadata.json & README.md")
    print(f"   ⏱️  总运行时间: {metadata['total_runtime_seconds']:.2f}秒")

    # 6. 生成最新结果链接
    latest_dir = os.path.join(EXPERIMENT_CONFIG['save_dir'], 'latest')
    if os.path.islink(latest_dir):
        os.unlink(latest_dir)
    elif os.path.exists(latest_dir):
        import shutil
        shutil.rmtree(latest_dir)
    os.symlink(os.path.abspath(save_dir), latest_dir, target_is_directory=True)
    print(f"   🔗 最新结果链接: result/comprehensive_experiments/latest/")

    return df_raw, df_summary, save_dir
