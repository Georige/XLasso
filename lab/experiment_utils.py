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
from unilasso.uni_lasso import fit_uni
from other_lasso import (
    AdaptiveLassoCV, GroupLassoCV, FusedLassoCV,
    AdaptiveSparseGroupLassoCV
)
from sklearn.linear_model import LassoCV, LogisticRegressionCV

# ------------------------------------------------------------------------------
# 实验配置
# ------------------------------------------------------------------------------
EXPERIMENT_CONFIG = {
    'random_state': 42,
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

    # 优先级3: XLasso全系
    'XLasso-Soft': {
        'class': 'xlasso',
        'params': {
            'enable_group_decomp': False,
            'k': 1.0,
            'lmda_min_ratio': 1e-2,
            'lmda_scale': 1.0,
            'n_lmdas': 100,
        },
        'task_type': ['classification', 'regression']
    },
    'XLasso-GroupDecomp': {
        'class': 'xlasso',
        'params': {
            'enable_group_decomp': True,
            'group_corr_threshold': 0.7,
            'enable_group_aware_filter': False,
            'k': 1.0,
            'lmda_min_ratio': 1e-2,
            'lmda_scale': 1.0,
            'n_lmdas': 100,
        },
        'task_type': ['classification', 'regression']
    },
    'XLasso-Full': {
        'class': 'xlasso',
        'params': {
            'enable_group_decomp': True,
            'group_corr_threshold': 0.7,
            'enable_group_aware_filter': True,
            'k': 1.0,
            'lmda_min_ratio': 1e-2,
            'lmda_scale': 1.0,
            'n_lmdas': 100,
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
def generate_experiment1_data(n=300, p=1000, sigma=1.0, family='gaussian'):
    """实验1：高维成对相关稀疏回归"""
    # 协方差矩阵：成对相关0.5
    cov = np.ones((p, p)) * 0.5
    np.fill_diagonal(cov, 1.0)
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 真实系数：前100个变量非零
    beta_true = np.zeros(p)
    beta_true[:100] = np.random.randn(100)

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:  # binomial
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true

def generate_experiment2_data(n=300, p=1000, sigma=1.0, rho=0.8, family='gaussian'):
    """实验2：AR(1)相关稀疏回归"""
    # AR(1)协方差矩阵
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = rho ** abs(i - j)
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 真实系数：奇数前50个变量非零
    beta_true = np.zeros(p)
    beta_true[1:100:2] = np.random.randn(50) * 2

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true

def generate_experiment3_data(n=200, p=500, sigma=0.5, rho=0.8, family='binomial'):
    """实验3：二分类偏移变量选择场景"""
    # AR(1)相关性结构
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = rho ** abs(i - j)
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 随机生成y标签
    y = np.random.randint(0, 2, size=n)

    # 对y=1的样本，前20个变量增加偏移量
    y1_idx = y == 1
    X[y1_idx, :20] += 0.5

    # 真实系数：前20个变量为真实相关变量，系数通过偏移隐含
    beta_true = np.zeros(p)
    beta_true[:20] = 1.0  # 标记前20个为真实变量

    if family != 'binomial':  # 回归任务
        y = X @ beta_true + np.random.randn(n) * sigma

    return X, y, beta_true


def generate_experiment4_data(n=300, p=1000, sigma=1.0, rho=0.85, family='gaussian'):
    """实验4：反符号孪生变量（降维打击场景）"""
    X = np.random.randn(n, p)
    beta_true = np.zeros(p)

    # 生成10对反符号孪生变量
    for i in range(10):
        common = np.random.randn(n)
        X[:, 2*i] = common * np.sqrt(rho) + np.random.randn(n) * np.sqrt(1-rho)
        X[:, 2*i+1] = -common * np.sqrt(rho) + np.random.randn(n) * np.sqrt(1-rho)
        beta_true[2*i] = 2.0
        beta_true[2*i+1] = -2.0

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
def run_algorithm(alg_name, X_train, y_train, X_test, y_test, family='gaussian', groups=None, beta_true=None):
    """运行单个算法"""
    try:
        alg_config = ALGORITHMS[alg_name]
        params = alg_config['params'].copy()
        params['family'] = family

        if alg_config['class'] == 'xlasso':
            # XLasso算法
            result = fit_uni(X_train, y_train, **params)
            # 选择最优lambda：取中间位置（实际应该用CV，这里简化处理）
            best_idx = len(result.lmdas) // 2
            coef = result.coefs[best_idx]
            intercept = result.intercept[best_idx]

            # 预测
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

        # 计算变量选择指标（如果有真实系数）
        if beta_true is not None:
            beta_true_nonzero = np.abs(beta_true) > 1e-8
            beta_pred_nonzero = np.abs(coef) > 1e-8
            tp = np.sum(beta_true_nonzero & beta_pred_nonzero)
            fp = np.sum(~beta_true_nonzero & beta_pred_nonzero)
            fn = np.sum(beta_true_nonzero & ~beta_pred_nonzero)

            metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['fdr'] = fp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

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
    if os.path.exists(latest_dir):
        import shutil
        shutil.rmtree(latest_dir)
    os.symlink(os.path.abspath(save_dir), latest_dir, target_is_directory=True)
    print(f"   🔗 最新结果链接: result/comprehensive_experiments/latest/")

    return df_raw, df_summary, save_dir
