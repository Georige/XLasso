#!/usr/bin/env python
"""
综合基准实验：对比Lasso、UniLasso、other_lasso系列与XLasso
Stage 2: 使用Stage1最优结构参数做lambda CV调优
Benchmark: 使用相同数据结构跑Lasso、UniLasso等对比算法

Usage:
    python run_benchmark_comparison.py --n-repeats 3 --n-jobs 8
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lab'))

import argparse
import csv
import time
import json
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from experiment_utils import (
    EXPERIMENT_CONFIG,
    generate_experiment1_data, generate_experiment2_data, generate_experiment3_data,
    generate_experiment4_data, generate_experiment5_data, generate_experiment6_data,
    generate_experiment7_data,
    run_algorithm
)
from other_lasso import (
    AdaptiveLassoCV, GroupLassoCV, FusedLassoCV,
    AdaptiveSparseGroupLassoCV
)
from unilasso.uni_lasso import fit_uni, cv_uni

# Stage 1 最优配置（从完整stage1_results.json分析得到，基于真实MSE/F1数据）
STAGE1_BEST_CONFIGS = {
    'exp1': {
        'k': 0.3, 'enable_group_decomp': False, 'group_corr_threshold': None, 'enable_group_aware_filter': False,
        'family': 'gaussian', 'best_lambda': None  # Group OFF与ON的MSE相同，OFF更简单
    },
    'exp2': {
        'k': 0.5, 'enable_group_decomp': False, 'group_corr_threshold': None, 'enable_group_aware_filter': False,
        'family': 'gaussian', 'best_lambda': None  # k=0.5的MSE(0.3260)优于k=0.3(0.3283)
    },
    'exp3': {
        'k': 0.3, 'enable_group_decomp': True, 'group_corr_threshold': 0.3, 'enable_group_aware_filter': False,
        'family': 'binomial', 'best_lambda': None  # Group ON的F1=0.9915远优于OFF的0.3014
    },
    'exp4': {
        'k': 0.3, 'enable_group_decomp': True, 'group_corr_threshold': 0.4, 'enable_group_aware_filter': True,
        'family': 'gaussian', 'best_lambda': None  # Group ON的MSE(0.7264)明显优于OFF(0.9075)
    },
    'exp5': {
        'k': 0.5, 'enable_group_decomp': False, 'group_corr_threshold': None, 'enable_group_aware_filter': False,
        'family': 'gaussian', 'best_lambda': None  # Group ON和OFF的MSE相同
    },
    'exp6': {
        'k': 0.5, 'enable_group_decomp': True, 'group_corr_threshold': 0.4, 'enable_group_aware_filter': True,
        'family': 'gaussian', 'best_lambda': None  # Group ON的MSE(0.3266)略优于OFF(0.3338)
    },
    'exp7': {
        'k': 1.0, 'enable_group_decomp': True, 'group_corr_threshold': 0.5, 'enable_group_aware_filter': True,
        'family': 'gaussian', 'best_lambda': None  # Group ON的MSE(0.4720)远优于OFF(2.0171)
    },
}

# 实验配置
EXPERIMENTS = {
    'exp1': ('高维成对相关稀疏回归', generate_experiment1_data),
    'exp2': ('AR(1)相关稀疏回归', generate_experiment2_data),
    'exp3': ('二分类偏移变量选择', generate_experiment3_data),
    'exp4': ('孪生变量反符号选择', generate_experiment4_data),
    'exp5': ('魔鬼等级1：绝对隐身陷阱', generate_experiment5_data),
    'exp6': ('魔鬼等级2：鸠占鹊巢陷阱', generate_experiment6_data),
    'exp7': ('魔鬼等级3：AR(1)符号雪崩', generate_experiment7_data),
}

# Benchmark算法配置
BENCHMARK_ALGORITHMS = {
    # 标准Lasso
    '标准Lasso': {
        'class': 'sklearn_lasso',
        'params': {},
    },
    # 原始UniLasso (k=1e9, 非自适应权重)
    '原始UniLasso': {
        'class': 'unilasso',
        'params': {'k': 1e9},
    },
    # Adaptive Lasso
    'Adaptive Lasso': {
        'class': AdaptiveLassoCV,
        'params': {
            'gammas': [1.0],
            'cv': 3,
            'n_jobs': 4,
            'max_iter': 1000,
        },
    },
    # Group Lasso (需要groups)
    'Group Lasso': {
        'class': GroupLassoCV,
        'params': {
            'cv': 3,
            'n_jobs': 4,
            'max_iter': 1000,
        },
        'need_groups': True,
    },
    # Fused Lasso
    'Fused Lasso': {
        'class': FusedLassoCV,
        'params': {
            'lambda_fused_ratios': [0.5, 1.0],
            'cv': 3,
            'n_jobs': 4,
            'max_iter': 1000,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='综合基准实验')
    parser.add_argument('--experiments', '-e', type=str, default='all',
                        help='要运行的实验，逗号分隔或all')
    parser.add_argument('--n-repeats', '-n', type=int, default=3,
                        help='重复次数')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='并行任务数')
    parser.add_argument('--result-dir', type=str, default=None,
                        help='结果保存目录')
    parser.add_argument('--skip-xlasso', action='store_true',
                        help='跳过XLasso Stage2，仅跑benchmark')
    return parser.parse_args()


def calculate_metrics(y_true, y_pred, beta_true, beta_pred, family='gaussian', y_prob=None):
    """计算指标"""
    metrics = {}
    if family == 'gaussian':
        from sklearn.metrics import mean_squared_error
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
    else:
        from sklearn.metrics import accuracy_score, roc_auc_score
        y_pred_label = (y_pred > 0.5).astype(int)
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred_label))
        if y_prob is not None:
            try:
                metrics['auc'] = float(roc_auc_score(y_true, y_prob))
            except:
                metrics['auc'] = 0.5

    # 变量选择指标
    true_nonzero = np.abs(beta_true) > 1e-8
    pred_nonzero = np.abs(beta_pred) > 1e-8
    tp = np.sum(true_nonzero & pred_nonzero)
    fp = np.sum(~true_nonzero & pred_nonzero)
    fn = np.sum(true_nonzero & ~pred_nonzero)
    metrics['tpr'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics['fdr'] = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics['f1'] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    metrics['n_selected'] = int(np.sum(pred_nonzero))
    return metrics


def run_xlasso_stage2(exp_id, exp_name, data_gen, cfg, n_repeats, result_dir):
    """运行XLasso Stage 2: 使用Stage1最优结构参数，CV选lambda"""
    family = cfg['family']
    alg_name = 'XLasso-Stage2'

    method_dir = os.path.join(result_dir, '01_XLasso_Stage2', f'{exp_id}_{alg_name}')
    os.makedirs(method_dir, exist_ok=True)

    raw_file = os.path.join(method_dir, f'{alg_name}_raw.csv')
    summary_file = os.path.join(method_dir, f'{alg_name}_summary.csv')

    csv_header = ['experiment_id', 'experiment_name', 'family', 'repeat', 'sigma',
                  'mse', 'accuracy', 'auc', 'tpr', 'fdr', 'f1', 'n_selected', 'est_error', 'time_seconds']
    with open(raw_file, 'w', newline='') as f:
        csv.writer(f).writerow(csv_header)

    summary_header = ['experiment_id', 'experiment_name', 'family', 'sigma',
                      'avg_mse', 'avg_accuracy', 'avg_auc', 'avg_tpr', 'avg_fdr', 'avg_f1', 'avg_n_selected', 'avg_est_error', 'avg_time_seconds']
    with open(summary_file, 'w', newline='') as f:
        csv.writer(f).writerow(summary_header)

    print(f"\n  🚀 XLasso Stage2: {exp_name}")
    all_results = []

    for repeat in range(n_repeats):
        np.random.seed(EXPERIMENT_CONFIG['random_state'] + repeat)
        sigmas = [0.5, 1.0, 2.5]

        for sigma in sigmas:
            X, y, beta_true = data_gen(n=300, sigma=sigma, family=family)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=EXPERIMENT_CONFIG['random_state'] + repeat
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            start_time = time.time()

            # 使用cv_uni做lambda CV
            result = cv_uni(
                X_train_scaled, y_train,
                family=family,
                gamma=cfg['k'],
                adaptive_weighting=True,
                enable_group_decomp=cfg['enable_group_decomp'],
                group_corr_threshold=cfg.get('group_corr_threshold'),
                enable_group_aware_filter=cfg.get('enable_group_aware_filter', False),
                lmda_min_ratio=1e-2,
                n_lmdas=100,
                n_folds=3
            )

            coef = result.coefs[result.best_idx].squeeze()
            intercept = result.intercept[result.best_idx].squeeze()

            if family == 'gaussian':
                y_pred = X_test_scaled @ coef + intercept
                y_prob = None
            else:
                z = X_test_scaled @ coef + intercept
                y_prob = 1 / (1 + np.exp(-z))
                y_pred = y_prob

            run_time = time.time() - start_time

            # 还原系数到原始尺度
            if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                coef_original = coef / scaler.scale_
            else:
                coef_original = coef

            metrics = calculate_metrics(y_test, y_pred, beta_true, coef_original, family, y_prob)
            metrics['est_error'] = float(np.mean((beta_true - coef_original) ** 2))
            metrics['time_seconds'] = run_time
            metrics['sigma'] = sigma

            with open(raw_file, 'a', newline='') as f:
                csv.writer(f).writerow([
                    exp_id, exp_name, family, repeat+1, sigma,
                    metrics.get('mse', np.nan), metrics.get('accuracy', np.nan), metrics.get('auc', np.nan),
                    metrics['tpr'], metrics['fdr'], metrics['f1'], metrics['n_selected'],
                    metrics.get('est_error', np.nan), run_time
                ])
            all_results.append(metrics)

    # 计算汇总
    if all_results:
        for sigma in [0.5, 1.0, 2.5]:
            sigma_results = [r for r in all_results if r['sigma'] == sigma]
            if sigma_results:
                avg_mse = np.nanmean([r.get('mse', np.nan) for r in sigma_results])
                avg_acc = np.nanmean([r.get('accuracy', np.nan) for r in sigma_results])
                avg_auc = np.nanmean([r.get('auc', np.nan) for r in sigma_results])
                avg_tpr = np.nanmean([r['tpr'] for r in sigma_results])
                avg_fdr = np.nanmean([r['fdr'] for r in sigma_results])
                avg_f1 = np.nanmean([r['f1'] for r in sigma_results])
                avg_n_sel = np.nanmean([r['n_selected'] for r in sigma_results])
                avg_est_err = np.nanmean([r.get('est_error', np.nan) for r in sigma_results])
                avg_time = np.nanmean([r['time_seconds'] for r in sigma_results])

                with open(summary_file, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        exp_id, exp_name, family, sigma,
                        avg_mse, avg_acc, avg_auc, avg_tpr, avg_fdr, avg_f1, avg_n_sel, avg_est_err, avg_time
                    ])

                print(f"    σ={sigma}: MSE={avg_mse:.4f}, F1={avg_f1:.4f}, TPR={avg_tpr:.4f}, n_sel={avg_n_sel:.1f}")


def run_benchmark(exp_id, exp_name, data_gen, family, n_repeats, result_dir):
    """运行对比算法benchmark"""
    for alg_name, alg_config in BENCHMARK_ALGORITHMS.items():
        method_dir = os.path.join(result_dir, f'02_Benchmark_{alg_name}', f'{exp_id}_{alg_name}')
        os.makedirs(method_dir, exist_ok=True)

        raw_file = os.path.join(method_dir, f'{alg_name}_raw.csv')
        summary_file = os.path.join(method_dir, f'{alg_name}_summary.csv')

        csv_header = ['experiment_id', 'experiment_name', 'family', 'repeat', 'sigma',
                      'mse', 'accuracy', 'auc', 'tpr', 'fdr', 'f1', 'n_selected', 'est_error', 'time_seconds']
        with open(raw_file, 'w', newline='') as f:
            csv.writer(f).writerow(csv_header)

        summary_header = ['experiment_id', 'experiment_name', 'family', 'sigma',
                          'avg_mse', 'avg_accuracy', 'avg_auc', 'avg_tpr', 'avg_fdr', 'avg_f1', 'avg_n_selected', 'avg_est_error', 'avg_time_seconds']
        with open(summary_file, 'w', newline='') as f:
            csv.writer(f).writerow(summary_header)

        print(f"\n  📊 {alg_name}: {exp_name}")
        all_results = []

        for repeat in range(n_repeats):
            np.random.seed(EXPERIMENT_CONFIG['random_state'] + repeat)
            sigmas = [0.5, 1.0, 2.5]

            for sigma in sigmas:
                X, y, beta_true = data_gen(n=300, sigma=sigma, family=family)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=EXPERIMENT_CONFIG['random_state'] + repeat
                )
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # 自动分组（用于Group Lasso等）
                groups = None
                if alg_config.get('need_groups'):
                    from other_lasso import GroupLasso
                    groups = GroupLasso.group_features_by_correlation(X_train_scaled, corr_threshold=0.7)

                start_time = time.time()
                y_pred, y_prob, coef = None, None, None

                try:
                    if alg_config['class'] == 'sklearn_lasso':
                        if family == 'gaussian':
                            model = LassoCV(cv=3, n_jobs=4, max_iter=1000, random_state=42)
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                            coef = model.coef_
                        else:
                            model = LogisticRegressionCV(
                                penalty='l1', cv=3, solver='liblinear',
                                max_iter=1000, n_jobs=4, random_state=42
                            )
                            model.fit(X_train_scaled, y_train)
                            y_prob = model.predict_proba(X_test_scaled)[:, 1]
                            y_pred = y_prob
                            coef = model.coef_[0]

                    elif alg_config['class'] == 'unilasso':
                        result = fit_uni(
                            X_train_scaled, y_train,
                            family=family,
                            k=alg_config['params']['k'],
                            lmda_scale=1.0,
                            lmda_min_ratio=1e-2
                        )
                        mid_idx = len(result.lmdas) // 2
                        coef = result.coefs[mid_idx]
                        intercept = result.intercept[mid_idx]
                        if family == 'gaussian':
                            y_pred = X_test_scaled @ coef + intercept
                        else:
                            z = X_test_scaled @ coef + intercept
                            y_prob = 1 / (1 + np.exp(-z))
                            y_pred = y_prob

                    else:
                        # other_lasso methods
                        params = alg_config['params'].copy()
                        if 'need_groups' in alg_config:
                            params['groups'] = groups
                        params['family'] = family

                        model = alg_config['class'](**params)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        coef = model.coef_
                        if hasattr(model, 'predict_proba') and family == 'binomial':
                            y_prob = model.predict_proba(X_test_scaled)[:, 1]

                    run_time = time.time() - start_time

                    # 还原系数
                    if coef is not None and hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                        coef_original = coef / scaler.scale_
                    else:
                        coef_original = coef

                    if y_pred is not None:
                        metrics = calculate_metrics(y_test, y_pred, beta_true, coef_original, family, y_prob)
                        metrics['time_seconds'] = run_time
                        metrics['sigma'] = sigma
                        if coef_original is not None:
                            metrics['est_error'] = float(np.mean((beta_true - coef_original) ** 2))

                        with open(raw_file, 'a', newline='') as f:
                            csv.writer(f).writerow([
                                exp_id, exp_name, family, repeat+1, sigma,
                                metrics.get('mse', np.nan), metrics.get('accuracy', np.nan), metrics.get('auc', np.nan),
                                metrics['tpr'], metrics['fdr'], metrics['f1'], metrics['n_selected'],
                                metrics.get('est_error', np.nan), run_time
                            ])
                        all_results.append(metrics)

                except Exception as e:
                    print(f"      ❌ {alg_name} repeat={repeat+1} sigma={sigma} failed: {e}")

        # 汇总
        if all_results:
            for sigma in [0.5, 1.0, 2.5]:
                sigma_results = [r for r in all_results if r['sigma'] == sigma]
                if sigma_results:
                    avg_mse = np.nanmean([r.get('mse', np.nan) for r in sigma_results])
                    avg_acc = np.nanmean([r.get('accuracy', np.nan) for r in sigma_results])
                    avg_auc = np.nanmean([r.get('auc', np.nan) for r in sigma_results])
                    avg_tpr = np.nanmean([r['tpr'] for r in sigma_results])
                    avg_fdr = np.nanmean([r['fdr'] for r in sigma_results])
                    avg_f1 = np.nanmean([r['f1'] for r in sigma_results])
                    avg_n_sel = np.nanmean([r['n_selected'] for r in sigma_results])
                    avg_est_err = np.nanmean([r.get('est_error', np.nan) for r in sigma_results])
                    avg_time = np.nanmean([r['time_seconds'] for r in sigma_results])

                    with open(summary_file, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            exp_id, exp_name, family, sigma,
                            avg_mse, avg_acc, avg_auc, avg_tpr, avg_fdr, avg_f1, avg_n_sel, avg_est_err, avg_time
                        ])

                    print(f"    σ={sigma}: MSE={avg_mse:.4f}, F1={avg_f1:.4f}, TPR={avg_tpr:.4f}, n_sel={avg_n_sel:.1f}")


def main():
    args = parse_args()

    # 解析实验列表
    if args.experiments == 'all':
        run_exps = list(EXPERIMENTS.keys())
    else:
        run_exps = [e.strip() for e in args.experiments.split(',')]

    # 结果目录
    if args.result_dir:
        result_dir = args.result_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'result',
            f"benchmark_{timestamp}"
        )
    os.makedirs(result_dir, exist_ok=True)

    # 保存配置
    config_info = {
        'timestamp': timestamp,
        'experiments': run_exps,
        'n_repeats': args.n_repeats,
        'xl_algorithms': ['XLasso-Stage2'],
        'benchmark_algorithms': list(BENCHMARK_ALGORITHMS.keys()),
        'stage1_best_configs': STAGE1_BEST_CONFIGS,
    }
    with open(os.path.join(result_dir, 'config.json'), 'w') as f:
        json.dump(config_info, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"📂 结果目录: {result_dir}")
    print(f"🧪 实验: {run_exps}")
    print(f"🔄 重复次数: {args.n_repeats}")
    print(f"{'='*80}")

    # 1. XLasso Stage2
    if not args.skip_xlasso:
        print(f"\n{'='*80}")
        print("🚀 Stage 2: XLasso使用Stage1最优结构参数做CV lambda调优")
        print(f"{'='*80}")

        for exp_id in run_exps:
            exp_name, data_gen = EXPERIMENTS[exp_id]
            cfg = STAGE1_BEST_CONFIGS[exp_id]
            print(f"\n{'='*60}")
            print(f"📊 {exp_id}: {exp_name}")
            print(f"{'='*60}")

            run_xlasso_stage2(exp_id, exp_name, data_gen, cfg, args.n_repeats, result_dir)

    # 2. Benchmark对比算法
    print(f"\n{'='*80}")
    print("📊 Benchmark: Lasso、UniLasso、其他Lasso变种")
    print(f"{'='*80}")

    for exp_id in run_exps:
        exp_name, data_gen = EXPERIMENTS[exp_id]
        cfg = STAGE1_BEST_CONFIGS[exp_id]
        family = cfg['family']

        print(f"\n{'='*60}")
        print(f"📊 {exp_id}: {exp_name} ({family})")
        print(f"{'='*60}")

        run_benchmark(exp_id, exp_name, data_gen, family, args.n_repeats, result_dir)

    print(f"\n{'='*80}")
    print(f"🎉 所有实验完成!")
    print(f"📂 结果保存于: {result_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
