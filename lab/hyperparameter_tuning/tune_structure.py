#!/usr/bin/env python
"""
两阶段结构参数调优脚本

Stage 1: 固定lambda=[0.5, 1.0, 1.5, 2.0]，网格搜索最优结构参数
Stage 2: 固定Stage1最优结构参数，CV搜索最优lambda

Usage:
    # Stage 1: 搜索所有实验的最优结构参数
    python tune_structure.py --stage 1 --n-repeats 3

    # Stage 1: 仅搜索exp1
    python tune_structure.py --stage 1 --experiments exp1 --n-repeats 3

    # Stage 2: 使用exp1的最优结构参数做lambda调优
    python tune_structure.py --stage 2 --experiments exp1 --n-repeats 3
"""
import os
import sys
import json
import argparse
import subprocess
import time
import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

# 确保上层目录在path中（这样才能找到unilasso包）
# __file__ = .../lab/hyperparameter_tuning/tune_structure.py
# dirname 1x = .../lab/hyperparameter_tuning
# dirname 2x = .../lab
# dirname 3x = .../ (XLasso根目录，包含lab和unilasso两个包)
xlasso_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, xlasso_root)
sys.path.insert(0, os.path.join(xlasso_root, 'lab'))

from experiment_utils import EXPERIMENT_CONFIG

# ------------------------------------------------------------------------------
# 结构参数搜索空间
# ------------------------------------------------------------------------------
K_VALUES = [0.3, 0.5, 1.0, 2.0, 3.0]
GROUP_CORR_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.9]  # 扩展低阈值测试Group效果
FIXED_LAMBDA_STAGE1 = [0.5, 1.0, 1.5, 2.0]  # Stage1固定lambda列表，测试何时开启Group

# 实验列表
ALL_EXPERIMENTS = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6', 'exp7']

# 二分类任务实验（使用binomial family）
BINOMIAL_EXPERIMENTS = ['exp3']

def get_experiment_family(exp_id, default_family):
    """获取实验对应的family类型，exp3为二分类任务"""
    if exp_id in BINOMIAL_EXPERIMENTS:
        return 'binomial'
    return default_family

# ------------------------------------------------------------------------------
# 命令行参数
# ------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='XLasso结构参数两阶段调优')
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True,
                        help='调优阶段: 1=结构参数搜索, 2=lambda调优')
    parser.add_argument('--experiments', type=str, default='all',
                        help='运行的实验，逗号分隔或all，默认all')
    parser.add_argument('--n-repeats', type=int, default=3,
                        help='每个配置的重复次数，默认3')
    parser.add_argument('--n-jobs', type=int, default=4,
                        help='并行任务数，默认4')
    parser.add_argument('--family', type=str, default='gaussian',
                        choices=['gaussian', 'binomial'],
                        help='任务类型，默认gaussian')
    parser.add_argument('--result-dir', type=str, default=None,
                        help='结果保存目录，默认hyperparameter_tuning/result/')
    return parser.parse_args()

# ------------------------------------------------------------------------------
# 运行单次实验配置
# ------------------------------------------------------------------------------
def run_single_config(exp_id, config_dict, stage, n_repeats, family, result_base_dir, fixed_lambda=None):
    """通过subprocess调用run_simulation_experiments.py运行单次配置"""
    # 根据配置选择算法名称
    if config_dict.get('enable_group_decomp'):
        if config_dict.get('enable_group_aware_filter'):
            alg_name = 'XLasso-Full'
        else:
            alg_name = 'XLasso-GroupDecomp'
    else:
        alg_name = 'XLasso-Soft'

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'run_simulation_experiments.py'),
    ]
    cmd += ['-e', exp_id]
    cmd += ['-n', str(n_repeats)]
    cmd += ['-f', family]
    cmd += ['-a', alg_name]

    if stage == 1:
        # 使用传入的 fixed_lambda 或默认的第一个值
        lambda_to_use = fixed_lambda if fixed_lambda is not None else FIXED_LAMBDA_STAGE1[0]
        cmd += ['--fixed-lambda', str(lambda_to_use)]

    # 注入结构参数
    if config_dict.get('k') is not None:
        cmd += ['--k', str(config_dict['k'])]
    if config_dict.get('enable_group_decomp') is False:
        cmd += ['--no-group-decomp']
    elif config_dict.get('enable_group_decomp') is True:
        cmd += ['--enable-group-decomp']
    if config_dict.get('group_corr_threshold') is not None:
        cmd += ['--group-corr-threshold', str(config_dict['group_corr_threshold'])]
    if config_dict.get('enable_group_aware_filter') is False:
        cmd += ['--no-group-aware-filter']

    # 生成唯一配置ID（包含lambda信息）
    config_id = _make_config_id(config_dict)
    lambda_str = f"λ{fixed_lambda}" if fixed_lambda is not None else ""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{exp_id}_{lambda_str}_{config_id}_{timestamp}"
    config_result_dir = os.path.join(result_base_dir, run_name)
    os.makedirs(config_result_dir, exist_ok=True)

    # 传递save-root，让run_simulation_experiments.py把结果写到config_result_dir
    cmd += ['--save-root', config_result_dir]

    env = os.environ.copy()
    # 需要包含上层目录（XLasso根目录），这样才能找到unilasso包
    xlasso_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env['PYTHONPATH'] = xlasso_root + os.pathsep + os.path.join(xlasso_root, 'lab')

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时
            env=env
        )
        if result.returncode != 0:
            return {
                'exp_id': exp_id,
                'config': config_dict,
                'success': False,
                'error': result.stderr[-500:] if result.stderr else 'unknown'
            }
        return {
            'exp_id': exp_id,
            'config': config_dict,
            'success': True,
            'config_id': config_id,
            'run_name': run_name,
            'stdout': result.stdout[-1000:] if result.stdout else ''
        }
    except subprocess.TimeoutExpired:
        return {
            'exp_id': exp_id,
            'config': config_dict,
            'success': False,
            'error': 'timeout > 600s'
        }
    except Exception as e:
        return {
            'exp_id': exp_id,
            'config': config_dict,
            'success': False,
            'error': str(e)
        }

def _make_config_id(config):
    """生成配置的唯一标识字符串"""
    parts = []
    if config.get('k') is not None:
        parts.append(f"k{config['k']}")
    if config.get('enable_group_decomp'):
        parts.append("gdc")
        parts.append(f"th{config.get('group_corr_threshold', 0.7)}")
        if config.get('enable_group_aware_filter'):
            parts.append("gaf")
    else:
        parts.append("nogdc")
    return '_'.join(parts) if parts else 'base'

def _parse_summary_csv(summary_path):
    """解析summary CSV，返回平均指标"""
    import pandas as pd
    if not os.path.exists(summary_path):
        return None
    try:
        df = pd.read_csv(summary_path)
        if len(df) == 0:
            return None
        # 取第一行（只有一个family='gaussian'配置时）
        row = df.iloc[0]
        return {
            'avg_mse': row.get('avg_mse', np.nan),
            'avg_tpr': row.get('avg_tpr', np.nan),
            'avg_fdr': row.get('avg_fdr', np.nan),
            'avg_f1': row.get('avg_f1', np.nan),
            'avg_est_error': row.get('avg_est_error', np.nan),
            'avg_n_selected': row.get('avg_n_selected', np.nan),
        }
    except Exception:
        return None

# ------------------------------------------------------------------------------
# Stage 1: 结构参数网格搜索
# ------------------------------------------------------------------------------
def stage1_grid_search(experiments, n_repeats, n_jobs, family, result_base_dir):
    """Stage 1: 固定lambda列表，搜索最优结构参数"""
    print(f"\n{'='*80}")
    print(f"Stage 1: 结构参数网格搜索 (lambda={FIXED_LAMBDA_STAGE1} 固定)")
    print(f"{'='*80}")

    # 生成全量配置组合
    # GroupDecomp=True 时 thresh ∈ {0.5, 0.7, 0.9}, filter ∈ {True, False}
    # GroupDecomp=False 时忽略 thresh 和 filter
    configs = []

    # GroupDecomp=False 组合
    for k in K_VALUES:
        configs.append({
            'k': k,
            'enable_group_decomp': False,
            'group_corr_threshold': None,
            'enable_group_aware_filter': False
        })

    # GroupDecomp=True 组合
    for k, thresh, filt in product(K_VALUES, GROUP_CORR_THRESHOLDS, [True, False]):
        configs.append({
            'k': k,
            'enable_group_decomp': True,
            'group_corr_threshold': thresh,
            'enable_group_aware_filter': filt
        })

    # 计算总运行次数：实验数 × lambda数 × 配置数
    lambda_list = FIXED_LAMBDA_STAGE1 if isinstance(FIXED_LAMBDA_STAGE1, list) else [FIXED_LAMBDA_STAGE1]
    total_runs = len(experiments) * len(lambda_list) * len(configs)
    print(f"实验数: {len(experiments)}, Lambda数: {len(lambda_list)}, 每实验配置数: {len(configs)}, 总运行次数: {total_runs}")
    print(f"并行任务数: {n_jobs}")
    print(f"Lambda值: {lambda_list}")
    print(f"每实验参数组合: k∈{K_VALUES}, group_decomp∈{{T,F}}, thresh∈{GROUP_CORR_THRESHOLDS}, filter∈{{T,F}}")
    print()

    # 提交所有任务
    tasks = []
    for exp_id in experiments:
        for fixed_lambda in lambda_list:
            for config in configs:
                tasks.append((exp_id, config, fixed_lambda))

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(run_single_config, exp_id, config, 1, n_repeats, get_experiment_family(exp_id, family), result_base_dir, fixed_lambda): (exp_id, config, fixed_lambda)
            for exp_id, config, fixed_lambda in tasks
        }

        for future in as_completed(futures):
            exp_id, config, fixed_lambda = futures[future]
            completed += 1
            res = future.result()

            if res['success']:
                config_id = res['config_id']
                run_name = res['run_name']
                # 找summary文件
                # 结果在 result_base_dir/run_name/算法名/raw.csv
                summary_rows = []
                for root, _, files in os.walk(os.path.join(result_base_dir, run_name)):
                    for f in files:
                        if f.endswith('_summary.csv'):
                            row = _parse_summary_csv(os.path.join(root, f))
                            if row:
                                summary_rows.append(row)

                if summary_rows:
                    avg_metrics = {k: sum(r[k] for r in summary_rows) / len(summary_rows)
                                   for k in summary_rows[0].keys()}
                else:
                    avg_metrics = None

                results.append({
                    'exp_id': exp_id,
                    'config': config,
                    'fixed_lambda': fixed_lambda,
                    'config_id': config_id,
                    'run_name': run_name,
                    'metrics': avg_metrics
                })
                status = '✅'
            else:
                results.append({
                    'exp_id': exp_id,
                    'config': config,
                    'fixed_lambda': fixed_lambda,
                    'error': res.get('error', 'unknown')
                })
                status = '❌'

            print(f"[{completed}/{total_runs}] {status} {exp_id} λ={fixed_lambda} {_make_config_id(config)} | "
                  f"MSE={avg_metrics.get('avg_mse', 'N/A') if avg_metrics else 'N/A'} | "
                  f"F1={avg_metrics.get('avg_f1', 'N/A') if avg_metrics else 'N/A'}")

    # 汇总每个实验的最优配置
    print(f"\n{'='*80}")
    print("Stage 1 结果汇总")
    print(f"{'='*80}")

    best_configs = {}
    lambda_list = FIXED_LAMBDA_STAGE1 if isinstance(FIXED_LAMBDA_STAGE1, list) else [FIXED_LAMBDA_STAGE1]

    for exp_id in experiments:
        exp_results = [r for r in results if r.get('exp_id') == exp_id and r.get('metrics') is not None]
        if not exp_results:
            print(f"\n{exp_id}: 所有配置均失败，跳过")
            best_configs[exp_id] = None
            continue

        # 按lambda分组显示结果
        for fixed_lambda in lambda_list:
            lambda_results = [r for r in exp_results if r.get('fixed_lambda') == fixed_lambda]
            if not lambda_results:
                continue

            # 按MSE排序（越小越好）
            lambda_results.sort(key=lambda x: x['metrics'].get('avg_mse', float('inf')))
            best = lambda_results[0]

            print(f"\n{exp_id} @ λ={fixed_lambda} 最优配置:")
            print(f"  结构参数: {best['config']}")
            gdc_status = "✅ Group ON" if best['config'].get('enable_group_decomp') else "❌ Group OFF"
            print(f"  {gdc_status}")
            print(f"  MSE={best['metrics']['avg_mse']:.4f}, "
                  f"TPR={best['metrics']['avg_tpr']:.4f}, "
                  f"FDR={best['metrics']['avg_fdr']:.4f}, "
                  f"F1={best['metrics']['avg_f1']:.4f}")

            # 打印Top3（Group ON vs OFF对比）
            gdc_on = [r for r in lambda_results if r['config'].get('enable_group_decomp')]
            gdc_off = [r for r in lambda_results if not r['config'].get('enable_group_decomp')]
            if gdc_on and gdc_off:
                best_on = min(gdc_on, key=lambda x: x['metrics'].get('avg_mse', float('inf')))
                best_off = min(gdc_off, key=lambda x: x['metrics'].get('avg_mse', float('inf')))
                print(f"  Group ON 最佳:  {_make_config_id(best_on['config'])}  MSE={best_on['metrics']['avg_mse']:.4f}")
                print(f"  Group OFF 最佳: {_make_config_id(best_off['config'])}  MSE={best_off['metrics']['avg_mse']:.4f}")

        # 保存该实验的最优配置（使用第一个lambda的结果）
        first_lambda = lambda_list[0]
        first_lambda_results = [r for r in exp_results if r.get('fixed_lambda') == first_lambda]
        if first_lambda_results:
            first_lambda_results.sort(key=lambda x: x['metrics'].get('avg_mse', float('inf')))
            best_configs[exp_id] = first_lambda_results[0]['config']
        else:
            best_configs[exp_id] = None

    # 保存结果
    stage1_result = {
        'stage': 1,
        'timestamp': datetime.datetime.now().isoformat(),
        'fixed_lambda': FIXED_LAMBDA_STAGE1,
        'experiments': experiments,
        'n_repeats': n_repeats,
        'all_results': [
            {**r, 'config_id': r.get('config_id', ''), 'run_name': r.get('run_name', ''),
             'config_str': _make_config_id(r['config'])}
            for r in results
        ],
        'best_configs': {
            exp_id: _make_config_id(cfg) if cfg else None
            for exp_id, cfg in best_configs.items()
        },
        'best_configs_full': best_configs
    }

    result_file = os.path.join(result_base_dir, 'stage1_results.json')
    with open(result_file, 'w') as f:
        json.dump(stage1_result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 Stage1结果已保存: {result_file}")

    return best_configs, stage1_result

# ------------------------------------------------------------------------------
# Stage 2: Lambda调优（使用Stage1最优结构参数）
# ------------------------------------------------------------------------------
def stage2_lambda_tuning(best_configs, experiments, n_repeats, n_jobs, family, result_base_dir):
    """Stage 2: 使用最优结构参数做CV lambda调优"""
    print(f"\n{'='*80}")
    print(f"Stage 2: Lambda CV调优 (使用Stage1最优结构参数)")
    print(f"{'='*80}")

    # 使用原始run_simulation_experiments.py的CV模式
    # 不传--fixed-lambda，让cv_uni自动选lambda
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stage2_dir = os.path.join(result_base_dir, f'stage2_{timestamp}')
    os.makedirs(stage2_dir, exist_ok=True)

    tasks = []
    for exp_id in experiments:
        cfg = best_configs.get(exp_id)
        if cfg is None:
            print(f"⚠️  {exp_id} 无最优配置，跳过")
            continue
        tasks.append((exp_id, cfg))

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(run_single_config, exp_id, cfg, 2, n_repeats, family, stage2_dir): (exp_id, cfg)
            for exp_id, cfg in tasks
        }

        for future in as_completed(futures):
            exp_id, cfg = futures[future]
            completed += 1
            res = future.result()

            if res['success']:
                run_name = res['run_name']
                summary_rows = []
                for root, _, files in os.walk(os.path.join(stage2_dir, run_name)):
                    for f in files:
                        if f.endswith('_summary.csv'):
                            row = _parse_summary_csv(os.path.join(root, f))
                            if row:
                                summary_rows.append(row)

                if summary_rows:
                    avg_metrics = {k: sum(r[k] for r in summary_rows) / len(summary_rows)
                                   for k in summary_rows[0].keys()}
                else:
                    avg_metrics = None

                results.append({
                    'exp_id': exp_id,
                    'config': cfg,
                    'metrics': avg_metrics
                })
                status = '✅'
            else:
                results.append({'exp_id': exp_id, 'config': cfg, 'error': res.get('error')})
                status = '❌'

            print(f"[{completed}/{len(tasks)}] {status} {exp_id} "
                  f"{_make_config_id(cfg) if cfg else 'N/A'}")

    # 保存结果
    stage2_result = {
        'stage': 2,
        'timestamp': datetime.datetime.now().isoformat(),
        'experiments': experiments,
        'n_repeats': n_repeats,
        'stage1_best_configs': {
            exp_id: _make_config_id(cfg) if cfg else None
            for exp_id, cfg in best_configs.items()
        },
        'results': [
            {**r, 'config_str': _make_config_id(r['config']) if r.get('config') else None}
            for r in results
        ]
    }

    result_file = os.path.join(stage2_dir, 'stage2_results.json')
    with open(result_file, 'w') as f:
        json.dump(stage2_result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*80}")
    print("Stage 2 结果汇总")
    print(f"{'='*80}")
    for r in results:
        if r.get('metrics'):
            print(f"{r['exp_id']}: 结构={_make_config_id(r['config'])} | "
                  f"MSE={r['metrics']['avg_mse']:.4f}, F1={r['metrics']['avg_f1']:.4f}, "
                  f"EstError={r['metrics']['avg_est_error']:.4f}, n_sel={r['metrics']['avg_n_selected']:.1f}")

    print(f"\n💾 Stage2结果已保存: {result_file}")

    return results

# ------------------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------------------
def main():
    args = parse_args()

    # 解析实验列表
    if args.experiments == 'all':
        experiments = ALL_EXPERIMENTS
    else:
        experiments = [e.strip() for e in args.experiments.split(',')]

    # 结果目录
    if args.result_dir:
        result_base_dir = args.result_dir
    else:
        result_base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'result',
            f"tune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    os.makedirs(result_base_dir, exist_ok=True)
    print(f"📂 结果目录: {result_base_dir}")

    if args.stage == 1:
        best_configs, _ = stage1_grid_search(
            experiments, args.n_repeats, args.n_jobs, args.family, result_base_dir
        )
        # Stage1完成后打印下一步指引
        print(f"\n{'='*80}")
        print("Stage 1 完成！下一步：")
        print(f"  python tune_structure.py --stage 2 --n-repeats 3")
        print(f"{'='*80}")
    else:
        # Stage 2: 需要先加载Stage1的最优配置
        # 查找最新的stage1_results.json
        result_parent = os.path.dirname(result_base_dir)
        stage1_files = []
        for root, dirs, files in os.walk(result_parent):
            for f in files:
                if f == 'stage1_results.json':
                    stage1_files.append(os.path.join(root, f))

        if not stage1_files:
            print("❌ 未找到stage1_results.json，请先运行 --stage 1")
            sys.exit(1)

        latest_stage1 = sorted(stage1_files)[-1]
        print(f"📂 加载Stage1结果: {latest_stage1}")
        with open(latest_stage1) as f:
            stage1_data = json.load(f)

        # 构建best_configs字典
        best_configs = {
            exp_id: stage1_data['best_configs_full'].get(exp_id)
            for exp_id in experiments
        }
        # 反序列化（JSON只存了字符串形式）
        # 需要从all_results重建完整config
        all_results = stage1_data.get('all_results', [])
        config_map = {}
        for r in all_results:
            cid = r.get('config_id', '')
            if cid and cid not in config_map:
                config_map[cid] = r['config']

        best_configs_full = {
            exp_id: config_map.get(stage1_data['best_configs'].get(exp_id, ''))
            if stage1_data['best_configs'].get(exp_id)
            else None
            for exp_id in experiments
        }

        stage2_results = stage2_lambda_tuning(
            best_configs_full, experiments, args.n_repeats, args.n_jobs,
            args.family, result_base_dir
        )

if __name__ == '__main__':
    main()
