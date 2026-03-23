#!/usr/bin/env python3
"""
Benchmark 运行入口
支持命令行参数，一键运行所有实验
"""
import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
from tqdm import tqdm

# 项目根目录加入路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmark_util import BenchmarkRunner, get_algorithm_list
from data_generator import (
    generate_experiment1_data,
    generate_experiment2_data,
    generate_experiment3_data,
    generate_experiment4_data,
    generate_experiment5_data,
    generate_experiment6_data,
    generate_experiment7_data,
)

# 7个模拟实验配置
SIMULATED_EXPERIMENTS = [
    {
        'id': 1,
        'name': '高维成对相关稀疏回归',
        'generator': generate_experiment1_data,
        'default_params': {'n': 300, 'p': 500, 'sigma': 1.0},
        'supports_family': ['gaussian', 'binomial']
    },
    {
        'id': 2,
        'name': 'AR(1)相关稀疏回归',
        'generator': generate_experiment2_data,
        'default_params': {'n': 300, 'p': 500, 'sigma': 1.0, 'rho': 0.8},
        'supports_family': ['gaussian', 'binomial']
    },
    {
        'id': 3,
        'name': '二分类偏移变量选择',
        'generator': generate_experiment3_data,
        'default_params': {'n': 300, 'p': 500, 'sigma': 1.0, 'rho': 0.8},
        'supports_family': ['binomial', 'gaussian']
    },
    {
        'id': 4,
        'name': '反符号孪生变量',
        'generator': generate_experiment4_data,
        'default_params': {'n': 300, 'p': 1000, 'sigma': 1.0, 'rho': 0.85},
        'supports_family': ['gaussian', 'binomial']
    },
    {
        'id': 5,
        'name': '绝对隐身陷阱',
        'generator': generate_experiment5_data,
        'default_params': {'n': 300, 'p': 1000, 'sigma': 1.0, 'rho': 0.8},
        'supports_family': ['gaussian', 'binomial']
    },
    {
        'id': 6,
        'name': '鸠占鹊巢陷阱',
        'generator': generate_experiment6_data,
        'default_params': {'n': 300, 'p': 500, 'sigma': 1.0, 'rho': 0.8},
        'supports_family': ['gaussian', 'binomial']
    },
    {
        'id': 7,
        'name': '自回归符号雪崩',
        'generator': generate_experiment7_data,
        'default_params': {'n': 300, 'p': 500, 'sigma': 1.0, 'rho': 0.9},
        'supports_family': ['gaussian', 'binomial']
    },
]


def run_default_benchmark():
    """运行默认的 benchmark"""
    from sklearn.datasets import load_diabetes, make_classification, make_regression

    print("=" * 80)
    print("XLasso Benchmark Runner")
    print("=" * 80)

    # 1. 糖尿病数据集 (回归)
    print("\n" + "=" * 60)
    print("1. Diabetes Dataset (Regression)")
    print("=" * 60)
    X, y = load_diabetes(return_X_y=True)
    print(f"X: {X.shape}, y: {y.shape}")

    algorithms = get_algorithm_list('regression')
    runner = BenchmarkRunner(
        algorithms=algorithms,
        X=X, y=y,
        n_repeats=2, cv=3,
        random_seed_start=2026,
        standardize=True
    )
    summary = runner.run(experiment_name="diabetes_regression")

    print("\nSummary:")
    print(summary.iloc[:, :8].to_string())

    # 2. 模拟高维稀疏数据集 (回归)
    print("\n" + "=" * 60)
    print("2. Simulated High-dimensional Sparse Data (Regression)")
    print("=" * 60)
    X, y, coef_true = make_regression(
        n_samples=500, n_features=200, n_informative=20,
        noise=1.0, random_state=2026, coef=True
    )
    print(f"X: {X.shape}, y: {y.shape}, 真实非零系数: {np.sum(coef_true != 0)}/{len(coef_true)}")

    algorithms = get_algorithm_list('regression')
    runner = BenchmarkRunner(
        algorithms=algorithms,
        X=X, y=y, coef_true=coef_true,
        n_repeats=2, cv=3,
        random_seed_start=2026,
        standardize=True
    )
    summary = runner.run(experiment_name="simulated_sparse_regression")

    print("\nSummary:")
    print(summary.iloc[:, :8].to_string())

    # 3. 分类数据集
    print("\n" + "=" * 60)
    print("3. Binary Classification Dataset")
    print("=" * 60)
    X, y = make_classification(
        n_samples=500, n_features=200, n_informative=20,
        n_redundant=0, n_clusters_per_class=2,
        random_state=2026, flip_y=0.1
    )
    print(f"X: {X.shape}, y: {y.shape}, 类别分布: {np.bincount(y)}")

    algorithms = get_algorithm_list('classification')
    runner = BenchmarkRunner(
        algorithms=algorithms,
        X=X, y=y,
        n_repeats=2, cv=3,
        random_seed_start=2026,
        standardize=True
    )
    summary = runner.run(experiment_name="binary_classification")

    print("\nSummary:")
    print(summary.iloc[:, :8].to_string())

    print("\n" + "=" * 80)
    print("All benchmarks completed!")
    print("=" * 80)


def run_custom_benchmark(args):
    """运行自定义 benchmark"""
    # 加载数据
    if args.dataset == 'diabetes':
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(return_X_y=True)
        coef_true = None
        task_type = 'regression'

    elif args.dataset == 'simulated_regression':
        from sklearn.datasets import make_regression
        X, y, coef_true = make_regression(
            n_samples=args.n_samples, n_features=args.n_features,
            n_informative=args.n_informative, noise=args.noise,
            random_state=2026, coef=True
        )
        task_type = 'regression'

    elif args.dataset == 'simulated_classification':
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=args.n_samples, n_features=args.n_features,
            n_informative=args.n_informative, n_redundant=0,
            random_state=2026
        )
        coef_true = None
        task_type = 'classification'

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # 获取算法列表
    if args.algorithms is None:
        algorithms = get_algorithm_list(task_type)
    else:
        all_algos = get_algorithm_list(task_type)
        algorithms = {name: all_algos[name] for name in args.algorithms if name in all_algos}

    # 运行
    runner = BenchmarkRunner(
        algorithms=algorithms,
        X=X, y=y, coef_true=coef_true,
        n_repeats=args.n_repeats, cv=args.cv,
        random_seed_start=args.seed_start,
        standardize=not args.no_standardize,
        task_type=task_type
    )
    summary = runner.run(experiment_name=args.experiment_name)

    print("\nSummary:")
    print(summary.to_string())


def run_smoke_test():
    """冒烟测试：快速验证所有7个实验和所有算法是否能正常运行"""
    print("=" * 80)
    print("🔥 SMOKE TEST: 验证所有7个模拟实验 + 所有算法兼容性")
    print("=" * 80)

    test_results = []
    total_tests = 0
    passed_tests = 0

    # 测试参数：小样本量，少重复，少cv，快速验证
    test_n = 300  # 实验3需要至少300样本保证类别平衡
    test_p = 50
    test_repeats = 1
    test_cv = 2

    for exp_config in SIMULATED_EXPERIMENTS:
        exp_id = exp_config['id']
        exp_name = exp_config['name']
        generator = exp_config['generator']
        families = exp_config['supports_family']

        print(f"\n📝 实验 {exp_id}: {exp_name}")

        for family in families:
            print(f"  🔍 任务类型: {family}")

            # 生成测试数据（减小规模加速测试）
            params = exp_config['default_params'].copy()
            params['n'] = test_n
            params['p'] = test_p
            params['family'] = family
            if 'rho' in params:
                params['rho'] = 0.5  # 降低相关性加速生成

            try:
                X, y, beta_true = generator(**params)
            except Exception as e:
                print(f"    ❌ 数据生成失败: {str(e)}")
                test_results.append({
                    'exp_id': exp_id,
                    'exp_name': exp_name,
                    'family': family,
                    'algorithm': 'data_generator',
                    'status': 'failed',
                    'error': str(e)
                })
                total_tests += 1
                continue

            # 获取对应任务类型的算法
            algorithms = get_algorithm_list('regression' if family == 'gaussian' else 'classification')
            print(f"    待测试算法数: {len(algorithms)}")

            # 自动分组用于Group类算法
            n_features = X.shape[1]
            # 默认分组：每10个特征一组
            default_groups = [list(range(i, min(i+10, n_features))) for i in range(0, n_features, 10)]

            for alg_name in tqdm(algorithms.keys(), desc="    测试算法"):
                total_tests += 1
                try:
                    # 处理需要分组的算法
                    alg_config = algorithms[alg_name]
                    if isinstance(alg_config, tuple) and len(alg_config) >= 2:
                        alg_class, alg_params, is_clf = alg_config
                        # 给GroupLasso类算法添加groups参数
                        if 'Group' in alg_name and 'groups' in alg_params and alg_params['groups'] is None:
                            alg_params = alg_params.copy()
                            if 'skglm' in alg_name.lower():
                                # skglm GroupLasso 每个特征单独一组：list of lists
                                alg_params['groups'] = [[i] for i in range(n_features)]
                            else:
                                # 其他Group算法用默认分组
                                alg_params['groups'] = default_groups
                        single_algo = {alg_name: (alg_class, alg_params, is_clf)}
                    else:
                        single_algo = {alg_name: alg_config}

                    # 运行小benchmark
                    runner = BenchmarkRunner(
                        algorithms=single_algo,
                        X=X, y=y, coef_true=beta_true,
                        n_repeats=test_repeats, cv=test_cv,
                        random_seed_start=2026,
                        standardize=True,
                        verbose=False
                    )
                    summary = runner.run(save_results=False)

                    # 检查结果是否正常
                    if summary is not None and len(summary) > 0:
                        test_results.append({
                            'exp_id': exp_id,
                            'exp_name': exp_name,
                            'family': family,
                            'algorithm': alg_name,
                            'status': 'passed',
                            'error': None
                        })
                        passed_tests += 1
                    else:
                        test_results.append({
                            'exp_id': exp_id,
                            'exp_name': exp_name,
                            'family': family,
                            'algorithm': alg_name,
                            'status': 'failed',
                            'error': '空结果'
                        })

                except Exception as e:
                    test_results.append({
                        'exp_id': exp_id,
                        'exp_name': exp_name,
                        'family': family,
                        'algorithm': alg_name,
                        'status': 'failed',
                        'error': str(e)
                    })

    # 输出测试报告
    print("\n" + "=" * 80)
    print("📊 SMOKE TEST 报告")
    print("=" * 80)
    print(f"总测试用例: {total_tests}")
    print(f"✅ 成功: {passed_tests}")
    print(f"❌ 失败: {total_tests - passed_tests}")
    print(f"通过率: {passed_tests / total_tests * 100:.1f}%")

    if passed_tests < total_tests:
        print("\n❌ 失败的测试:")
        for res in test_results:
            if res['status'] == 'failed':
                print(f"  实验{res['exp_id']} [{res['family']}] - {res['algorithm']}: {res['error']}")
    else:
        print("\n🎉 所有测试通过!")

    # 保存测试结果
    import pandas as pd
    df = pd.DataFrame(test_results)
    df.to_csv('smoke_test_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 详细测试结果已保存到: smoke_test_results.csv")

    return passed_tests == total_tests


def main():
    parser = argparse.ArgumentParser(description="XLasso Benchmark Runner")
    parser.add_argument('--mode', choices=['default', 'custom', 'smoke'], default='default',
                        help='运行模式: default(默认实验)/custom(自定义实验)/smoke(冒烟测试)')
    parser.add_argument('--dataset', choices=['diabetes', 'simulated_regression', 'simulated_classification'] + [f'exp{i}' for i in range(1, 8)],
                        default='diabetes', help='数据集名称, exp1-exp7对应7个模拟实验')
    parser.add_argument('--experiment-name', type=str, default=None, help='实验名称')
    parser.add_argument('--algorithms', nargs='+', help='要运行的算法名称，默认全部')
    parser.add_argument('--n-repeats', type=int, default=3, help='重复实验次数，默认3')
    parser.add_argument('--cv', type=int, default=5, help='交叉验证折数，默认5')
    parser.add_argument('--seed-start', type=int, default=2026, help='随机种子起始值，默认2026')
    parser.add_argument('--no-standardize', action='store_true', help='不标准化数据')
    parser.add_argument('--family', choices=['gaussian', 'binomial'], default='gaussian', help='任务类型，仅对模拟实验有效')

    # 模拟数据参数
    parser.add_argument('--n-samples', type=int, default=300, help='模拟数据样本量')
    parser.add_argument('--n-features', type=int, default=None, help='模拟数据特征数，默认使用实验默认值')
    parser.add_argument('--sigma', type=float, default=1.0, help='模拟数据噪声水平')
    parser.add_argument('--rho', type=float, default=None, help='相关系数，默认使用实验默认值')

    args = parser.parse_args()

    if args.mode == 'smoke':
        success = run_smoke_test()
        sys.exit(0 if success else 1)
    elif args.mode == 'default':
        run_default_benchmark()
    else:
        # 自定义模式支持模拟实验
        if args.dataset.startswith('exp'):
            # 运行7个模拟实验中的一个
            exp_id = int(args.dataset[3:])
            exp_config = SIMULATED_EXPERIMENTS[exp_id - 1]
            print(f"运行模拟实验 {exp_id}: {exp_config['name']}")

            # 生成数据
            params = exp_config['default_params'].copy()
            params['family'] = args.family
            if args.n_samples is not None:
                params['n'] = args.n_samples
            if args.n_features is not None:
                params['p'] = args.n_features
            if args.sigma is not None:
                params['sigma'] = args.sigma
            if args.rho is not None and 'rho' in params:
                params['rho'] = args.rho

            X, y, beta_true = exp_config['generator'](**params)
            print(f"X shape: {X.shape}, y shape: {y.shape}")
            if beta_true is not None:
                print(f"真实非零系数: {np.sum(beta_true != 0)}/{len(beta_true)}")

            # 获取算法
            algorithms = get_algorithm_list('regression' if args.family == 'gaussian' else 'classification')
            if args.algorithms is not None:
                algorithms = {name: algorithms[name] for name in args.algorithms if name in algorithms}

            # 运行
            runner = BenchmarkRunner(
                algorithms=algorithms,
                X=X, y=y, coef_true=beta_true,
                n_repeats=args.n_repeats,
                cv=args.cv,
                random_seed_start=args.seed_start,
                standardize=not args.no_standardize
            )
            summary = runner.run(experiment_name=args.experiment_name or f"{exp_config['name']}_{args.family}")

            print("\nSummary:")
            print(summary.to_string())
        else:
            run_custom_benchmark(args)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
