#!/usr/bin/env python3
"""
NLasso 超参数调优主脚本
支持两阶段调优流程：
- Stage1: 固定lambda，搜索最优结构化参数
- Stage2: 在最优结构化参数下，CV调优lambda
"""
import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from tuning_util import (
    TuningResultSaver,
    generate_param_combinations,
    run_single_tuning_experiment,
    STAGE1_PARAM_SPACE,
    STAGE1_FIXED_LAMBDA,
    STAGE2_LAMBDA_SPACE,
    EXPERIMENT_NAMES
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="NLasso 超参数调优")
    parser.add_argument(
        "--experiment-id", "-e",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        required=True,
        help="实验ID (1-7)"
    )
    parser.add_argument(
        "--stage", "-s",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="调优阶段: 1=仅Stage1, 2=仅Stage2, 3=两阶段完整调优"
    )
    parser.add_argument(
        "--n-repeats", "-n",
        type=int,
        default=3,
        help="每个参数配置的重复次数"
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=2026,
        help="起始随机种子"
    )
    parser.add_argument(
        "--family", "-f",
        type=str,
        choices=["gaussian", "binomial"],
        default="gaussian",
        help="任务类型: gaussian=回归, binomial=分类"
    )
    parser.add_argument(
        "--result-root", "-r",
        type=str,
        default="tuning/result",
        help="结果保存根目录"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="打印详细信息"
    )
    return parser.parse_args()


def run_stage1(
    experiment_id: int,
    n_repeats: int,
    seed_start: int,
    family: str,
    saver: TuningResultSaver,
    verbose: bool = False
) -> dict:
    """
    运行Stage1: 固定lambda，搜索最优结构化参数

    Returns:
        最优结构化参数字典
    """
    print(f"\n{'='*80}")
    print(f"🚀 执行Stage1: 固定lambda={STAGE1_FIXED_LAMBDA}，搜索最优结构化参数")
    print(f"{'='*80}")

    # 生成所有参数组合
    param_combinations = generate_param_combinations(STAGE1_PARAM_SPACE)
    total_combinations = len(param_combinations)
    print(f"📊 共 {total_combinations} 个参数组合待测试\n")

    best_r2 = -float('inf')
    best_accuracy = -float('inf')
    best_params = None
    best_fdr = float('inf')

    for i, params in enumerate(param_combinations):
        # 添加固定的lambda
        params['lambda_'] = STAGE1_FIXED_LAMBDA

        print(f"[{i+1}/{total_combinations}] 测试参数组合: {params}")

        # 运行实验
        raw_results, summary_stats = run_single_tuning_experiment(
            experiment_id=experiment_id,
            params=params,
            n_repeats=n_repeats,
            seed_start=seed_start,
            family=family,
            verbose=verbose
        )

        # 保存结果
        saver.save_experiment_result(
            experiment_id=experiment_id,
            params=params,
            raw_results=raw_results,
            summary_stats=summary_stats,
            stage=1
        )

        # 更新最优参数
        current_r2 = summary_stats.get('r2_mean', -float('inf'))
        current_accuracy = summary_stats.get('accuracy_mean', -float('inf'))
        current_fdr = summary_stats.get('fdr_mean', float('inf'))

        if family == "gaussian":
            # 回归任务：优先看R²
            if current_r2 > best_r2 or (current_r2 == best_r2 and current_fdr < best_fdr):
                best_r2 = current_r2
                best_fdr = current_fdr
                best_params = params.copy()
                print(f"   🌟 发现新最优参数! R²={best_r2:.4f}, FDR={best_fdr:.4f}")
        else:
            # 分类任务：优先看准确率
            if current_accuracy > best_accuracy or (current_accuracy == best_accuracy and current_fdr < best_fdr):
                best_accuracy = current_accuracy
                best_fdr = current_fdr
                best_params = params.copy()
                print(f"   🌟 发现新最优参数! 准确率={best_accuracy:.4f}, FDR={best_fdr:.4f}")

    print(f"\n✅ Stage1完成!")
    if best_params:
        print(f"🏆 最优结构化参数: {best_params}")
        if family == "gaussian":
            print(f"   最优R²: {best_r2:.4f}, 最优FDR: {best_fdr:.4f}")
        else:
            print(f"   最优准确率: {best_accuracy:.4f}, 最优FDR: {best_fdr:.4f}")

    return best_params


def run_stage2(
    experiment_id: int,
    stage1_best_params: dict,
    n_repeats: int,
    seed_start: int,
    family: str,
    saver: TuningResultSaver,
    verbose: bool = False
) -> dict:
    """
    运行Stage2: 在最优结构化参数下，CV调优lambda

    Returns:
        最优参数字典（包含最优lambda）
    """
    print(f"\n{'='*80}")
    print(f"🚀 执行Stage2: 在最优结构化参数下，调优lambda")
    print(f"{'='*80}")

    # 使用Stage1的最优参数，移除固定的lambda
    base_params = {k: v for k, v in stage1_best_params.items() if k != 'lambda_'}
    print(f"📊 基于Stage1最优结构化参数: {base_params}")
    print(f"📊 Lambda搜索空间: {STAGE2_LAMBDA_SPACE[0]:.6f} 到 {STAGE2_LAMBDA_SPACE[-1]:.6f} (共{len(STAGE2_LAMBDA_SPACE)}个点)\n")

    best_r2 = -float('inf')
    best_accuracy = -float('inf')
    best_params = None
    best_fdr = float('inf')
    best_lambda = None

    for i, lambda_val in enumerate(STAGE2_LAMBDA_SPACE):
        # 合并参数
        params = base_params.copy()
        params['lambda_'] = lambda_val

        print(f"[{i+1}/{len(STAGE2_LAMBDA_SPACE)}] 测试lambda={lambda_val:.6f}")

        # 运行实验
        raw_results, summary_stats = run_single_tuning_experiment(
            experiment_id=experiment_id,
            params=params,
            n_repeats=n_repeats,
            seed_start=seed_start,
            family=family,
            verbose=verbose
        )

        # 保存结果
        saver.save_experiment_result(
            experiment_id=experiment_id,
            params=params,
            raw_results=raw_results,
            summary_stats=summary_stats,
            stage=2
        )

        # 更新最优参数
        current_r2 = summary_stats.get('r2_mean', -float('inf'))
        current_accuracy = summary_stats.get('accuracy_mean', -float('inf'))
        current_fdr = summary_stats.get('fdr_mean', float('inf'))

        if family == "gaussian":
            # 回归任务：优先看R²
            if current_r2 > best_r2 or (current_r2 == best_r2 and current_fdr < best_fdr):
                best_r2 = current_r2
                best_fdr = current_fdr
                best_lambda = lambda_val
                best_params = params.copy()
                print(f"   🌟 发现新最优lambda! lambda={best_lambda:.6f}, R²={best_r2:.4f}, FDR={best_fdr:.4f}")
        else:
            # 分类任务：优先看准确率
            if current_accuracy > best_accuracy or (current_accuracy == best_accuracy and current_fdr < best_fdr):
                best_accuracy = current_accuracy
                best_fdr = current_fdr
                best_lambda = lambda_val
                best_params = params.copy()
                print(f"   🌟 发现新最优lambda! lambda={best_lambda:.6f}, 准确率={best_accuracy:.4f}, FDR={best_fdr:.4f}")

    print(f"\n✅ Stage2完成!")
    if best_params:
        print(f"🏆 最优完整参数: {best_params}")
        if family == "gaussian":
            print(f"   最优R²: {best_r2:.4f}, 最优FDR: {best_fdr:.4f}")
        else:
            print(f"   最优准确率: {best_accuracy:.4f}, 最优FDR: {best_fdr:.4f}")

    return best_params


def main():
    args = parse_args()
    exp_id = args.experiment_id
    exp_name = EXPERIMENT_NAMES[exp_id]

    print("="*80)
    print(f"🚀 开始NLasso超参数调优")
    print(f"📊 实验: {exp_id} - {exp_name}")
    print(f"📋 阶段: {args.stage}")
    print(f"🔄 重复次数: {args.n_repeats}")
    print(f"🎲 起始种子: {args.seed_start}")
    print(f"📈 任务类型: {args.family}")
    print("="*80)

    # 初始化结果保存器
    saver = TuningResultSaver(result_root=args.result_root)
    print(f"\n📂 结果保存目录: {saver.get_timestamp_dir()}")

    # 执行调优
    stage1_best_params = None

    if args.stage in [1, 3]:
        # 执行Stage1
        stage1_best_params = run_stage1(
            experiment_id=exp_id,
            n_repeats=args.n_repeats,
            seed_start=args.seed_start,
            family=args.family,
            saver=saver,
            verbose=args.verbose
        )

    if args.stage in [2, 3]:
        # 执行Stage2
        if args.stage == 2:
            # 如果仅执行Stage2，使用默认结构化参数
            print("\n⚠️  仅执行Stage2，使用默认结构化参数")
            stage1_best_params = {
                'lambda_ridge': 10.0,
                'gamma': 0.3,
                's': 1.0,
                'group_threshold': 0.7
            }
        elif stage1_best_params is None:
            raise ValueError("Stage1未执行，无法获取最优结构化参数")

        stage2_best_params = run_stage2(
            experiment_id=exp_id,
            stage1_best_params=stage1_best_params,
            n_repeats=args.n_repeats,
            seed_start=args.seed_start,
            family=args.family,
            saver=saver,
            verbose=args.verbose
        )

    # 保存总体汇总
    print(f"\n{'='*80}")
    print("📊 保存总体汇总...")
    overall_path, ranking_path = saver.save_overall_summary()
    print(f"   总体汇总: {overall_path}")
    print(f"   最优排名: {ranking_path}")

    print(f"\n{'='*80}")
    print("🎉 NLasso超参数调优完成!")
    print(f"📂 最终结果目录: {saver.get_timestamp_dir()}")
    print("="*80)


if __name__ == "__main__":
    main()
