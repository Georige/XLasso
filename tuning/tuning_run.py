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
from typing import Dict, Any

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from tuning.tuning_util import (
    TuningResultSaver,
    generate_param_combinations,
    run_single_tuning_experiment,
    STAGE1_PARAM_SPACE,
    STAGE1_FIXED_LAMBDA,
    STAGE2_LAMBDA_SPACE,
    EXPERIMENT_NAMES
)


def print_metrics_summary(summary_stats: Dict[str, Any], family: str, prefix: str = "   "):
    """
    打印全面的指标汇总信息

    Args:
        summary_stats: 汇总统计字典
        family: 任务类型 ("gaussian" 或 "binomial")
        prefix: 打印前缀
    """
    if family == "gaussian":
        # 回归任务指标
        mse = summary_stats.get('mse_mean', float('nan'))
        mae = summary_stats.get('mae_mean', float('nan'))
        r2 = summary_stats.get('r2_mean', float('nan'))
        fdr = summary_stats.get('fdr_mean', float('nan'))
        tpr = summary_stats.get('tpr_mean', float('nan'))
        sparsity = summary_stats.get('sparsity_mean', float('nan'))
        n_nonzero = summary_stats.get('n_nonzero_mean', float('nan'))
        coef_sse = summary_stats.get('coef_sse_mean', float('nan'))
        train_time = summary_stats.get('train_time_mean', float('nan'))

        print(f"{prefix}📊 指标汇总:")
        print(f"{prefix}   ├─ 预测精度: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
        print(f"{prefix}   ├─ 变量选择: FDR={fdr:.6f}, TPR={tpr:.6f}")
        print(f"{prefix}   ├─ 稀疏性: Sparsity={sparsity:.4f}, n_nonzero={n_nonzero:.1f}")
        print(f"{prefix}   ├─ 系数估计: Coef_SSE={coef_sse:.6f}")
        print(f"{prefix}   └─ 运行时间: Train={train_time:.4f}s")
    else:
        # 分类任务指标
        accuracy = summary_stats.get('accuracy_mean', float('nan'))
        f1 = summary_stats.get('f1_mean', float('nan'))
        auc = summary_stats.get('auc_mean', float('nan'))
        fdr = summary_stats.get('fdr_mean', float('nan'))
        tpr = summary_stats.get('tpr_mean', float('nan'))
        sparsity = summary_stats.get('sparsity_mean', float('nan'))
        n_nonzero = summary_stats.get('n_nonzero_mean', float('nan'))
        coef_sse = summary_stats.get('coef_sse_mean', float('nan'))
        train_time = summary_stats.get('train_time_mean', float('nan'))

        print(f"{prefix}📊 指标汇总:")
        print(f"{prefix}   ├─ 分类性能: Acc={accuracy:.6f}, F1={f1:.6f}, AUC={auc:.6f}")
        print(f"{prefix}   ├─ 变量选择: FDR={fdr:.6f}, TPR={tpr:.6f}")
        print(f"{prefix}   ├─ 稀疏性: Sparsity={sparsity:.4f}, n_nonzero={n_nonzero:.1f}")
        print(f"{prefix}   ├─ 系数估计: Coef_SSE={coef_sse:.6f}")
        print(f"{prefix}   └─ 运行时间: Train={train_time:.4f}s")


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

    best_mse = float('inf')
    best_accuracy = -float('inf')
    best_params = None
    best_fdr = float('inf')
    best_summary = None

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
        current_mse = summary_stats.get('mse_mean', float('inf'))
        current_accuracy = summary_stats.get('accuracy_mean', -float('inf'))
        current_fdr = summary_stats.get('fdr_mean', float('inf'))

        if family == "gaussian":
            # 回归任务：优先看MSE（越小越好）
            if current_mse < best_mse or (current_mse == best_mse and current_fdr < best_fdr):
                best_mse = current_mse
                best_fdr = current_fdr
                best_params = params.copy()
                best_summary = summary_stats.copy()
                print(f"   🌟 发现新最优参数!")
                print_metrics_summary(summary_stats, family, prefix="   ")
        else:
            # 分类任务：优先看准确率
            if current_accuracy > best_accuracy or (current_accuracy == best_accuracy and current_fdr < best_fdr):
                best_accuracy = current_accuracy
                best_fdr = current_fdr
                best_params = params.copy()
                best_summary = summary_stats.copy()
                print(f"   🌟 发现新最优参数!")
                print_metrics_summary(summary_stats, family, prefix="   ")

    print(f"\n✅ Stage1完成!")
    if best_params:
        print(f"🏆 最优结构化参数: {best_params}")
        if best_summary:
            print_metrics_summary(best_summary, family, prefix="   ")

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

    best_mse = float('inf')
    best_accuracy = -float('inf')
    best_params = None
    best_fdr = float('inf')
    best_lambda = None
    best_summary = None

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
        current_mse = summary_stats.get('mse_mean', float('inf'))
        current_accuracy = summary_stats.get('accuracy_mean', -float('inf'))
        current_fdr = summary_stats.get('fdr_mean', float('inf'))

        if family == "gaussian":
            # 回归任务：优先看MSE（越小越好）
            if current_mse < best_mse or (current_mse == best_mse and current_fdr < best_fdr):
                best_mse = current_mse
                best_fdr = current_fdr
                best_lambda = lambda_val
                best_params = params.copy()
                best_summary = summary_stats.copy()
                print(f"   🌟 发现新最优lambda! lambda={best_lambda:.6f}")
                print_metrics_summary(summary_stats, family, prefix="   ")
        else:
            # 分类任务：优先看准确率
            if current_accuracy > best_accuracy or (current_accuracy == best_accuracy and current_fdr < best_fdr):
                best_accuracy = current_accuracy
                best_fdr = current_fdr
                best_lambda = lambda_val
                best_params = params.copy()
                best_summary = summary_stats.copy()
                print(f"   🌟 发现新最优lambda! lambda={best_lambda:.6f}")
                print_metrics_summary(summary_stats, family, prefix="   ")

    print(f"\n✅ Stage2完成!")
    if best_params:
        print(f"🏆 最优完整参数: {best_params}")
        if best_summary:
            print_metrics_summary(best_summary, family, prefix="   ")

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
