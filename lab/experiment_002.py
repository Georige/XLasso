#!/usr/bin/env python3
"""
实验2：AR(1)相关稀疏回归
n=300, p=1000, 变量间AR(1)相关系数rho=0.8
奇数索引前50个变量系数服从U(0.5,2)，剩余950个为0
对比5种方法在不同信噪比下的性能
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加父目录到路径，导入unilasso模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unilasso.uni_lasso import cv_unilasso, cv_uni, predict
from data_generator import generate_experiment2

# ------------------------------------------------------------------------------
# 实验配置
# ------------------------------------------------------------------------------
CONFIG = {
    "n_repeats": 5,  # 实验重复次数，可通过命令行参数覆盖
    "sigmas": [0.5, 1.0, 2.5],  # 噪声标准差，对应低/中/高信噪比
    "family": "gaussian",
    "n_folds": 5,  # 交叉验证折数
    "seed": 42,
    # 对比方法配置（和实验1保持一致）
    "methods": {
        "1. UniLasso (基准)": {
            "type": "original",  # 原版带非负硬约束
            "params": {}
        },
        "2. 软约束": {
            "type": "xlasso",
            "params": {
                "adaptive_weighting": False,
                "enable_group_constraint": False,
                "alpha": 1.0,
                "beta": 1.0,
                "backend": "numba"
            }
        },
        "3. 软约束+自适应惩罚": {
            "type": "xlasso",
            "params": {
                "adaptive_weighting": True,
                "enable_group_constraint": False,
                "alpha": 1.0,
                "beta": 1.0,
                "weight_method": "p_value",
                "backend": "numba"
            }
        },
        "4. 软约束+组约束": {
            "type": "xlasso",
            "params": {
                "adaptive_weighting": False,
                "enable_group_constraint": True,
                "alpha": 1.0,
                "beta": 1.0,
                "group_penalty": 5.0,
                "corr_threshold": 0.7,
                "backend": "numba"
            }
        },
        "5. 完整XLasso": {
            "type": "xlasso",
            "params": {
                "adaptive_weighting": True,
                "enable_group_constraint": True,
                "alpha": 1.0,
                "beta": 1.0,
                "group_penalty": 5.0,
                "weight_method": "p_value",
                "corr_threshold": 0.7,
                "backend": "numba"
            }
        }
    }
}

# ------------------------------------------------------------------------------
# 评价指标计算
# ------------------------------------------------------------------------------
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      beta_true: np.ndarray, beta_pred: np.ndarray) -> dict:
    """计算所有评价指标"""
    # 预测性能
    mse = np.mean((y_true - y_pred) ** 2)

    # 变量选择性能
    true_nonzero = np.where(beta_true != 0)[0]
    pred_nonzero = np.where(np.abs(beta_pred) > 1e-8)[0]

    tp = len(np.intersect1d(true_nonzero, pred_nonzero))
    fp = len(pred_nonzero) - tp
    fn = len(true_nonzero) - tp

    tpr = tp / len(true_nonzero) if len(true_nonzero) > 0 else 0.0
    fdr = fp / len(pred_nonzero) if len(pred_nonzero) > 0 else 0.0
    precision = 1.0 - fdr
    recall = tpr

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # 稀疏性
    n_selected = len(pred_nonzero)

    return {
        "mse": mse,
        "tpr": tpr,
        "fdr": fdr,
        "f1": f1,
        "n_selected": n_selected
    }

# ------------------------------------------------------------------------------
# 主实验逻辑
# ------------------------------------------------------------------------------
def run_experiment():
    np.random.seed(CONFIG["seed"])

    # 初始化结果存储
    all_results = []

    # 实验重复
    for repeat in tqdm(range(CONFIG["n_repeats"]), desc="实验重复"):
        # 遍历不同信噪比
        for sigma in tqdm(CONFIG["sigmas"], desc="信噪比", leave=False):
            # 生成数据（AR(1)相关）
            X, y, beta_true = generate_experiment2(sigma=sigma)
            n, p = X.shape

            # 遍历所有方法
            for method_name, method_config in tqdm(CONFIG["methods"].items(), desc="方法", leave=False):
                try:
                    # 拟合模型
                    if method_config["type"] == "original":
                        # 原版UniLasso（带非负硬约束）
                        fit = cv_unilasso(
                            X=X,
                            y=y,
                            family=CONFIG["family"],
                            n_folds=CONFIG["n_folds"],
                            seed=CONFIG["seed"] + repeat,
                            verbose=False
                        )
                    else:
                        # XLasso系列方法
                        fit = cv_uni(
                            X=X,
                            y=y,
                            family=CONFIG["family"],
                            n_folds=CONFIG["n_folds"],
                            seed=CONFIG["seed"] + repeat,
                            verbose=False,
                            **method_config["params"]
                        )

                    # 预测（使用最佳lambda）
                    # 先提取最佳模型，再预测
                    best_fit = extract_cv(fit)
                    y_pred = predict(best_fit, X)

                    # 计算指标
                    metrics = calculate_metrics(y, y_pred, beta_true, beta_pred)

                    # 保存结果
                    result_row = {
                        "repeat": repeat,
                        "sigma": sigma,
                        "method": method_name,
                        **metrics
                    }
                    all_results.append(result_row)

                    # 实时打印关键结果
                    tqdm.write(f"[重复{repeat} σ={sigma}] {method_name:<20} | MSE: {metrics['mse']:.4f} | F1: {metrics['f1']:.4f} | TPR: {metrics['tpr']:.4f} | FDR: {metrics['fdr']:.4f}")

                except Exception as e:
                    tqdm.write(f"[错误] {method_name} 在σ={sigma} 重复{repeat}失败: {str(e)}")
                    continue

    # 转换为DataFrame并保存
    df = pd.DataFrame(all_results)
    result_dir = "result/exp_002"
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "experiment_002_results.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n实验完成！结果已保存到: {output_path}")

    # 打印汇总结果
    print("\n" + "="*80)
    print("实验2 汇总结果（平均值）：")
    print("="*80)

    summary = df.groupby(["sigma", "method"]).agg({
        "mse": "mean",
        "tpr": "mean",
        "fdr": "mean",
        "f1": "mean",
        "n_selected": "mean"
    }).round(4)

    print(summary.to_string())
    print("="*80)

    # 保存汇总结果
    summary_path = os.path.join(result_dir, "experiment_002_summary.csv")
    summary.to_csv(summary_path, encoding="utf-8-sig")
    print(f"汇总结果已保存到: {summary_path}")

    return df, summary

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="实验2：AR(1)相关稀疏回归")

    # 基础实验参数
    parser.add_argument("-n", "--n-repeats", type=int, default=CONFIG["n_repeats"],
                        help=f"实验重复次数，默认值：{CONFIG['n_repeats']}")
    parser.add_argument("--sigmas", nargs="+", type=float, default=CONFIG["sigmas"],
                        help=f"噪声标准差列表，默认值：{' '.join(map(str, CONFIG['sigmas']))}")
    parser.add_argument("--family", type=str, default=CONFIG["family"],
                        choices=["gaussian", "binomial", "poisson", "multinomial", "cox"],
                        help=f"GLM家族类型，默认值：{CONFIG['family']}")
    parser.add_argument("--n-folds", type=int, default=CONFIG["n_folds"],
                        help=f"交叉验证折数，默认值：{CONFIG['n_folds']}")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"],
                        help=f"随机种子，默认值：{CONFIG['seed']}")
    parser.add_argument("--rho", type=float, default=0.8,
                        help="AR(1)相关系数，默认值：0.8")

    # 模型超参数
    parser.add_argument("--backend", type=str, default="numba", choices=["numba", "pytorch"],
                        help="求解器后端，默认值：numba")
    parser.add_argument("--group-penalty", type=float, default=5.0,
                        help="组一致性惩罚强度，默认值：5.0")
    parser.add_argument("--corr-threshold", type=float, default=0.7,
                        help="分组相关系数阈值，默认值：0.7")
    parser.add_argument("--weight-method", type=str, default="p_value",
                        choices=["t_statistic", "p_value", "correlation"],
                        help="显著性权重计算方法，默认值：p_value")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="XLasso alpha参数（显著变量负惩罚强度），默认值：1.0")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="XLasso beta参数（不显著变量惩罚强度），默认值：1.0")
    parser.add_argument("--lmda-min-ratio", type=float, default=1e-4,
                        help="正则化路径最小lambda比例，默认值：1e-4")

    args = parser.parse_args()

    # 更新全局配置
    CONFIG["n_repeats"] = args.n_repeats
    CONFIG["sigmas"] = args.sigmas
    CONFIG["family"] = args.family
    CONFIG["n_folds"] = args.n_folds
    CONFIG["seed"] = args.seed

    # 更新所有方法的公共参数
    for method in CONFIG["methods"].values():
        if method["type"] == "xlasso":
            method["params"].update({
                "backend": args.backend,
                "alpha": args.alpha,
                "beta": args.beta,
                "group_penalty": args.group_penalty,
                "corr_threshold": args.corr_threshold,
                "weight_method": args.weight_method,
                "lmda_min_ratio": args.lmda_min_ratio
            })

    # 更新generate_experiment2的rho参数
    CONFIG["rho"] = args.rho

    # 打印配置信息
    print("="*80)
    print("实验2 配置信息：")
    for k, v in vars(args).items():
        print(f"  {k:<20} = {v}")
    print("="*80)

    run_experiment()
