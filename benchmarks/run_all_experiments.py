#!/usr/bin/env python3
"""
连续运行所有7个benchmark实验，每两分钟报告进度和阶段性结论
"""
import os
import sys
import time
import subprocess
from pathlib import Path
import pandas as pd
from datetime import datetime

# 配置
BENCHMARK_SCRIPT = Path(__file__).parent / "benchmark_run.py"
RESULT_DIR = Path(__file__).parent / "result"
EXPERIMENTS = list(range(1, 8))
FAMILIES = ["gaussian", "binomial"]
CHECK_INTERVAL = 120  # 2分钟检查一次

# 跟踪任务状态
tasks = {}
completed_tasks = set()
failed_tasks = set()

def run_experiment(exp_id, family):
    """运行单个实验"""
    cmd = [
        sys.executable, str(BENCHMARK_SCRIPT),
        "--mode", "custom",
        "--dataset", f"exp{exp_id}",
        "--family", family,
        "--n-repeats", "3",
        "--cv", "5",
        "--seed-start", "2026"
    ]
    log_file = Path(__file__).parent / f"exp{exp_id}_{family}.log"
    print(f"🚀 启动实验 {exp_id} [{family}]，日志文件: {log_file.name}")

    # 后台运行
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).parent
    )
    tasks[(exp_id, family)] = {
        "proc": proc,
        "log_file": log_file,
        "start_time": datetime.now(),
        "status": "running"
    }

def check_task_status():
    """检查所有任务状态"""
    global completed_tasks, failed_tasks
    for (exp_id, family), task_info in tasks.items():
        if task_info["status"] != "running":
            continue

        retcode = task_info["proc"].poll()
        if retcode is None:
            continue

        if retcode == 0:
            task_info["status"] = "completed"
            completed_tasks.add((exp_id, family))
            print(f"✅ 实验 {exp_id} [{family}] 完成，耗时: {datetime.now() - task_info['start_time']}")
        else:
            task_info["status"] = "failed"
            failed_tasks.add((exp_id, family))
            print(f"❌ 实验 {exp_id} [{family}] 失败，退出码: {retcode}，日志: {task_info['log_file'].name}")

def get_latest_result_dir():
    """获取最新的结果目录"""
    if not RESULT_DIR.exists():
        return None
    dirs = [d for d in RESULT_DIR.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda x: x.stat().st_mtime)

def analyze_results(result_dir):
    """分析已生成的结果"""
    if not result_dir:
        return {}

    raw_files = list(result_dir.rglob("*raw.csv"))
    if not raw_files:
        return {}

    all_results = []
    for f in raw_files:
        try:
            df = pd.read_csv(f)
            all_results.append(df)
        except Exception as e:
            continue

    if not all_results:
        return {}

    combined = pd.concat(all_results, ignore_index=True)

    # 动态构建聚合字典，只包含存在的列
    agg_dict = {}
    possible_metrics = ["r2", "accuracy", "tpr", "fdr", "train_time"]
    for metric in possible_metrics:
        if metric in combined.columns:
            agg_dict[metric] = "mean"

    # 计算每个算法的平均指标
    if not agg_dict:
        return {}

    algo_summary = combined.groupby("algorithm").agg(agg_dict).round(4).reset_index()

    analysis = {
        "total_algorithms": len(algo_summary["algorithm"].unique()),
        "metrics": {
            "r2_mean": algo_summary["r2"].mean() if "r2" in algo_summary.columns else None,
            "accuracy_mean": algo_summary["accuracy"].mean() if "accuracy" in algo_summary.columns else None,
            "tpr_mean": algo_summary["tpr"].mean() if "tpr" in algo_summary.columns else None,
            "fdr_mean": algo_summary["fdr"].mean() if "fdr" in algo_summary.columns else None,
            "time_mean": algo_summary["train_time"].mean() if "train_time" in algo_summary.columns else None
        },
        "best_algorithms": {}
    }

    # 找出各指标最优算法
    if "r2" in algo_summary.columns and not algo_summary["r2"].isna().all():
        valid_r2 = algo_summary.dropna(subset=["r2"])
        if len(valid_r2) > 0:
            best_r2 = valid_r2.sort_values("r2", ascending=False).iloc[0]
            analysis["best_algorithms"]["r2"] = (best_r2["algorithm"], best_r2["r2"])

    if "tpr" in algo_summary.columns and not algo_summary["tpr"].isna().all():
        valid_tpr = algo_summary.dropna(subset=["tpr"])
        if len(valid_tpr) > 0:
            best_tpr = valid_tpr.sort_values("tpr", ascending=False).iloc[0]
            analysis["best_algorithms"]["tpr"] = (best_tpr["algorithm"], best_tpr["tpr"])

    if "fdr" in algo_summary.columns and not algo_summary["fdr"].isna().all():
        valid_fdr = algo_summary.dropna(subset=["fdr"])
        if len(valid_fdr) > 0:
            best_fdr = valid_fdr.sort_values("fdr", ascending=True).iloc[0]
            analysis["best_algorithms"]["fdr"] = (best_fdr["algorithm"], best_fdr["fdr"])

    return analysis

def print_progress():
    """打印进度报告"""
    total_tasks = len(EXPERIMENTS) * len(FAMILIES)
    completed = len(completed_tasks)
    failed = len(failed_tasks)
    running = total_tasks - completed - failed

    print("\n" + "="*80)
    print(f"📊 进度报告 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print("="*80)
    print(f"总任务数: {total_tasks} | 运行中: {running} | 已完成: {completed} | 失败: {failed}")
    print(f"完成度: {completed / total_tasks * 100:.1f}%")

    result_dir = get_latest_result_dir()
    if result_dir:
        print(f"\n📂 结果存储目录: {result_dir.resolve()}")
        analysis = analyze_results(result_dir)
        if analysis:
            print(f"\n📈 阶段性分析:")
            print(f"已评估算法数: {analysis['total_algorithms']}")
            metrics = analysis['metrics']
            if metrics['r2_mean'] is not None:
                print(f"平均R²: {metrics['r2_mean']:.4f}")
            if metrics['accuracy_mean'] is not None:
                print(f"平均准确率: {metrics['accuracy_mean']:.4f}")
            if metrics['tpr_mean'] is not None:
                print(f"平均真阳性率: {metrics['tpr_mean']:.4f}")
            if metrics['fdr_mean'] is not None:
                print(f"平均假发现率: {metrics['fdr_mean']:.4f}")
            if metrics['time_mean'] is not None:
                print(f"平均运行时间: {metrics['time_mean']:.2f}s")

            if analysis['best_algorithms']:
                print(f"\n🏆 当前最优算法:")
                for metric, (alg, value) in analysis['best_algorithms'].items():
                    if metric == 'fdr':
                        print(f"  {metric.upper()}: {alg} ({value:.4f} ↓)")
                    else:
                        print(f"  {metric.upper()}: {alg} ({value:.4f} ↑)")
        else:
            print("\n⏳ 结果正在生成中...")
    else:
        print("\n📂 结果目录尚未生成")

    if failed_tasks:
        print(f"\n❌ 失败的任务:")
        for exp_id, family in failed_tasks:
            print(f"  实验{exp_id} [{family}]")

    print("="*80 + "\n")

def main():
    print("🚀 启动所有benchmark实验...")
    print(f"共 {len(EXPERIMENTS)} 个实验，每个实验2个任务类型，总计 {len(EXPERIMENTS)*2} 个任务")

    # 启动所有实验
    for exp_id in EXPERIMENTS:
        for family in FAMILIES:
            run_experiment(exp_id, family)

    # 主循环
    try:
        while True:
            check_task_status()
            print_progress()

            if len(completed_tasks) + len(failed_tasks) == len(tasks):
                break

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n⚠️  收到中断信号，停止所有任务...")
        for task_info in tasks.values():
            if task_info["proc"].poll() is None:
                task_info["proc"].terminate()
        sys.exit(1)

    # 最终报告
    print("\n" + "="*80)
    print("🎉 所有实验执行完毕!")
    print("="*80)
    result_dir = get_latest_result_dir()
    if result_dir:
        print(f"最终结果目录: {result_dir.resolve()}")
        analysis = analyze_results(result_dir)
        if analysis:
            print(f"\n📊 最终汇总:")
            print(f"总评估算法数: {analysis['total_algorithms']}")
            metrics = analysis['metrics']
            if metrics['r2_mean'] is not None:
                print(f"平均R²: {metrics['r2_mean']:.4f}")
            if metrics['accuracy_mean'] is not None:
                print(f"平均准确率: {metrics['accuracy_mean']:.4f}")
            if metrics['tpr_mean'] is not None:
                print(f"平均真阳性率: {metrics['tpr_mean']:.4f}")
            if metrics['fdr_mean'] is not None:
                print(f"平均假发现率: {metrics['fdr_mean']:.4f}")
            if metrics['time_mean'] is not None:
                print(f"平均运行时间: {metrics['time_mean']:.2f}s")

            print(f"\n🏆 全局最优算法:")
            for metric, (alg, value) in analysis['best_algorithms'].items():
                if metric == 'fdr':
                    print(f"  {metric.upper()}: {alg} ({value:.4f} ↓)")
                else:
                    print(f"  {metric.upper()}: {alg} ({value:.4f} ↑)")

    if failed_tasks:
        print(f"\n❌ 失败任务数: {len(failed_tasks)}")
        for exp_id, family in failed_tasks:
            print(f"  实验{exp_id} [{family}] - 日志: benchmarks/exp{exp_id}_{family}.log")
    else:
        print("\n✅ 所有任务全部成功!")

    print("="*80)

if __name__ == "__main__":
    main()
