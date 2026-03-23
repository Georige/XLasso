#!/usr/bin/env python3
"""
NLasso 最优参数记忆模块
持久化存储每个模拟实验场景下的最优结构化参数和指标
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# 记忆文件路径
MEMORY_FILE = Path(__file__).parent / "optimal_params.json"


def load_optimal_params() -> Dict[int, Dict[str, Any]]:
    """
    加载已保存的最优参数

    Returns:
        字典，key为实验ID，value为最优参数字典
    """
    if not MEMORY_FILE.exists():
        return {}

    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_optimal_params(
    experiment_id: int,
    experiment_name: str,
    optimal_params: Dict[str, Any],
    optimal_metrics: Dict[str, Any],
    stage: int = 3,
    family: str = "gaussian"
):
    """
    保存最优参数和指标到记忆文件

    Args:
        experiment_id: 实验ID (1-7)
        experiment_name: 实验名称
        optimal_params: 最优参数字典
        optimal_metrics: 最优指标字典
        stage: 调优阶段 (1, 2, or 3)
        family: 任务类型 ("gaussian" or "binomial")
    """
    all_params = load_optimal_params()

    key = f"{experiment_id}_{family}"
    all_params[key] = {
        'experiment_id': experiment_id,
        'experiment_name': experiment_name,
        'family': family,
        'stage': stage,
        'optimal_params': optimal_params,
        'optimal_metrics': optimal_metrics,
        'saved_at': __import__('datetime').datetime.now().isoformat()
    }

    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_params, f, indent=2, ensure_ascii=False)


def get_optimal_params(experiment_id: int, family: str = "gaussian") -> Optional[Dict[str, Any]]:
    """
    获取指定实验的最优参数

    Args:
        experiment_id: 实验ID (1-7)
        family: 任务类型 ("gaussian" or "binomial")

    Returns:
        最优参数字典，如果不存在则返回None
    """
    all_params = load_optimal_params()
    key = f"{experiment_id}_{family}"
    if key in all_params:
        return all_params[key]['optimal_params']
    return None


def get_optimal_metrics(experiment_id: int, family: str = "gaussian") -> Optional[Dict[str, Any]]:
    """
    获取指定实验的最优指标

    Args:
        experiment_id: 实验ID (1-7)
        family: 任务类型 ("gaussian" or "binomial")

    Returns:
        最优指标字典，如果不存在则返回None
    """
    all_params = load_optimal_params()
    key = f"{experiment_id}_{family}"
    if key in all_params:
        return all_params[key]['optimal_metrics']
    return None


def list_all_optimal() -> Dict[str, Dict[str, Any]]:
    """列出所有已保存的最优参数"""
    return load_optimal_params()


if __name__ == "__main__":
    print("="*60)
    print("NLasso 最优参数记忆模块")
    print("="*60)
    print(f"\n记忆文件: {MEMORY_FILE}")

    all_optimal = list_all_optimal()
    if not all_optimal:
        print("\n暂无保存的最优参数")
    else:
        print(f"\n已保存 {len(all_optimal)} 组最优参数:")
        for key, data in sorted(all_optimal.items()):
            print(f"\n  {key}: {data['experiment_name']} ({data['family']})")
            print(f"    阶段: {data['stage']}")
            print(f"    最优参数: {data['optimal_params']}")
            print(f"    最优指标: {data['optimal_metrics']}")
            print(f"    保存时间: {data['saved_at']}")
