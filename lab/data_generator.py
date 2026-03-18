#!/usr/bin/env python3
"""
XLasso 模拟实验数据生成器
包含所有设计的模拟实验场景的数据生成逻辑
"""
import numpy as np
from scipy.linalg import toeplitz
from typing import Tuple


def generate_experiment1(n: int = 300, p: int = 1000, sigma: float = 1.0, corr: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    实验1：高维成对相关稀疏回归
    变量间成对相关性为corr，前100个系数来自标准正态分布，其余为0

    Args:
        n: 样本量
        p: 变量维度
        sigma: 噪声标准差
        corr: 变量间成对相关系数

    Returns:
        X: 设计矩阵 (n, p)
        y: 响应变量 (n,)
        true_beta: 真实系数 (p,)
    """
    # 生成协方差矩阵
    cov = np.ones((p, p)) * corr
    np.fill_diagonal(cov, 1.0)

    # 生成X
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 生成真实系数
    true_beta = np.zeros(p)
    true_beta[:100] = np.random.randn(100)

    # 生成响应
    y = X @ true_beta + np.random.randn(n) * sigma

    return X, y, true_beta


def generate_experiment2(n: int = 300, p: int = 1000, sigma: float = 1.0, rho: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    实验2：AR(1)相关稀疏回归
    变量间AR(1)相关性rho = 0.8，奇数索引前50个系数服从U(0.5,2)，其余为0

    Args:
        n: 样本量
        p: 变量维度
        sigma: 噪声标准差
        rho: AR(1)相关系数

    Returns:
        X: 设计矩阵 (n, p)
        y: 响应变量 (n,)
        true_beta: 真实系数 (p,)
    """
    # 生成AR(1)协方差矩阵
    cov = toeplitz(rho ** np.arange(p))

    # 生成X
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 生成真实系数：第1,3,5,...,99共50个奇数索引变量
    true_beta = np.zeros(p)
    non_zero_idx = np.arange(0, 100, 2)  # 0-based索引，对应1,3,...99
    true_beta[non_zero_idx] = np.random.uniform(0.5, 2.0, size=len(non_zero_idx))

    # 生成响应
    y = X @ true_beta + np.random.randn(n) * sigma

    return X, y, true_beta


def generate_experiment3(n: int = 200, p: int = 500, offset: float = 0.5, rho: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    实验3：二分类偏移变量选择
    y随机从0,1抽取，y=1的样本前20个变量偏移offset

    Args:
        n: 样本量
        p: 变量维度
        offset: y=1样本前20个变量的偏移量
        rho: AR(1)相关系数

    Returns:
        X: 设计矩阵 (n, p)
        y: 二分类响应变量 (n,)
        true_variables: 真实相关变量索引 (前20个)
    """
    # 生成AR(1)协方差矩阵
    cov = toeplitz(rho ** np.arange(p))

    # 生成初始X
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 生成二分类响应
    y = np.random.binomial(1, 0.5, size=n)

    # 对y=1的样本前20个变量加偏移
    X[y == 1, :20] += offset

    # 真实相关变量是前20个
    true_beta = np.zeros(p)
    true_beta[:20] = 1.0

    return X, y, true_beta


def generate_experiment4(n: int = 100, sigma: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    实验4：高相关反符号变量选择
    前10个和中间10个变量高度相关但系数符号相反

    Args:
        n: 样本量
        sigma: 噪声标准差

    Returns:
        X: 设计矩阵 (n, 30)
        y: 响应变量 (n,)
        true_beta: 真实系数 (30,)
    """
    X = np.zeros((n, 30))

    # 前10个变量
    X[:, :10] = np.random.normal(4.0, 1.0, size=(n, 10))

    # 中间10个变量：前10个加噪声
    X[:, 10:20] = X[:, :10] + np.random.normal(0.0, 1.0, size=(n, 10))

    # 最后10个变量：独立噪声
    X[:, 20:] = np.random.normal(4.0, 1.0, size=(n, 10))

    # 生成真实系数
    true_beta = np.zeros(30)
    true_beta[:10] = np.random.uniform(0.0, 2.0, size=10)  # 正系数
    true_beta[10:20] = np.random.uniform(-2.0, 0.0, size=10)  # 负系数

    # 生成响应
    y = X @ true_beta + np.random.randn(n) * sigma

    return X, y, true_beta


def get_experiment_configs() -> dict:
    """获取所有实验的配置信息"""
    return {
        "experiment1": {
            "name": "高维成对相关稀疏回归",
            "params": {
                "n": 300,
                "p": 1000,
                "sigma_options": [0.5, 1.0, 2.5],
                "corr": 0.5
            },
            "description": "变量间成对相关0.5，前100个系数服从N(0,1)，其余为0"
        },
        "experiment2": {
            "name": "AR(1)相关稀疏回归",
            "params": {
                "n": 300,
                "p": 1000,
                "sigma_options": [0.5, 1.0, 2.5],
                "rho": 0.8
            },
            "description": "AR(1)相关rho=0.8，奇数索引前50个系数服从U(0.5,2)，其余为0"
        },
        "experiment3": {
            "name": "二分类偏移变量选择",
            "params": {
                "n": 200,
                "p": 500,
                "offset": 0.5,
                "rho": 0.8
            },
            "description": "二分类任务，y=1样本前20个变量偏移0.5，其余为噪声"
        },
        "experiment4": {
            "name": "高相关反符号变量选择",
            "params": {
                "n": 100,
                "sigma": 0.5
            },
            "description": "前10和中间10个变量高度相关，系数符号相反，验证模型鲁棒性"
        }
    }


if __name__ == "__main__":
    # 测试所有数据生成器
    print("Testing data generators...")

    # 实验1测试
    X1, y1, beta1 = generate_experiment1(n=10, p=200)  # 测试用p=200，保证至少有100个变量
    print(f"Experiment1: X shape={X1.shape}, y shape={y1.shape}, non-zero beta={np.sum(beta1 != 0)}")

    # 实验2测试
    X2, y2, beta2 = generate_experiment2(n=10, p=200)  # 测试用p=200，保证至少有100个变量
    print(f"Experiment2: X shape={X2.shape}, y shape={y2.shape}, non-zero beta={np.sum(beta2 != 0)}")

    # 实验3测试
    X3, y3, beta3 = generate_experiment3(n=10, p=50)  # 测试用p=50，保证至少有20个变量
    print(f"Experiment3: X shape={X3.shape}, y shape={y3.shape}, class count={np.sum(y3==0)}/{np.sum(y3==1)}")

    # 实验4测试
    X4, y4, beta4 = generate_experiment4(n=10)
    print(f"Experiment4: X shape={X4.shape}, y shape={y4.shape}, non-zero beta={np.sum(beta4 != 0)}")

    print("All data generators work correctly!")
