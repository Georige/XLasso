"""
模拟实验数据生成器
包含论文中的7种模拟实验场景
"""
import numpy as np


def generate_experiment1_data(n=300, p=500, sigma=1.0, family='gaussian'):
    """实验1：高维成对相关稀疏回归 (paper.md Section 5.1.1)
    n=300, p=500, X~N(0,Σ) with Σ_ij=0.5, first 20 variables β=1.0, σ=1.0
    """
    # 协方差矩阵：成对相关0.5
    cov = np.ones((p, p)) * 0.5
    np.fill_diagonal(cov, 1.0)
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 真实系数：前20个变量 β=1.0，其余480个为0
    beta_true = np.zeros(p)
    beta_true[:20] = 1.0

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:  # binomial
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true


def generate_experiment2_data(n=300, p=500, sigma=1.0, rho=0.8, family='gaussian'):
    """实验2：AR(1)相关稀疏回归 (paper.md Section 5.1.2)
    n=300, p=500, X~N(0,Σ) with Σ_ij=0.8^|i-j|, odd-indexed first 20 variables β=1.0, σ=1.0
    """
    # AR(1)协方差矩阵
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = rho ** abs(i - j)
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 真实系数：奇数索引的前20个变量 β=1.0 (j=1,3,5,...,39)，其余480个为0
    beta_true = np.zeros(p)
    beta_true[1:40:2] = 1.0

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true


def generate_experiment3_data(n=300, p=500, sigma=1.0, rho=0.8, family='binomial'):
    """实验3：二分类偏移变量选择 (paper.md Section 5.1.3)
    n=300, p=500, X~N(0,Σ) with AR(1) ρ=0.8, first 20 variables β=1.0
    y=1 samples have offset 0.6 on first 20 variables, ~150 each class
    """
    # AR(1)相关性结构
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = rho ** abs(i - j)
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 真实系数：前20个变量 β=1.0，其余480个为0
    beta_true = np.zeros(p)
    beta_true[:20] = 1.0

    if family == 'binomial':
        # 二分类：基于系数生成y，再对y=1样本施加偏移
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
        # 确保类别平衡：约150个y=1，约150个y=0
        y1_idx = y == 1
        if np.sum(y1_idx) > 160:
            # 如果y=1过多，随机降采样
            drop_idx = np.random.choice(np.where(y1_idx)[0], int(np.sum(y1_idx) - 150), replace=False)
            y[drop_idx] = 0
        elif np.sum(y1_idx) < 140:
            # 如果y=1过少，随机上采样一些y=1
            add_idx = np.random.choice(np.where(y == 0)[0], int(150 - np.sum(y1_idx)), replace=False)
            y[add_idx] = 1
        # 对y=1的样本，前20个变量增加偏移量0.6
        y1_idx = y == 1
        X[y1_idx, :20] += 0.6
    else:
        # 回归任务
        y = X @ beta_true + np.random.randn(n) * sigma

    return X, y, beta_true


def generate_experiment4_data(n=300, p=1000, sigma=1.0, rho=0.85, family='gaussian'):
    """实验4：反符号孪生变量 (paper.md Section 5.1.4)
    n=300, p=1000, 10 pairs twin variables with ρ=0.85
    β_{2t-1}=2.0, β_{2t}=-2.5, remaining 980 variables are noise
    """
    X = np.random.randn(n, p)
    beta_true = np.zeros(p)

    # 生成10对反符号孪生变量
    for i in range(10):
        common = np.random.randn(n)
        X[:, 2*i] = common * np.sqrt(rho) + np.random.randn(n) * np.sqrt(1-rho)
        X[:, 2*i+1] = -common * np.sqrt(rho) + np.random.randn(n) * np.sqrt(1-rho)
        beta_true[2*i] = 2.0
        beta_true[2*i+1] = -2.5

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true


def generate_experiment5_data(n=300, p=1000, sigma=1.0, rho=0.8, family='gaussian'):
    """实验5 魔鬼等级1：绝对隐身陷阱 Perfect Masking (paper.md Section 5.1.5)
    n=300, p=1000, 10 pairs twin variables with ρ=0.8
    β_{2t-1}=2.0, β_{2t}=-2.5, remaining 980 variables are independent noise
    核心挑战：通过精确设计X与y的关系使真实信号的边际相关性归零
    """
    X = np.random.randn(n, p)
    beta_true = np.zeros(p)

    # 生成10对孪生变量（与exp4相同结构）
    for i in range(10):
        common = np.random.randn(n)
        X[:, 2*i] = common * np.sqrt(rho) + np.random.randn(n) * np.sqrt(1-rho)
        X[:, 2*i+1] = -common * np.sqrt(rho) + np.random.randn(n) * np.sqrt(1-rho)
        beta_true[2*i] = 2.0
        beta_true[2*i+1] = -2.5

    # 完美掩藏：构造X使每个孪生变量对的边际效应归零
    # 原理：y = β₁X₁ + β₂X₂ + ...，若X₁,X₂共享common factor，
    # 则通过在构造X时让其与y正交，可以使单变量回归系数→0
    # 实际实现：在生成X后，对每个孪生变量对施以与y精确抵消的调整
    y_base = X @ beta_true + np.random.randn(n) * sigma

    # 对每个孪生变量对，旋转使边际相关归零
    for i in range(10):
        # 计算X_{2i}和X_{2i+1}在y_base上的边际系数
        # 通过Householder变换旋转这对变量，使其与y_base的协方差归一化
        # 实现：找与[y_base, X_{2i}, X_{2i+1}]正交的方向
        v1 = X[:, 2*i].copy()
        v2 = X[:, 2*i+1].copy()

        # 使v1与y正交（边际相关归零）
        proj_y = np.dot(v1, y_base) / (np.dot(y_base, y_base) + 1e-10)
        v1_ortho = v1 - proj_y * y_base
        v1_ortho = v1_ortho / (np.linalg.norm(v1_ortho) + 1e-10)

        # 对v2同样处理，使正交于y_base
        proj_y2 = np.dot(v2, y_base) / (np.dot(y_base, y_base) + 1e-10)
        v2_ortho = v2 - proj_y2 * y_base
        v2_ortho = v2_ortho / (np.linalg.norm(v2_ortho) + 1e-10)

        X[:, 2*i] = v1_ortho * np.std(X[:, 2*i]) + np.mean(X[:, 2*i])
        X[:, 2*i+1] = v2_ortho * np.std(X[:, 2*i+1]) + np.mean(X[:, 2*i+1])

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true


def generate_experiment6_data(n=300, p=500, sigma=1.0, rho=0.8, family='gaussian'):
    """实验6 魔鬼等级2：鸠占鹊巢陷阱 The Decoy Trap (paper.md Section 5.1.6)
    n=300, p=500, 每3变量一组共5组
    真信号: X1,X2,X4,X5,X7,X8,X13,X14,X16,X17 (每组前2个)，β=1.0
    噪声诱饵: X3,X6,X9,X15,X18 (每组第3个)，与同组真信号ρ=0.8
    剩余482个独立噪声变量，β=0
    """
    X = np.random.randn(n, p)
    beta_true = np.zeros(p)

    # 5组三变量结构 (0-indexed: 对应paper中的1-indexed)
    # 组0: X0,X1,X2 | 组1: X3,X4,X5 | 组2: X6,X7,X8
    # 组3: X12,X13,X14 | 组4: X15,X16,X17
    # 真信号: 每组前2个变量 (a,b)；噪声诱饵: 第3个变量 (c)
    groups = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (12, 13, 14), (15, 16, 17)]

    for a, b, c in groups:
        # a,b为真信号(独立), c为噪声诱饵(与a,b相关)
        common_ab = np.random.randn(n) * np.sqrt(0.5)
        indep_a = np.random.randn(n) * np.sqrt(0.5)
        indep_b = np.random.randn(n) * np.sqrt(0.5)
        indep_c = np.random.randn(n) * np.sqrt(1 - rho)

        X[:, a] = common_ab + indep_a
        X[:, b] = common_ab + indep_b
        # c与a,b均相关ρ
        common_ac = np.random.randn(n) * np.sqrt(rho)
        X[:, c] = common_ac + indep_c

        beta_true[a] = 1.0
        beta_true[b] = 1.0
        # beta_true[c] = 0 (保持为0)

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true


def generate_experiment7_data(n=300, p=500, sigma=1.0, rho=0.9, family='gaussian'):
    """实验7 魔鬼等级3：自回归符号雪崩 AR(1) Sign Avalanche (paper.md Section 5.1.7)
    n=300, p=500, 前20变量AR(1)链条 Σ_ij=0.9^|i-j|
    β_j = (-1)^(j+1) * 2.0 * 0.9^((j-1)/2), j=1..20
    例如: β1=2.0, β2=-1.8, β3=1.62, β4=-1.458, ...
    剩余480个变量β=0
    """
    # AR(1)协方差矩阵
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov[i, j] = rho ** abs(i - j)
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)

    # 真实系数：前20个变量正负交替衰减
    beta_true = np.zeros(p)
    for j in range(1, 21):
        beta_true[j-1] = ((-1) ** (j + 1)) * 2.0 * (0.9 ** ((j - 1) / 2))

    if family == 'gaussian':
        y = X @ beta_true + np.random.randn(n) * sigma
    else:
        z = X @ beta_true + np.random.randn(n) * sigma
        y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

    return X, y, beta_true
