import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple

class SoftUniLassoLoss(nn.Module):
    """
    自定义的 Lasso 损失函数，包含均方误差 (MSE)、L1 正则化以及创新的负系数惩罚。
    """
    def __init__(self, lmda: float, negative_penalty: float):
        super(SoftUniLassoLoss, self).__init__()
        self.lmda = lmda
        self.negative_penalty = negative_penalty

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # 1. 基础数据拟合损失 (MSE)
        mse_loss = torch.mean((y_pred - y_true) ** 2)

        # 2. L1 正则化 (Lasso) - 促使特征稀疏化
        l1_penalty = float(self.lmda) * torch.sum(torch.abs(weights))

        # 3. 创新点：对负系数的软惩罚
        # 使用 ReLU 激活函数拦截负值：relu(-weights) 会将正权重变为 0，将负权重转为正值并进行惩罚
        neg_penalty = float(self.negative_penalty) * torch.sum(torch.relu(-weights))

        return mse_loss + l1_penalty + neg_penalty


class SoftUniLassoLossBinomial(nn.Module):
    """
    自定义的 Binomial Lasso 损失函数，包含 logistic loss、L1 正则化以及创新的负系数惩罚。
    """
    def __init__(self, lmda: float, negative_penalty: float):
        super(SoftUniLassoLossBinomial, self).__init__()
        self.lmda = lmda
        self.negative_penalty = negative_penalty

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        使用 binary cross-entropy 损失 (logistic loss)
        logits: 线性预测值 X @ w + b
        y_true: 二元标签 (0 或 1)
        """
        # 1. Logistic loss (binary cross-entropy)
        # 使用 PyTorch 的 binary_cross_entropy_with_logits 获得数值稳定性
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_true, reduction='mean'
        )

        # 2. L1 正则化 (Lasso) - 促使特征稀疏化
        l1_penalty = float(self.lmda) * torch.sum(torch.abs(weights))

        # 3. 创新点：对负系数的软惩罚
        neg_penalty = float(self.negative_penalty) * torch.sum(torch.relu(-weights))

        return bce_loss + l1_penalty + neg_penalty

def _fit_custom_lasso_pytorch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lmda: float,
    negative_penalty: float,
    epochs: int = 1000,
    lr: float = 0.01,
    family: str = "gaussian"
) -> Tuple[np.ndarray, float]:
    """
    使用 PyTorch 拟合带有自定义 Loss 的单次 Lasso 模型。
    """
    # 将 NumPy 数组转换为 PyTorch 张量，并指定 requires_grad=True 让系统追踪梯度
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    # 初始化权重和截距
    n_features = X_train.shape[1]
    weights = torch.randn((n_features, 1), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    # 官方文档参考：使用 Adam 优化器 (torch.optim.Adam)。替代方案是 SGD，但 Adam 收敛更平稳。
    optimizer = optim.Adam([weights, bias], lr=lr)

    if family == "gaussian":
        criterion = SoftUniLassoLoss(lmda, negative_penalty)
    else:
        criterion = SoftUniLassoLossBinomial(lmda, negative_penalty)

    for epoch in range(epochs):
        optimizer.zero_grad()           # 清空上一步的残余梯度
        logits = torch.matmul(X_tensor, weights) + bias  # 前向传播计算预测值

        if family == "gaussian":
            loss = criterion(logits, y_tensor, weights)      # 计算自定义损失
        else:
            # Binomial: logits is the linear part, criterion computes BCE
            loss = criterion(logits, y_tensor, weights)

        loss.backward()                 # 反向传播计算梯度
        optimizer.step()                # 更新权重

    return weights.detach().numpy().flatten(), bias.detach().item()


from sklearn.model_selection import KFold

def cv_uni(
    X: np.ndarray,
    y: np.ndarray,
    family: str = "gaussian",
    n_folds: int = 5,
    lmdas: Optional[np.ndarray] = None,
    negative_penalty: float = 1.0,
    seed: Optional[int] = None
) -> dict:  # 为简化展示，我们暂时返回字典，而非原始复杂的 UniLassoCVResult
    """
    创新版交叉验证单变量引导 Lasso 回归 (cv_uni)。
    支持 gaussian 和 binomial 两种 family。
    """
    if family not in ("gaussian", "binomial"):
        raise ValueError(f"family 必须为 'gaussian' 或 'binomial'，得到 '{family}'")

    if lmdas is None:
        # 默认生成一个 lambda 路径（从大到小）
        lmdas = np.logspace(-1, -4, 20)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # 用于记录每个 lambda 在所有折叠上的平均损失
    avg_losses = []

    print(f"开始使用 {n_folds} 折交叉验证，验证 {len(lmdas)} 个正则化参数 (family={family})...")

    for lmda in lmdas:
        fold_losses = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 1. 在训练集上拟合我们的自定义 PyTorch 模型
            weights, bias = _fit_custom_lasso_pytorch(
                X_train, y_train, lmda=lmda, negative_penalty=negative_penalty,
                family=family
            )

            # 2. 在验证集上评估
            if family == "gaussian":
                # MSE 评估
                y_val_pred = X_val @ weights + bias
                val_loss = np.mean((y_val_pred - y_val) ** 2)
            else:
                # Binomial: 使用负对数似然 (BCE)
                logits = X_val @ weights + bias
                # 数值稳定的 BCE 计算
                val_loss = np.mean(
                    np.where(y_val == 1,
                             np.log1p(np.exp(-np.clip(logits, -500, 500))),
                             np.log1p(np.exp(np.clip(logits, -500, 500))) + np.log(2))
                )
            fold_losses.append(val_loss)

        avg_losses.append(np.mean(fold_losses))

    # 寻找最佳的 lambda
    best_idx = np.argmin(avg_losses)
    best_lmda = lmdas[best_idx]

    print(f"最佳 Lambda: {best_lmda:.5f}")

    # 3. 使用最佳 lambda 在全量数据上进行最终拟合
    final_weights, final_bias = _fit_custom_lasso_pytorch(
        X, y, lmda=best_lmda, negative_penalty=negative_penalty, family=family
    )

    # 组装结果
    return {
        "coefs": final_weights,
        "intercept": final_bias,
        "best_lmda": best_lmda,
        "lmdas": lmdas,
        "avg_losses": avg_losses
    }