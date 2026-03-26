"""
AdaptiveFlippedLasso 求解器模块
包含权重计算和拟合的核心算法
"""
import numpy as np
from sklearn.linear_model import Ridge, Lasso


def compute_adaptive_weights(beta_ridge: np.ndarray, gamma: float = 1.0, eps: float = 1e-5) -> np.ndarray:
    """
    计算归一化自适应权重

    步骤1: w_raw = 1 / (|β_ridge| + ε)^γ
    步骤2: w_norm = w_raw / mean(w_raw)，限制在 [0, 1]

    Args:
        beta_ridge: Ridge 回归系数 (p,)
        gamma: 指数衰减参数
        eps: 防止除零的小常量
    Returns:
        weights: 归一化权重 (p,)
    """
    raw_weights = 1.0 / (np.abs(beta_ridge) + eps) ** gamma
    weights = raw_weights / np.mean(raw_weights)
    return np.clip(weights, 0.0, 1.0)


def flip_features(X: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """
    翻转特征方向使与 Ridge 系数同向

    Args:
        X: 特征矩阵 (n, p)
        signs: 符号向量 (p,)
    Returns:
        X_flipped: 翻转后的特征矩阵 (n, p)
    """
    return X * signs


def scale_features(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    特征缩放：X / weights

    Args:
        X: 特征矩阵 (n, p) 或 (n, p_flipped)
        weights: 权重向量 (p,) 或 (p_flipped,)
    Returns:
        X_scaled: 缩放后的特征矩阵
    """
    return X / weights


def reconstruct_coefficients(
    coef_adaptive: np.ndarray,
    weights: np.ndarray,
    signs: np.ndarray
) -> np.ndarray:
    """
    逆重构系数：final_coef = (coef / weights) * signs

    Args:
        coef_adaptive: 非负 Lasso 系数 (p,)
        weights: 归一化权重 (p,)
        signs: 特征方向符号 (p,)
    Returns:
        final_coef: 原始空间的系数 (p,)
    """
    return (coef_adaptive / weights) * signs


def fit_adaptive_flipped_lasso(
    X: np.ndarray,
    y: np.ndarray,
    lambda_ridge: float = 10.0,
    lambda_: float = 0.01,
    gamma: float = 1.0,
    alpha_min_ratio: float = 1e-4,
    n_alpha: int = 50,
    max_iter: int = 1000,
    tol: float = 1e-4,
    fit_intercept: bool = True,
    random_state: int = 2026,
    verbose: bool = False,
) -> dict:
    """
    完整的 AdaptiveFlippedLasso 拟合流程

    Returns:
        dict with keys: signs, weights, coef_, intercept_, lambda_, beta_ridge_
    """
    # 第一阶段：Ridge 回归
    ridge = Ridge(alpha=lambda_ridge, fit_intercept=fit_intercept, random_state=random_state)
    ridge.fit(X, y)
    beta_ridge = ridge.coef_

    # 符号
    signs = np.sign(beta_ridge)
    signs[signs == 0] = 1.0

    # 权重
    weights = compute_adaptive_weights(beta_ridge, gamma)

    # 翻转和缩放
    X_flipped = flip_features(X, signs)
    X_adaptive = scale_features(X_flipped, weights)

    # 计算 alpha 路径
    if lambda_ is not None:
        alphas = [lambda_]
    else:
        alpha_max = np.max(np.abs(X_adaptive.T @ y)) / len(y)
        alpha_min = alpha_max * alpha_min_ratio
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)[::-1]

    # 求解最优 alpha
    best_score = -np.inf
    best_coef = None
    best_alpha = None

    for alpha in alphas:
        model = Lasso(
            alpha=alpha,
            positive=True,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        model.fit(X_adaptive, y)
        score = model.score(X_adaptive, y)

        if score > best_score:
            best_score = score
            best_coef = model.coef_.copy()
            best_alpha = alpha

    if best_coef is None:
        best_coef = np.zeros(X_adaptive.shape[1])
        best_alpha = alphas[0]

    # 逆重构
    final_coef = reconstruct_coefficients(best_coef, weights, signs)

    # 截距
    if fit_intercept:
        intercept = np.mean(y) - np.mean(X_flipped / weights @ final_coef)
    else:
        intercept = 0.0

    if verbose:
        print(f"[AdaptiveFlippedLasso] lambda_ridge={lambda_ridge}, lambda={best_alpha:.6f}, non-zero={np.sum(final_coef != 0)}")

    return {
        'signs': signs,
        'weights': weights,
        'coef_': final_coef,
        'intercept_': intercept,
        'lambda_': best_alpha,
        'beta_ridge_': beta_ridge,
    }
