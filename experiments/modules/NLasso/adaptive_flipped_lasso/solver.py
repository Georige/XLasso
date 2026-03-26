"""
AdaptiveFlippedLasso 求解器模块
包含权重计算和拟合的核心算法
"""
import numpy as np
from sklearn.linear_model import Ridge, Lasso, RidgeCV, lasso_path


def compute_adaptive_weights(beta_ridge: np.ndarray, gamma: float = 1.0, eps: float = 1e-5) -> np.ndarray:
    """
    计算归一化自适应权重

    步骤1: w_raw = 1 / (|β_ridge| + ε)^γ
    步骤2: w_norm = w_raw / mean(w_raw)，不限制上限

    注意：不再裁剪到 [0,1]，因为噪声特征的大权重（>1）是算法的关键，
    配合非负 Lasso 可以有效抑制噪声。

    Args:
        beta_ridge: Ridge 回归系数 (p,)
        gamma: 指数衰减参数
        eps: 防止除零的小常量
    Returns:
        weights: 归一化权重 (p,)
    """
    raw_weights = 1.0 / (np.abs(beta_ridge) + eps) ** gamma
    weights = raw_weights / np.mean(raw_weights)
    # 注意：不裁剪到 [0,1]，噪声权重大于1是算法的关键
    return weights


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


def fit_adaptive_flipped_lasso_cv(
    X: np.ndarray,
    y: np.ndarray,
    lambda_ridge_list: tuple = (0.1, 1.0, 10.0, 100.0),
    gamma_list: tuple = (0.5, 1.0, 2.0),
    cv: int = 5,
    alpha_min_ratio: float = 1e-4,
    n_alpha: int = 100,
    max_iter: int = 1000,
    tol: float = 1e-4,
    fit_intercept: bool = True,
    random_state: int = 2026,
    verbose: bool = False,
) -> dict:
    """
    严格隔离的 AdaptiveFlippedLasso 交叉验证

    算法流程（Strict NLasso CV）：
    阶段一：K 折严格内部寻优
        for each fold k:
            严格隔离训练折 / 验证折
            RidgeCV(仅训练折) → beta_ridge_k
            for each gamma:
                计算 weights_k（基于训练折）
                扭曲训练折和验证折
                LassoPath(训练折) → 全部 alpha 的系数路径
                在验证折上打分 → Error_Matrix[gamma, alpha, k]
    阶段二：选拔最优参数（平均 MSE 最小的 gamma, alpha）
    阶段三：全量数据终极拟合（带回最佳参数）

    Args:
        X: 特征矩阵 (n, p)
        y: 目标向量 (n,)
        lambda_ridge_list: Ridge alpha 候选值
        gamma_list: gamma 指数候选值
        cv: 交叉验证折数
        alpha_min_ratio: alpha 下界比例
        n_alpha: Lasso 路径 alpha 数量
        max_iter: 最大迭代次数
        tol: 收敛阈值
        fit_intercept: 是否拟合截距
        random_state: 随机种子
        verbose: 是否打印详细信息

    Returns:
        dict with keys: signs, weights, coef_, intercept_, best_gamma_, best_alpha_,
                        best_lambda_ridge_, beta_ridge_, cv_score_
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error

    n = X.shape[0]
    n_gamma = len(gamma_list)
    eps = 1e-5

    # ============================================================
    # 阶段一：K 折严格内部寻优
    # ============================================================
    # Error_Matrix[gamma_idx, alpha_idx, fold_idx]
    error_matrix = np.full((n_gamma, n_alpha, cv), np.inf)

    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        if verbose:
            print(f"[Fold {fold_idx + 1}/{cv}] train={len(train_idx)}, val={len(val_idx)}")

        # --- 先验提取：仅在训练折上做 RidgeCV（绝对隔离）---
        ridge_cv = RidgeCV(alphas=lambda_ridge_list, cv=3)
        ridge_cv.fit(X_tr, y_tr)
        beta_ridge_fold = ridge_cv.coef_

        signs_fold = np.sign(beta_ridge_fold)
        signs_fold[signs_fold == 0] = 1.0

        # --- 遍历不同 gamma 评估（每个 gamma 有自己专属的 alpha 路径）---
        for gamma_idx, gamma in enumerate(gamma_list):
            # 计算权重（仅基于训练折 Ridge）
            raw_weights = 1.0 / (np.abs(beta_ridge_fold) + eps) ** gamma
            weights = raw_weights / np.mean(raw_weights)

            # 空间重构：分别扭曲训练折和验证折
            X_adaptive_tr = (X_tr * signs_fold) / weights
            X_adaptive_va = (X_va * signs_fold) / weights

            # 每个 gamma 有自己专属的 alpha 搜索路径
            alpha_max = np.max(np.abs(X_adaptive_tr.T @ y_tr)) / len(y_tr)
            alpha_min = alpha_max * alpha_min_ratio
            alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)[::-1]

            # lasso_path：一次性算出全部 alpha 的系数
            _, coefs_path, _ = lasso_path(
                X_adaptive_tr, y_tr,
                alphas=alphas,
                positive=True,
                max_iter=max_iter,
                tol=tol,
            )

            # 在验证集上打分
            preds = X_adaptive_va @ coefs_path  # (n_val, n_alphas)

            # MSE: mean((y_va - pred_i)^2), 向量化
            mse_path = np.mean((y_va[:, np.newaxis] - preds) ** 2, axis=0)  # (n_alphas,)
            error_matrix[gamma_idx, :, fold_idx] = mse_path

    # ============================================================
    # 阶段二：选拔最优参数
    # ============================================================
    # 对 K 折求平均：mean_error[gamma_idx, alpha_idx]
    mean_error = np.mean(error_matrix, axis=2)

    # 找最小 MSE 对应的 (gamma_idx, alpha_idx)
    best_gamma_idx, best_alpha_idx = np.unravel_index(
        np.argmin(mean_error), mean_error.shape
    )
    best_gamma = gamma_list[best_gamma_idx]
    best_cv_mse = mean_error[best_gamma_idx, best_alpha_idx]

    # 重建 best_gamma 对应的 alpha 路径（在全量数据上重新算，以获取正确的 alpha 值）
    ridge_stage2 = RidgeCV(alphas=lambda_ridge_list, cv=cv)
    ridge_stage2.fit(X, y)
    beta_ridge_s2 = ridge_stage2.coef_
    signs_s2 = np.sign(beta_ridge_s2)
    signs_s2[signs_s2 == 0] = 1.0
    raw_weights_s2 = 1.0 / (np.abs(beta_ridge_s2) + eps) ** best_gamma
    weights_s2 = raw_weights_s2 / np.mean(raw_weights_s2)
    X_ad_s2 = (X * signs_s2) / weights_s2
    alpha_max_s2 = np.max(np.abs(X_ad_s2.T @ y)) / len(y)
    alpha_min_s2 = alpha_max_s2 * alpha_min_ratio
    alphas_s2 = np.logspace(np.log10(alpha_min_s2), np.log10(alpha_max_s2), n_alpha)[::-1]
    best_alpha = alphas_s2[best_alpha_idx]

    if verbose:
        print(f"\n[Stage 2] Best: gamma={best_gamma}, alpha={best_alpha:.6f}, CV_MSE={best_cv_mse:.6f}")

    # ============================================================
    # 阶段三：全量数据终极拟合
    # ============================================================
    # RidgeCV on 全量数据 → 最终先验
    ridge_final = RidgeCV(alphas=lambda_ridge_list, cv=cv)
    ridge_final.fit(X, y)
    beta_ridge_final = ridge_final.coef_
    best_lambda_ridge = ridge_final.alpha_

    signs_final = np.sign(beta_ridge_final)
    signs_final[signs_final == 0] = 1.0

    # 最终权重
    raw_weights_final = 1.0 / (np.abs(beta_ridge_final) + eps) ** best_gamma
    weights_final = raw_weights_final / np.mean(raw_weights_final)

    # 最终空间重构
    X_adaptive_final = (X * signs_final) / weights_final

    # 最终 Lasso 拟合（用阶段二选出的最优 alpha）
    lasso_final = Lasso(
        alpha=best_alpha,
        positive=True,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    lasso_final.fit(X_adaptive_final, y)

    # 逆重构系数
    final_coef = (lasso_final.coef_ / weights_final) * signs_final

    if fit_intercept:
        intercept_final = lasso_final.intercept_
    else:
        intercept_final = 0.0

    if verbose:
        print(f"[Stage 3] Final: lambda_ridge={best_lambda_ridge}, "
              f"gamma={best_gamma}, alpha={best_alpha:.6f}, "
              f"non_zero={np.sum(final_coef != 0)}/{len(final_coef)}")

    return {
        'signs': signs_final,
        'weights': weights_final,
        'coef_': final_coef,
        'intercept_': intercept_final,
        'best_gamma_': best_gamma,
        'best_alpha_': best_alpha,
        'best_lambda_ridge_': best_lambda_ridge,
        'beta_ridge_': beta_ridge_final,
        'cv_score_': -best_cv_mse,        # 负 MSE，越大越好
        'cv_mse_': best_cv_mse,
        'mean_error_matrix_': mean_error,  # (n_gamma, n_alpha), 用于分析
        'alphas_': alphas,
        'gamma_list_': gamma_list,
    }
