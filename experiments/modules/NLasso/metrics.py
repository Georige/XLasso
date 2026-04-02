"""
NLasso 内置指标计算模块
与Benchmark维度完全对齐：精度、稀疏性、运行时间、收敛性
"""
import numpy as np
from typing import Optional, Tuple

# =============================================================================
# 精度指标
# =============================================================================

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方误差（MSE）：越小越好"""
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差（MAE）：越小越好"""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R²决定系数：越大越好，范围(-inf, 1]"""
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    if ss_total < 1e-10:
        return 1.0 if ss_residual < 1e-10 else 0.0
    return 1.0 - (ss_residual / ss_total)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """分类准确率：越大越好"""
    return np.mean(y_true == y_pred)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """F1分数：精确率和召回率的调和平均，越大越好"""
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray, beta_true: np.ndarray, threshold: float = 1e-6) -> float:
    """真阳性率（TPR）：选对的真实变量占所有真实变量的比例，越大越好"""
    true_nonzero = np.abs(beta_true) > threshold
    pred_nonzero = np.abs(y_pred) > threshold

    if np.sum(true_nonzero) == 0:
        return 1.0

    return np.sum(true_nonzero & pred_nonzero) / np.sum(true_nonzero)


def false_discovery_rate(y_true: np.ndarray, y_pred: np.ndarray, beta_true: np.ndarray, threshold: float = 1e-6) -> float:
    """假发现率（FDR）：选错的变量占所有选中变量的比例，越小越好"""
    true_nonzero = np.abs(beta_true) > threshold
    pred_nonzero = np.abs(y_pred) > threshold

    if np.sum(pred_nonzero) == 0:
        return 0.0

    return np.sum(~true_nonzero & pred_nonzero) / np.sum(pred_nonzero)


def auc_score(y_true: np.ndarray, y_score: np.ndarray, pos_label: int = 1) -> float:
    """
    AUC（Area Under Curve）：分类任务的曲线下面积
    使用 sklearn.metrics.roc_auc_score 计算
    """
    from sklearn.metrics import roc_auc_score

    y_true_binary = (y_true == pos_label).astype(int)

    # 处理边界情况：只有一个类别
    if len(np.unique(y_true_binary)) < 2:
        return float('nan')

    # 处理边界情况：y_score 全相同
    if np.all(y_score == y_score[0]):
        # 所有分数相同 → 返回 nan（无法区分）
        return float('nan')

    try:
        return roc_auc_score(y_true_binary, y_score)
    except Exception:
        return float('nan')


# =============================================================================
# 稀疏性指标
# =============================================================================

def sparsity(coef: np.ndarray, threshold: float = 1e-6) -> float:
    """稀疏度：零系数占比，越大越稀疏"""
    return np.sum(np.abs(coef) <= threshold) / len(coef)


def n_nonzero(coef: np.ndarray, threshold: float = 1e-6) -> int:
    """非零系数个数：越小越稀疏"""
    return int(np.sum(np.abs(coef) > threshold))


def coef_sse(beta_true: np.ndarray, beta_pred: np.ndarray) -> float:
    """系数误差平方和：衡量系数估计的准确性，越小越好"""
    return np.sum((beta_true - beta_pred) ** 2)


# =============================================================================
# 综合指标计算
# =============================================================================

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coef: Optional[np.ndarray] = None,
    beta_true: Optional[np.ndarray] = None,
    train_time: Optional[float] = None,
    predict_time: Optional[float] = None,
    n_iter: Optional[int] = None,
    threshold: float = 1e-6
) -> dict:
    """
    计算回归任务所有指标
    Returns: 包含所有指标的字典
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

    if coef is not None:
        metrics['sparsity'] = sparsity(coef, threshold)
        metrics['n_nonzero'] = n_nonzero(coef, threshold)

        if beta_true is not None:
            metrics['coef_sse'] = coef_sse(beta_true, coef)
            metrics['tpr'] = true_positive_rate(None, coef, beta_true, threshold)
            metrics['fdr'] = false_discovery_rate(None, coef, beta_true, threshold)

    if train_time is not None:
        metrics['train_time'] = train_time

    if predict_time is not None:
        metrics['predict_time'] = predict_time

    if n_iter is not None:
        metrics['n_iter'] = n_iter

    return metrics


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    coef: Optional[np.ndarray] = None,
    beta_true: Optional[np.ndarray] = None,
    train_time: Optional[float] = None,
    predict_time: Optional[float] = None,
    n_iter: Optional[int] = None,
    threshold: float = 1e-6,
    pos_label: int = 1
) -> dict:
    """
    计算分类任务所有指标
    Returns: 包含所有指标的字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, pos_label)
    }

    if y_score is not None:
        metrics['auc'] = auc_score(y_true, y_score, pos_label)

    if coef is not None:
        metrics['sparsity'] = sparsity(coef, threshold)
        metrics['n_nonzero'] = n_nonzero(coef, threshold)

        if beta_true is not None:
            metrics['coef_sse'] = coef_sse(beta_true, coef)
            metrics['tpr'] = true_positive_rate(None, coef, beta_true, threshold)
            metrics['fdr'] = false_discovery_rate(None, coef, beta_true, threshold)

    if train_time is not None:
        metrics['train_time'] = train_time

    if predict_time is not None:
        metrics['predict_time'] = predict_time

    if n_iter is not None:
        metrics['n_iter'] = n_iter

    return metrics
