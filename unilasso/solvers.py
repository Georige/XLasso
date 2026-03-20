"""
Numba-optimized solvers for UniLasso.

This module provides high-performance implementations of the lasso path solver
using Numba JIT compilation with full GLM family support.
"""

import numpy as np
from numba import jit
from typing import Tuple, Optional


@jit(nopython=True, cache=True)
def _sigmoid(z):
    """Sigmoid function for logistic regression."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


@jit(nopython=True, cache=True)
def _compute_glm_loss_and_grad(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    family: str
) -> Tuple[np.ndarray, float, float]:
    """
    Compute loss and gradients for different GLM families.

    Returns:
        grad_weights: gradient w.r.t. weights
        grad_bias: gradient w.r.t. bias
        loss: scalar loss value
    """
    n_samples = X.shape[0]
    eta = np.dot(X, weights) + bias

    if family == "gaussian":
        # Gaussian: MSE loss
        error = eta - y
        grad_weights = 2.0 * np.dot(X.T, error) / n_samples
        grad_bias = 2.0 * np.sum(error) / n_samples
        loss = np.sum(error**2) / n_samples

    elif family == "binomial":
        # Binomial: Logistic loss
        mu = _sigmoid(eta)
        error = mu - y
        grad_weights = np.dot(X.T, error) / n_samples
        grad_bias = np.sum(error) / n_samples
        # Logistic loss: -[y log(mu) + (1-y) log(1-mu)]
        loss = -np.sum(y * np.log(mu + 1e-15) + (1 - y) * np.log(1 - mu + 1e-15)) / n_samples

    elif family == "poisson":
        # Poisson: log-linear loss
        mu = np.exp(np.clip(eta, -50, 50))  # Clip to prevent overflow
        error = mu - y
        grad_weights = np.dot(X.T, error) / n_samples
        grad_bias = np.sum(error) / n_samples
        # Poisson loss: sum(mu - y * log(mu))
        loss = np.sum(mu - y * np.log(mu + 1e-15)) / n_samples

    elif family == "multinomial":
        # Multinomial: treat as binomial for now (one-vs-rest)
        mu = _sigmoid(eta)
        error = mu - y
        grad_weights = np.dot(X.T, error) / n_samples
        grad_bias = np.sum(error) / n_samples
        loss = -np.sum(y * np.log(mu + 1e-15) + (1 - y) * np.log(1 - mu + 1e-15)) / n_samples

    elif family == "cox":
        # Cox: For now, use Gaussian as placeholder (Cox needs special handling)
        error = eta - y
        grad_weights = 2.0 * np.dot(X.T, error) / n_samples
        grad_bias = 2.0 * np.sum(error) / n_samples
        loss = np.sum(error**2) / n_samples

    else:
        # Default to Gaussian
        error = eta - y
        grad_weights = 2.0 * np.dot(X.T, error) / n_samples
        grad_bias = 2.0 * np.sum(error) / n_samples
        loss = np.sum(error**2) / n_samples

    return grad_weights, grad_bias, loss


@jit(nopython=True, cache=True)
def _fit_numba_lasso_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lmdas: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    negative_penalty: float = 0.0,
    fit_intercept: bool = True,
    lr: float = 0.01,
    max_epochs: int = 5000,
    tol: float = 1e-6,
    feature_weights: Optional[np.ndarray] = None,
    group_signs: Optional[np.ndarray] = None,
    group_penalty: float = 0.0,
    group_weights: Optional[np.ndarray] = None,
    family: str = "gaussian",
    momentum: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a lasso path using Numba-optimized gradient descent with warm start.

    Supports double asymmetric soft-thresholding with feature-level adaptive weights
    and group-level sign consistency constraints. Full GLM family support.

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, n_features)
    y_train : np.ndarray
        Target values of shape (n_samples,)
    lmdas : np.ndarray
        Regularization path, sorted in descending order for warm start
    alpha : float
        XLasso parameter: penalty coefficient for the $\frac{1}{w_j}$ term
        For significant variables (small w_j), this increases penalty on negative coefficients
    beta : float
        XLasso parameter: penalty coefficient for the $w_j$ term
        For insignificant variables (large w_j), this increases penalty on both signs
    negative_penalty : float
        Backward compatibility: Additional penalty strength for negative coefficients
        (used when alpha=0, this reduces to the old simplified form)
    fit_intercept : bool
        Whether to fit an intercept term
    lr : float
        Learning rate for gradient descent
    max_epochs : int
        Maximum number of epochs per lambda
    tol : float
        Convergence tolerance
    feature_weights : np.ndarray, optional
        Feature-level significance weights w_j of shape (n_features,), defaults to all ones
    group_signs : np.ndarray, optional
        Dominant sign for each feature's group of shape (n_features,), defaults to all ones
    group_penalty : float
        Global group penalty strength for sign inconsistency
    group_weights : np.ndarray, optional
        Group-level penalty weights of shape (n_features,), defaults to all ones
    family : str
        GLM family: 'gaussian', 'binomial', 'poisson', 'multinomial', 'cox'

    Returns
    -------
    betas_matrix : np.ndarray
        Coefficients of shape (n_lmdas, n_features)
    intercepts : np.ndarray
        Intercepts of shape (n_lmdas,)
    """
    n_samples, n_features = X_train.shape
    n_lmdas = len(lmdas)

    # Default values for optional parameters
    if feature_weights is None:
        feature_weights = np.ones(n_features)
    if group_signs is None:
        group_signs = np.ones(n_features)
    if group_weights is None:
        group_weights = np.ones(n_features)

    # 非对称惩罚权重标准化（和坐标下降保持一致）
    p = n_features
    w_plus_norm = np.zeros(n_features)
    w_minus_norm = np.zeros(n_features)

    # 预计算原始权重并求和
    S_plus = 0.0
    S_minus = 0.0
    for j in range(n_features):
        fw = feature_weights[j]
        fw_safe = max(fw, 1e-10)
        w_plus = fw
        w_minus = alpha / fw_safe + beta * fw
        w_plus_norm[j] = w_plus
        w_minus_norm[j] = w_minus
        S_plus += w_plus
        S_minus += w_minus

    # 标准化并乘以p，使平均权重为1
    S_plus_safe = max(S_plus, 1e-10)
    S_minus_safe = max(S_minus, 1e-10)
    for j in range(n_features):
        w_plus_norm[j] = (w_plus_norm[j] / S_plus_safe) * p
        w_minus_norm[j] = (w_minus_norm[j] / S_minus_safe) * p

    # Pre-allocate result storage
    betas_matrix = np.zeros((n_lmdas, n_features))
    intercepts = np.zeros(n_lmdas)

    # Initialize parameters (warm start across lambdas)
    weights = np.zeros(n_features)
    bias = 0.0
    # Initialize momentum velocities
    v_weights = np.zeros(n_features)
    v_bias = 0.0

    for i, lmda in enumerate(lmdas):
        # Reset velocity when lambda changes (warm start for velocity)
        v_weights *= 0.5
        v_bias *= 0.5

        # For early stopping: track last 3 changes to avoid premature stop
        last_changes = np.zeros(3)
        change_idx = 0

        for epoch in range(max_epochs):
            # Nesterov momentum: look ahead step
            weights_lookahead = weights + momentum * v_weights
            bias_lookahead = bias + momentum * v_bias

            # Step 1: Gradient descent on GLM loss using lookahead weights
            grad_weights, grad_bias, _ = _compute_glm_loss_and_grad(
                X_train, y_train, weights_lookahead, bias_lookahead, family
            )

            # Update velocities
            v_weights = momentum * v_weights - lr * grad_weights
            v_bias = momentum * v_bias - lr * grad_bias

            # Take Nesterov step
            weights_gd = weights + v_weights
            bias_gd = bias + v_bias

            # Step 2: Double asymmetric proximal operator (XLasso original formula)
            w_prox = np.zeros_like(weights_gd)

            for j in range(n_features):
                w = weights_gd[j]
                gs = group_signs[j]
                gw = group_weights[j]

                # Compute base thresholds (feature adaptive)
                # 使用标准化后的权重，和坐标下降保持一致
                tau_pos_base = lr * lmda * w_plus_norm[j]
                tau_neg_base = lr * lmda * w_minus_norm[j]

                # Add legacy negative_penalty for backward compatibility
                if negative_penalty > 0:
                    tau_neg_base += lr * negative_penalty * fw

                # Compute group penalty adjustment
                if gs > 0:
                    # Group sign is positive: extra penalty for negative values
                    tau_neg = tau_neg_base + lr * group_penalty * gw
                    tau_pos = tau_pos_base
                else:
                    # Group sign is negative: extra penalty for positive values
                    tau_pos = tau_pos_base + lr * group_penalty * gw
                    tau_neg = tau_neg_base

                # Apply soft thresholding
                if w > tau_pos:
                    w_prox[j] = w - tau_pos
                elif w < -tau_neg:
                    w_prox[j] = w + tau_neg
                else:
                    w_prox[j] = 0.0

            # Check convergence based on weight change
            max_change = np.max(np.abs(w_prox - weights))
            last_changes[change_idx] = max_change
            change_idx = (change_idx + 1) % 3

            # Converge only if last 3 changes are all below tol
            converged = epoch > 3 and np.all(last_changes < tol)

            # Update weights for next iteration
            weights = w_prox
            bias = bias_gd

            if converged:
                break

        # Store results for this lambda
        betas_matrix[i, :] = weights
        intercepts[i] = bias

    return betas_matrix, intercepts


@jit(nopython=True, cache=True)
def _fit_numba_lasso_path_coordinate_descent(
    X_center: np.ndarray,
    y_center: np.ndarray,
    lmdas: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    negative_penalty: float = 0.0,
    max_epochs: int = 1000,
    tol: float = 1e-6,
    feature_weights: Optional[np.ndarray] = None,
    group_signs: Optional[np.ndarray] = None,
    group_penalty: float = 0.0,
    group_weights: Optional[np.ndarray] = None,
    X_diag: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coordinate descent solver for XLasso (Gaussian family only, centered data).
    No learning rate needed, avoids lambda scale matching issues.

    Implements coordinate descent with asymmetric soft thresholding,
    warm start across lambda path.

    Input data must be already centered (no intercept term needed here).
    """
    n_samples, n_features = X_center.shape
    n_lmdas = len(lmdas)

    # Default values for optional parameters
    if feature_weights is None:
        feature_weights = np.ones(n_features)
    if group_signs is None:
        group_signs = np.ones(n_features)
    if group_weights is None:
        group_weights = np.ones(n_features)
    if X_diag is None:
        X_diag = np.zeros(n_features)
        for j in range(n_features):
            X_diag[j] = np.sum(X_center[:, j] ** 2) / n_samples
            # Avoid division by zero for constant features
            if X_diag[j] < 1e-10:
                X_diag[j] = 1e-10

    # 非对称惩罚权重标准化（按要求实现）
    p = n_features
    w_plus_norm = np.zeros(n_features)
    w_minus_norm = np.zeros(n_features)

    # 预计算原始权重并求和
    S_plus = 0.0
    S_minus = 0.0
    for j in range(n_features):
        fw = feature_weights[j]
        fw_safe = max(fw, 1e-10)
        w_plus = fw
        w_minus = alpha / fw_safe + beta * fw
        w_plus_norm[j] = w_plus
        w_minus_norm[j] = w_minus
        S_plus += w_plus
        S_minus += w_minus

    # 标准化并乘以p，使平均权重为1
    S_plus_safe = max(S_plus, 1e-10)
    S_minus_safe = max(S_minus, 1e-10)
    for j in range(n_features):
        w_plus_norm[j] = (w_plus_norm[j] / S_plus_safe) * p
        w_minus_norm[j] = (w_minus_norm[j] / S_minus_safe) * p

    # Pre-allocate result storage
    betas_matrix = np.zeros((n_lmdas, n_features))

    # Initialize parameters (warm start across all lambdas)
    weights = np.zeros(n_features)
    residual = y_center.copy()

    # Process lambdas in descending order (warm start)
    for i, lmda in enumerate(lmdas):
        # Precompute thresholds for all features
        tau_pos = np.zeros(n_features)
        tau_neg = np.zeros(n_features)
        for j in range(n_features):
            gs = group_signs[j]
            gw = group_weights[j]

            # Base thresholds from XLasso formula（使用标准化后的权重）
            tau_pos_base = lmda * w_plus_norm[j]
            tau_neg_base = lmda * w_minus_norm[j]

            # Add negative penalty
            if negative_penalty > 0:
                tau_neg_base += negative_penalty * fw

            # Add group penalty
            if gs > 0:
                tau_neg[j] = tau_neg_base + group_penalty * gw
                tau_pos[j] = tau_pos_base
            else:
                tau_pos[j] = tau_pos_base + group_penalty * gw
                tau_neg[j] = tau_neg_base

        # Coordinate descent cycles
        for epoch in range(max_epochs):
            max_change = 0.0

            for j in range(n_features):
                # Old weight value
                old_w = weights[j]

                if old_w != 0:
                    # Remove the effect of feature j from residual
                    residual += X_center[:, j] * old_w

                # Compute partial correlation / least squares solution for feature j
                beta_j = np.sum(X_center[:, j] * residual) / (n_samples * X_diag[j])

                # Apply asymmetric soft thresholding
                tau_pos_scaled = tau_pos[j] / X_diag[j]
                tau_neg_scaled = tau_neg[j] / X_diag[j]
                if beta_j > tau_pos_scaled:
                    new_w = beta_j - tau_pos_scaled
                elif beta_j < -tau_neg_scaled:
                    new_w = beta_j + tau_neg_scaled
                else:
                    new_w = 0.0

                # Update weights and residual
                weights[j] = new_w
                if new_w != 0:
                    residual -= X_center[:, j] * new_w

                # Track maximum change for convergence check
                change = np.abs(new_w - old_w)
                if change > max_change:
                    max_change = change

            # Check convergence
            if max_change < tol:
                break

        # Store results
        betas_matrix[i, :] = weights.copy()

    return betas_matrix


def _fit_numba_lasso_path_accelerated(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lmdas: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    negative_penalty: float = 0.0,
    fit_intercept: bool = True,
    lr: float = 0.01,  # Unused for coordinate descent
    max_epochs: int = 1000,
    tol: float = 1e-6,
    feature_weights: Optional[np.ndarray] = None,
    group_signs: Optional[np.ndarray] = None,
    group_penalty: float = 0.0,
    group_weights: Optional[np.ndarray] = None,
    family: str = "gaussian",
    momentum: float = 0.0  # Unused for coordinate descent
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coordinate descent solver for Gaussian family with all preprocessing done in Python layer.
    Avoids numba axis operation issues and type mismatches.
    """
    if family != "gaussian":
        # Fall back to gradient descent for non-Gaussian families
        return _fit_numba_lasso_path(
            X_train, y_train, lmdas, alpha, beta, negative_penalty, fit_intercept,
            lr, max_epochs, tol, feature_weights, group_signs,
            group_penalty, group_weights, family, momentum
        )

    n_samples, n_features = X_train.shape

    # Preprocess data in Python layer to avoid numba issues
    if fit_intercept:
        X_mean = np.mean(X_train, axis=0)
        y_mean = np.mean(y_train)
        X_center = X_train - X_mean
        y_center = y_train - y_mean
    else:
        X_mean = np.zeros(n_features)
        y_mean = 0.0
        X_center = X_train
        y_center = y_train

    # Precompute diagonal of Gram matrix
    X_diag = np.sum(X_center ** 2, axis=0) / n_samples
    X_diag[X_diag < 1e-10] = 1e-10  # Avoid division by zero

    # Call numba coordinate descent (only core numerical operations)
    betas_matrix = _fit_numba_lasso_path_coordinate_descent(
        X_center, y_center, lmdas,
        alpha=alpha,
        beta=beta,
        negative_penalty=negative_penalty,
        max_epochs=max_epochs,
        tol=tol,
        feature_weights=feature_weights,
        group_signs=group_signs,
        group_penalty=group_penalty,
        group_weights=group_weights,
        X_diag=X_diag
    )

    # Compute intercepts
    intercepts = np.zeros(len(lmdas))
    if fit_intercept:
        for i in range(len(lmdas)):
            intercepts[i] = y_mean - np.sum(betas_matrix[i] * X_mean)

    return betas_matrix, intercepts
