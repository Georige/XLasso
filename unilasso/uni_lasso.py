"""
UniLasso: Univariate-Guided Sparse Regression

This module implements core LOO functions for Univariate-Guided Lasso regression.

Reference: https://arxiv.org/abs/2501.18360
"""



import numpy as np
import pandas as pd
from numba import jit, prange
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import adelie as ad


from typing import List, Optional, Tuple, Union, Callable
import logging

from .univariate_regression import fit_loo_univariate_models
from .config import VALID_FAMILIES, VALID_UNIVARIATE_MODELS
from .utils import warn_zero_variance, warn_removed_lmdas
from .solvers import _fit_numba_lasso_path


# Configure logger
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------


def _compute_feature_significance_weights(
    univariate_results: dict,
    weight_method: str = "t_statistic",
    gamma: float = 1.0,
    sharp_scale: float = 0.0,
    weight_max_scale: float = None
) -> np.ndarray:
    """
    Compute feature-level significance weights for adaptive penalty.

    Parameters
    ----------
    univariate_results : dict
        Dictionary containing univariate regression results with keys:
        't_stats' (for t-statistic), 'p_values' (for p-value), 'correlations' (for correlation)
    weight_method : str
        Method to compute weights: 't_statistic', 'p_value', or 'correlation'
    gamma : float
        Contrast adjustment parameter for weight scaling:
        - gamma < 1: 平滑权重差异，适合高相关降维打击场景
        - gamma = 1: 默认线性映射
        - gamma > 1: 锐化权重差异，适合高频噪声场景
    sharp_scale : float
        Exponential sharpening parameter (only for p_value method):
        - sharp_scale = 0: 禁用指数锐化（默认）
        - sharp_scale > 0: 对p值大的非显著变量施加指数级惩罚增强
        推荐值：5~20，值越大非显著变量的惩罚增长越快
    weight_max_scale : float, optional
        If provided, weights are normalized to [1, weight_max_scale].
        Defaults to None, which returns raw weights without normalization.

    Returns
    -------
    weights : np.ndarray
        Array of significance weights of shape (n_features,)
    """
    n_features = len(univariate_results['beta'])

    if weight_method == "t_statistic":
        stats = np.abs(univariate_results.get('t_stats', np.ones(n_features)))
        # t统计量越大越显著，权重越大（和方案A匹配，w越大惩罚越轻）
        w_base = stats
        # 应用 gamma 对比度调整（gamma>1放大差异）
        weights = w_base ** gamma
    elif weight_method == "p_value":
        p_vals = univariate_results.get('p_values', np.ones(n_features))
        # 安全截断 p 值
        p_vals_safe = np.clip(p_vals, a_min=1e-7, a_max=0.99)
        # p值越小越显著，权重越大（和方案A匹配，w越大惩罚越轻）
        # 超指数放大差异：对于显著p值，权重呈指数级增长
        w_base = np.exp(10.0 / (-np.log(p_vals_safe) + 1e-10))
        # 应用 gamma 对比度调整（gamma>1放大差异）
        weights = w_base ** gamma
        # 应用指数锐化：p越大（越不显著），权重越小，惩罚越重
        if sharp_scale > 0:
            weights = weights * np.exp(-p_vals_safe * sharp_scale)
    elif weight_method == "correlation":
        stats = np.abs(univariate_results.get('correlations', np.ones(n_features)))
        # 相关性越大越显著，权重越大（和方案A匹配，w越大惩罚越轻）
        w_base = stats
        # 应用 gamma 对比度调整（gamma>1放大差异）
        weights = w_base ** gamma
    else:
        raise ValueError(f"Unknown weight method: {weight_method}")

    # 如果指定了最大缩放比例，进行归一化
    if weight_max_scale is not None:
        min_w = np.min(weights)
        max_w = np.max(weights)
        if max_w == min_w:
            return np.ones(n_features)
        normalized = (weights - min_w) / (max_w - min_w)
        weights = 1.0 + normalized * (weight_max_scale - 1.0)

    return weights


@jit(nopython=True, parallel=True, cache=True)
def _parallel_corr_matrix(X: np.ndarray, min_var: float = 1e-8) -> np.ndarray:
    """
    Parallelized correlation matrix computation with low variance feature filtering.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    min_var : float
        Minimum variance threshold, features with variance below this will have
        correlation set to 0 to avoid numerical issues

    Returns
    -------
    corr_matrix : np.ndarray
        Correlation matrix of shape (n_features, n_features)
    """
    n_samples, n_features = X.shape
    corr_matrix = np.zeros((n_features, n_features), dtype=np.float64)

    # Precompute means and standard deviations
    means = np.zeros(n_features, dtype=np.float64)
    stds = np.zeros(n_features, dtype=np.float64)

    for j in prange(n_features):
        x = X[:, j]
        mean_x = np.mean(x)
        var_x = np.var(x)
        means[j] = mean_x
        if var_x < min_var:
            stds[j] = 0.0
        else:
            stds[j] = np.sqrt(var_x)

    # Compute correlation matrix in parallel (upper triangle only)
    for i in prange(n_features):
        if stds[i] == 0.0:
            continue
        x_centered = X[:, i] - means[i]
        for j in range(i, n_features):
            if stds[j] == 0.0:
                corr_matrix[i, j] = 0.0
                corr_matrix[j, i] = 0.0
                continue
            y_centered = X[:, j] - means[j]
            cov = np.dot(x_centered, y_centered) / n_samples
            corr = cov / (stds[i] * stds[j])
            # Clip to [-1, 1] to avoid numerical errors
            corr = max(min(corr, 1.0), -1.0)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix


def _greedy_correlation_grouping(
    corr_matrix: np.ndarray,
    corr_threshold: float = 0.7,
    max_group_size: int = 20
) -> List[List[int]]:
    """
    Greedy non-overlapping correlation grouping of features.

    Parameters
    ----------
    corr_matrix : np.ndarray
        Absolute correlation matrix of shape (n_features, n_features)
    corr_threshold : float
        Threshold for grouping correlated features
    max_group_size : int
        Maximum size of each group

    Returns
    -------
    groups : List[List[int]]
        List of feature groups, each group is a list of feature indices
    """
    n_features = corr_matrix.shape[0]
    assigned = np.zeros(n_features, dtype=bool)
    groups = []

    # Compute feature importance (sum of absolute correlations) to prioritize grouping
    feature_importance = np.sum(np.abs(corr_matrix), axis=1)

    while not np.all(assigned):
        # Pick the most important unassigned feature as group leader
        unassigned_idx = np.where(~assigned)[0]
        leader = unassigned_idx[np.argmax(feature_importance[unassigned_idx])]

        # Find all unassigned features correlated with leader
        correlated = np.where(
            (~assigned) & (np.abs(corr_matrix[leader]) >= corr_threshold)
        )[0]

        # Limit group size, keep most correlated first
        if len(correlated) > max_group_size:
            # Sort by correlation with leader
            corr_values = corr_matrix[leader, correlated]
            sorted_idx = np.argsort(-np.abs(corr_values))
            correlated = correlated[sorted_idx[:max_group_size]]

        # Add group
        groups.append(correlated.tolist())
        assigned[correlated] = True

    return groups




import numpy as np
from typing import Optional, Callable
import adelie as ad

class UniLassoResultBase:
    """
    Base class for UniLasso results, encapsulating model outputs.
    """

    def __init__(self,
                 coefs: np.ndarray,
                 intercept: np.ndarray,
                 family: str,
                 gamma: np.ndarray,
                 gamma_intercept: np.ndarray,
                 beta: np.ndarray,
                 beta_intercepts: np.ndarray,
                 lasso_model: ad.grpnet,
                 lmdas: np.ndarray):
        """
        Initializes the base UniLasso result object.

        Parameters:
        - coefs (np.ndarray): Coefficients of the univariate-guided lasso.
        - intercept (np.ndarray): Intercept of the univariate-guided lasso.
        - family (str): Family of the response variable ('gaussian', 'binomial', 'cox').
        - gamma (np.ndarray): Hidden gamma coefficients.
        - gamma_intercept (np.ndarray): Hidden gamma intercept.
        - beta (np.ndarray): Hidden beta coefficients.
        - beta_intercepts (np.ndarray): Hidden beta intercepts.
        - lasso_model (ad.grpnet): The fitted Lasso model.
        - lmdas (np.ndarray): Regularization path.
        """
        self.coefs = coefs
        self.intercept = intercept
        self.family = family
        self._gamma = gamma
        self._gamma_intercept = gamma_intercept
        self._beta = beta
        self._beta_intercepts = beta_intercepts
        self.lasso_model = lasso_model
        self.lmdas = lmdas
        # 新增：分组信息（可选）
        self.groups = None
        self.group_signs = None

    def get_gamma(self) -> np.ndarray:
        """Returns the hidden gamma coefficients."""
        return self._gamma

    def get_gamma_intercept(self) -> np.ndarray:
        """Returns the hidden gamma intercept."""
        return self._gamma_intercept

    def get_beta(self) -> np.ndarray:
        """Returns the hidden beta coefficients."""
        return self._beta

    def get_beta_intercepts(self) -> np.ndarray:
        """Returns the hidden beta intercepts."""
        return self._beta_intercepts

    def __repr__(self):
        """Custom string representation of the result object."""
        return (f"{self.__class__.__name__}(coefs={self.coefs.shape}, "
                f"intercept={self.intercept.shape}, "
                f"lasso_model={type(self.lasso_model).__name__}, "
                f"lmdas={self.lmdas.shape})")


class UniLassoResult(UniLassoResultBase):
    """
    Class for storing standard UniLasso results.
    """
    pass


class UniLassoCVResult(UniLassoResultBase):
    """
    Class for storing cross-validation UniLasso results.
    """

    def __init__(self, 
                 coefs: np.ndarray, 
                 intercept: np.ndarray, 
                 family: str,
                 gamma: np.ndarray, 
                 gamma_intercept: np.ndarray, 
                 beta: np.ndarray, 
                 beta_intercepts: np.ndarray, 
                 lasso_model: ad.grpnet, 
                 lmdas: np.ndarray,
                 avg_losses: np.ndarray, 
                 cv_plot: Optional[Callable] = None, 
                 best_idx: Optional[int] = None, 
                 best_lmda: Optional[float] = None):
        """
        Initializes the cross-validation result object.

        Additional Parameters:
        - avg_losses (np.ndarray): Average cross-validation losses.
        - cv_plot (Optional[Callable]): Function to generate cross-validation plot.
        - best_idx (Optional[int]): Index of the best-performing regularization parameter.
        - best_lmda (Optional[float]): Best regularization parameter.
        """
        super().__init__(coefs, intercept, family, gamma, gamma_intercept, beta, beta_intercepts, lasso_model, lmdas)
        self.avg_losses = avg_losses
        self.cv_plot = cv_plot
        self.best_idx = best_idx
        self.best_lmda = best_lmda

    def __repr__(self):
        base_repr = super().__repr__()
        return (f"{base_repr}, best_lmda={self.best_lmda}, "
                f"best_idx={self.best_idx}, avg_losses={self.avg_losses.shape})")
    


@jit(nopython=True, cache=True)
def _fit_univariate_regression_gaussian_numba(
            X: np.ndarray, 
            y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit univariate Gaussian regression for each feature in X."""
    n, p = X.shape
    beta_intercepts = np.zeros(p)
    beta_coefs = np.zeros(p)

    for j in range(p):
        xj = np.expand_dims(X[:, j], axis=1)
        xj_mean = np.mean(xj)
        y_mean = np.mean(y)
        sxy = np.sum(xj[:, 0] * y) - n * xj_mean * y_mean
        sxx = np.sum(xj[:, 0] ** 2) - n * xj_mean ** 2
        slope = sxy / sxx
        beta_intercepts[j] = y_mean - slope * xj_mean
        beta_coefs[j] = slope

    return beta_intercepts, beta_coefs


def fit_univariate_regression(
            X: np.ndarray,
            y: np.ndarray,
            family: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit univariate regression model for each feature in X.

    Args:
        X: Feature matrix of shape (n, p).
        y: Target vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'multinomial', 'poisson', 'cox').

    Returns:
        Tuple containing:
            - beta_intercepts: Intercepts of the regression model.
            - beta_coefs: Coefficients of the regression model.
    """
    n, p = X.shape

    if family == "gaussian":
        beta_intercepts, beta_coefs = _fit_univariate_regression_gaussian_numba(X, y)
    elif family in {"binomial", "cox"}:
        if family == "binomial":
            glm_y = ad.glm.binomial(y)
        elif family == "cox":
            glm_y = ad.glm.cox(start=np.zeros(n), stop=y[:, 0], status=y[:, 1])

        beta_intercepts = np.zeros(p)
        beta_coefs = np.zeros(p)

        for j in range(p):
            if family == "binomial":
                X_j = np.column_stack([np.ones(n), X[:, j]])
            else:
                # Cox model requires no intercept term
                X_j = np.column_stack([np.zeros(n), X[:, j]])
            X_j = np.asfortranarray(X_j)
            glm_fit = ad.grpnet(X_j,
                                glm_y,
                                intercept=False,
                                lmda_path=[0.0])
            coefs = glm_fit.betas.toarray()

            if family == "binomial":
                beta_intercepts[j] = coefs[0][0]

            beta_coefs[j] = coefs[0][1]
    elif family in {"poisson", "multinomial"}:
        # For Poisson and Multinomial: use simple estimation
        beta_intercepts = np.zeros(p)
        beta_coefs = np.zeros(p)

        # Use Gaussian as a simple fallback for quick estimation
        # This is used for feature importance estimation
        for j in range(p):
            xj = X[:, j]
            # Simple correlation-based beta for Poisson
            if family == "poisson":
                y_mean = np.mean(y)
                x_mean = np.mean(xj)
                cov = np.mean((xj - x_mean) * (y - y_mean))
                var_x = np.mean((xj - x_mean)**2)
                if var_x > 1e-10:
                    beta_coefs[j] = cov / var_x
                beta_intercepts[j] = np.log(y_mean + 1e-10) - beta_coefs[j] * x_mean
            else:  # multinomial
                # For multinomial, use one-vs-rest approach
                y_bin = (y != np.min(y)).astype(float)
                y_mean = np.mean(y_bin)
                x_mean = np.mean(xj)
                cov = np.mean((xj - x_mean) * (y_bin - y_mean))
                var_x = np.mean((xj - x_mean)**2)
                if var_x > 1e-10:
                    beta_coefs[j] = cov / var_x
                beta_intercepts[j] = y_mean - beta_coefs[j] * x_mean
    else:
        raise ValueError(f"Unsupported family type: {family}")

    return beta_intercepts, beta_coefs


def fit_univariate_models(
            X: np.ndarray,
            y: np.ndarray,
            family: str = "gaussian",
            univariate_model: str = "linear",
            spline_df: int = 5,
            spline_degree: int = 3,
            tree_max_depth: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit univariate least squares regression for each feature in X and compute
    leave-one-out (LOO) fitted values.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'multinomial', 'poisson', 'cox').
        univariate_model: Type of univariate model to use: 'linear', 'spline', or 'tree'.
        spline_df: Degrees of freedom for spline regression (if univariate_model="spline").
        spline_degree: Degree of polynomial for spline regression (default: 3 for cubic).
        tree_max_depth: Maximum depth for decision tree regression (if univariate_model="tree").

    Returns:
        Tuple containing:
            - loo_fits: Leave-one-out fitted values.
            - beta_intercepts: Intercepts from univariate regressions.
            - beta_coefs: Slopes from univariate regressions.
    """
    # Ensure y is float for all families
    y_float = np.asarray(y, dtype=float)

    # For nonlinear models, use the model-specific LOO fits directly
    if univariate_model != "linear":
        loo_result = fit_loo_univariate_models(
            X, y_float,
            family=family,
            univariate_model=univariate_model,
            spline_df=spline_df,
            spline_degree=spline_degree,
            tree_max_depth=tree_max_depth
        )
        loo_fits = loo_result["fit"]
        beta_coefs = loo_result["beta"]
        beta_intercepts = loo_result["beta0"]
        return loo_fits, beta_intercepts, beta_coefs

    # Linear model - original behavior
    beta_intercepts, beta_coefs = fit_univariate_regression(X, y_float, family)

    # For Poisson and Multinomial: use Gaussian LOO as approximation
    if family in {"poisson", "multinomial"}:
        loo_fits = fit_loo_univariate_models(X, y_float, family="gaussian")["fit"]
    else:
        loo_fits = fit_loo_univariate_models(X, y_float, family=family)["fit"]

    return loo_fits, beta_intercepts, beta_coefs


def _format_unilasso_feature_matrix(X: np.ndarray,
                                    remove_zero_var: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Format and validate feature matrix for UniLasso."""

    X = np.array(X, dtype=float)
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values.")

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError("X must be a 1D or 2D NumPy array.")

    if remove_zero_var:
        zero_var_idx = np.where(np.var(X, axis=0) == 0)[0]
        if len(zero_var_idx) > 0:
            warn_zero_variance(len(zero_var_idx), X.shape[1])
            X = np.delete(X, zero_var_idx, axis=1)
            if X.shape[1] == 0:
                raise ValueError("All features have zero variance.")
    else:
        zero_var_idx = None
    
    return X, zero_var_idx



def _format_unilasso_input(
            X: np.ndarray, 
            y: np.ndarray, 
            family: str, 
            lmdas: Optional[Union[float, List[float], np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]]:
    """Format and validate input for UniLasso."""
    if family not in VALID_FAMILIES:
        raise ValueError(f"Family must be one of {VALID_FAMILIES}")
    
    X, zero_var_idx = _format_unilasso_feature_matrix(X, True)
    y = _format_y(y, family)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows (samples).")

    lmdas = _format_lmdas(lmdas)

    return X, y, family, lmdas, zero_var_idx


def _format_y(
        y: Union[np.ndarray, pd.DataFrame], 
        family: str) -> np.ndarray:
    """Format and validate y based on the family."""
    if family in {"gaussian", "binomial"}:
        y = np.array(y, dtype=float).flatten()
        if family == "binomial" and not np.all(np.isin(y, [0, 1])):
            raise ValueError("For `binomial` family, y must be binary with values 0 and 1.")
    elif family == "cox":
        if isinstance(y, (pd.DataFrame, dict)):
            if not 'time' in y.columns or not 'status' in y.columns:
                raise ValueError("For `cox` family, y must be a DataFrame with columns 'time' and 'status'.")
            y = np.column_stack((y["time"], y["status"]))
        if y.shape[1] != 2:
            raise ValueError("For `cox` family, y must have two columns corresponding to time and status.")
        if not np.all(y[:, 0] >= 0):
            raise ValueError("For `cox` family, time values must be nonnegative.")
        if not np.all(np.isin(y[:, 1], [0, 1])):
            raise ValueError("For `cox` family, status values must be binary with values 0 and 1.")
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")
    
    return y


def _format_lmdas(lmdas: Optional[Union[float, List[float], np.ndarray]]) -> Optional[np.ndarray]:
    """Format and validate lmdas."""
    if lmdas is None:
        return None
    if isinstance(lmdas, (float, int)):
        lmdas = [float(lmdas)]

    if not isinstance(lmdas, list) and not isinstance(lmdas, np.ndarray):
        raise ValueError("lmdas must be a nonnegative float, list of floats, or NumPy array of floats.")
    
    lmdas = np.array(lmdas, dtype=float)

    if np.any(np.isnan(lmdas)) or np.any(np.isinf(lmdas)):
        raise ValueError("Regularizers contain NaN or infinite values.")
    
    if np.any(lmdas < 0):
        raise ValueError("Regularizers must be nonnegative.")

    return lmdas


def _prepare_unilasso_input(
                X: np.ndarray,
                y: np.ndarray,
                family: str,
                lmdas: Optional[Union[float, List[float], np.ndarray]],
                univariate_model: str = "linear",
                spline_df: int = 5,
                spline_degree: int = 3,
                tree_max_depth: int = 2
) -> Tuple[np.ndarray,
           np.ndarray,
           np.ndarray,
           np.ndarray,
           np.ndarray,
           Optional[ad.glm.GlmBase64],
           Optional[List[ad.constraint.ConstraintBase64]],
           Optional[np.ndarray]]:
    """Prepare input for UniLasso."""
    X, y, family, lmdas, zero_var_idx = _format_unilasso_input(X, y, family, lmdas)

    loo_fits, beta_intercepts, beta_coefs_fit = fit_univariate_models(
        X, y,
        family=family,
        univariate_model=univariate_model,
        spline_df=spline_df,
        spline_degree=spline_degree,
        tree_max_depth=tree_max_depth
    )
    loo_fits = np.asfortranarray(loo_fits)

    # Only get glm_family and constraints for families supported by adelie
    # For poisson and multinomial, we use our custom solver and don't need these
    # For nonlinear models (spline/tree), we also skip adelie initialization
    if family in {"gaussian", "binomial", "cox"} and univariate_model == "linear":
        glm_family = _get_glm_family(family, y)
        constraints = [ad.constraint.lower(b=np.zeros(1)) for _ in range(X.shape[1])]
    else:
        glm_family = None
        constraints = None

    return X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, lmdas, zero_var_idx




def _get_glm_family(family: str, 
                    y: np.ndarray) -> ad.glm.GlmBase64:
    """Get the appropriate GLM family."""
    if family == "gaussian":
        return ad.glm.gaussian(y)
    elif family == "binomial":
        return ad.glm.binomial(y)
    elif family == "cox":
        return ad.glm.cox(start=np.zeros(len(y)), stop=y[:, 0], status=y[:, 1])
    else:
        raise ValueError(f"Unsupported family: {family}")



def _handle_zero_variance(
            gamma_hat_fit: np.ndarray,
            beta_coefs_fit: np.ndarray,
            zero_var_idx: Optional[np.ndarray],
            cur_num_var: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Handle zero variance features."""
    if zero_var_idx is not None and len(zero_var_idx) > 0:
        total_num_var = cur_num_var + len(zero_var_idx)
        num_regs = gamma_hat_fit.shape[0]
        gamma_hat = np.zeros((num_regs, total_num_var))
        beta_coefs = np.zeros((num_regs, total_num_var))
        pos_var_idx = np.setdiff1d(np.arange(total_num_var), zero_var_idx)
        gamma_hat[:, pos_var_idx] = gamma_hat_fit
        beta_coefs[:, pos_var_idx] = beta_coefs_fit
    else:
        gamma_hat = gamma_hat_fit
        beta_coefs = beta_coefs_fit
    return gamma_hat, beta_coefs



def _print_unilasso_results(
            gamma_hat: np.ndarray, 
            lmdas: np.ndarray, 
            best_idx: Optional[int] = None
) -> None:
    """Print UniLasso results."""

    if gamma_hat.ndim == 1:
        num_selected = np.sum(gamma_hat != 0)
    else:
        num_selected = np.sum(gamma_hat != 0, axis=1)

    # check if interactive environment
    try:
        get_ipython()

        from IPython.core.display import display, HTML
        display(HTML("\n\n<b> --- UniLasso Results --- </b>"))
    except NameError:
        print("\n\n\033[1m --- UniLasso Results --- \033[0m")

    print(f"Number of Selected Features: {num_selected}")
    print(f"Regularization path (rounded to 3 decimal places): {np.round(lmdas, 3)}")
    if best_idx is not None:
        print(f"Best Regularization Parameter: {lmdas[best_idx]}")



def _format_output(lasso_model: ad.grpnet,
                   beta_coefs_fit: np.ndarray,
                   beta_intercepts: np.ndarray,
                   zero_var_idx: Optional[np.ndarray],
                   X: np.ndarray,
                   fit_intercept: bool,
                   reverse_indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Format UniLasso output."""
    theta_hat = lasso_model.betas.toarray()
    theta_0 = lasso_model.intercepts

    beta_coefs_fit = beta_coefs_fit.squeeze()
    beta_intercepts = beta_intercepts.squeeze()

    if reverse_indices is not None:
        theta_hat = theta_hat[reverse_indices]
        theta_0 = theta_0[reverse_indices]


    gamma_hat_fit = theta_hat * beta_coefs_fit
    gamma_hat, beta_coefs = _handle_zero_variance(gamma_hat_fit, beta_coefs_fit, zero_var_idx, X.shape[1])
    gamma_hat = gamma_hat.squeeze()
    beta_coefs = beta_coefs.squeeze()

    if fit_intercept:
        gamma_0 = theta_0 + np.sum(theta_hat * beta_intercepts, axis=1)
        gamma_0 = gamma_0.squeeze()
    else:
        gamma_0 = np.zeros(len(theta_0))
   
    return gamma_hat, gamma_0, beta_coefs





def _configure_lmda_min_ratio(n: int,
                              p: int) -> np.ndarray:
    """Configure lambda min ratio for UniLasso."""
    return 0.01 if n < p else 1e-4




def _check_lmda_min_ratio(lmda_min_ratio: float) -> float:
    """Check lambda min ratio for UniLasso."""
    if lmda_min_ratio <= 0:
        raise ValueError("Minimum regularization ratio must be positive.")
    if lmda_min_ratio > 1:
        raise ValueError("Minimum regularization ratio must be less than 1.")
    return lmda_min_ratio
    

def _configure_lmda_path(X: np.ndarray, 
                         y: np.ndarray,
                         family: str,
                         n_lmdas: Optional[int], 
                         lmda_min_ratio: Optional[float]) -> np.ndarray:
    """Configure the regularization path for UniLasso."""

    n, p = X.shape
    if n_lmdas is None:
        n_lmdas = 100
    
    if lmda_min_ratio is None:
        lmda_min_ratio = _configure_lmda_min_ratio(n, p)

    assert n_lmdas > 0, "Number of regularization parameters must be positive."
    _check_lmda_min_ratio(lmda_min_ratio)
    
    if family == "cox":
        y = y[:, 0]

    # Define function to standardize columns using n (not n-1)
    def moment_sd(z):
        return np.sqrt(np.sum((z - np.mean(z))**2) / len(z))

    X_standardized = (X - np.mean(X, axis=0)) / np.apply_along_axis(moment_sd, 0, X)
    X_standardized = np.array(X_standardized)  

    # Standardize y (centering only)
    y = y - np.mean(y)

    n = X_standardized.shape[0]
    lambda_max = np.max(np.abs(X_standardized.T @ y)) / n

    lambda_path = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_max * lmda_min_ratio), n_lmdas))

    return lambda_path





# ------------------------------------------------------------------------------
# Perform cross-validation UniLasso
# ------------------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt

def plot(unilasso_fit) -> None:
    """
    Plots the Lasso coefficient paths as a function of the regularization parameter (lambda),
    with the number of active (nonzero) coefficients labeled at the top.

    Parameters:
    - unilasso_fit: UniLassoResult object containing fitted coefficients and lambda values.
    """
    
    assert hasattr(unilasso_fit, "coefs") and hasattr(unilasso_fit, "lmdas"), \
        "Input must have 'coefs' and 'lmdas' attributes."

    coefs, lambdas = unilasso_fit.coefs, unilasso_fit.lmdas
    if coefs.ndim == 1 or len(lambdas) == 1:
        print("Only one regularization parameter was used. No path to plot.")
        return
    
    plt.figure(figsize=(8, 6))
    neg_log_lambdas = -np.log(lambdas)  # lambdas are already in descending order

    # Compute the number of nonzero coefficients at each lambda
    n_nonzero = np.sum(coefs != 0, axis=1)

    # Plot coefficient paths
    for i in range(coefs.shape[1]):  
        plt.plot(neg_log_lambdas, coefs[:, i], lw=2)

    # Labels and formatting
    plt.xlabel(r"$-\log(\lambda)$", fontsize=12)
    plt.ylabel("Coefficients", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Add secondary x-axis for the number of active coefficients
    ax1 = plt.gca()  
    ax2 = ax1.twiny()  
    ax2.set_xlim(ax1.get_xlim())  
    # Dynamic tick calculation for better readability
    tick_indices = np.linspace(0, len(neg_log_lambdas) - 1, min(6, len(neg_log_lambdas)), dtype=int)
    ax2.set_xticks(neg_log_lambdas[tick_indices])  
    ax2.set_xticklabels(n_nonzero[tick_indices]) 
    ax2.set_xlabel("Number of Active Coefficients", fontsize=12)

    plt.show()


def plot_cv(cv_result: UniLassoCVResult) -> None:
    """
    Plots the cross-validation
    curve as a function of the regularization parameter (lambda).
    """
    cv_result.cv_plot()



def extract_cv(cv_result: UniLassoCVResult) -> UniLassoResult:
    """
    Extract the best coefficients and intercept from a cross-validated UniLasso result.

    Args:
        - cv_result: UniLassoCVResult object.
    
    Returns:
        - UniLassoResult object with the best coefficients and intercept.
    """

    best_coef = cv_result.coefs[cv_result.best_idx].squeeze()
    best_intercept = cv_result.intercept[cv_result.best_idx].squeeze()

    extracted_fit = UniLassoResult(
        coefs=best_coef,
        intercept=best_intercept,
        family=cv_result.family,
        gamma=best_coef,
        gamma_intercept=best_intercept,
        beta=cv_result._beta,
        beta_intercepts=cv_result._beta_intercepts,
        lasso_model=cv_result.lasso_model,
        lmdas=cv_result.lmdas
    )

    return extracted_fit



def cv_unilasso(
            X: np.ndarray,
            y: np.ndarray,
            family: str = "gaussian",
            n_folds: int = 5,
            lmda_min_ratio: float = None,
            verbose: bool = False,
            seed: Optional[int] = None
) ->  UniLassoCVResult:
    """
    Perform cross-validation UniLasso.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'cox').
        n_folds: Number of cross-validation folds.
        lmda_min_ratio: Minimum ratio of the largest to smallest regularization parameter.
        verbose: Whether to print results.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing UniLasso results.
    """
    if lmda_min_ratio is None:
        lmda_min_ratio = _configure_lmda_min_ratio(X.shape[0], X.shape[1])

    assert n_folds > 1, "Number of folds must be greater than 1."
    _check_lmda_min_ratio(lmda_min_ratio)

    X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, _, zero_var_idx = _prepare_unilasso_input(X, y, family, None)
    fit_intercept = False if family == "cox" else True

    cv_lasso = ad.cv_grpnet(
        X=loo_fits,
        glm=glm_family,
        seed=seed,
        n_folds=n_folds,
        groups=None,
        min_ratio=lmda_min_ratio,
        intercept=fit_intercept,
        constraints=constraints,
        tol=1e-7
    )

    # refit lasso along a regularization path that stops at the best chosen lambda
    lasso_model = cv_lasso.fit(
        X=loo_fits,
        glm=glm_family,
        groups=None,
        intercept=fit_intercept,
        constraints=constraints,
    )

    gamma_hat, gamma_0, beta_coefs = _format_output(lasso_model,
                                                    beta_coefs_fit,
                                                    beta_intercepts,
                                                    zero_var_idx,
                                                    X,
                                                    fit_intercept)

    

    cv_plot = cv_lasso.plot_loss
    if verbose:
        _print_unilasso_results(gamma_hat, cv_lasso.lmdas, int(cv_lasso.best_idx))
        cv_plot()
    

    unilasso_result = UniLassoCVResult(
        coefs=gamma_hat,
        intercept=gamma_0,
        family=family,
        gamma=gamma_hat,
        gamma_intercept=gamma_0,
        beta=beta_coefs,
        beta_intercepts=beta_intercepts,
        lasso_model=lasso_model,
        lmdas=cv_lasso.lmdas,
        avg_losses=cv_lasso.avg_losses,
        cv_plot=cv_plot,
        best_idx=int(cv_lasso.best_idx),
        best_lmda=cv_lasso.lmdas[cv_lasso.best_idx]
    )

    return unilasso_result


# ------------------------------------------------------------------------------
# UniLasso with external CV splits (for benchmark compatibility)
# ------------------------------------------------------------------------------
def cv_unilasso_with_splits(
            X: np.ndarray,
            y: np.ndarray,
            cv_splits: list,
            family: str = "gaussian",
            lmda_min_ratio: float = None,
            verbose: bool = False,
            seed: Optional[int] = None
) -> UniLassoCVResult:
    """
    Perform cross-validation UniLasso with EXTERNAL CV splits.

    This version allows external control of CV splits for fair comparison in benchmarks.
    The LOO fits are computed on the full training data (X), then Lasso CV is performed
    using the provided cv_splits.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        cv_splits: List of (train_idx, val_idx) tuples for cross-validation.
        family: Family of the response variable ('gaussian', 'binomial', 'cox').
        lmda_min_ratio: Minimum ratio of the largest to smallest regularization parameter.
        verbose: Whether to print results.
        seed: Random seed (used for refit).

    Returns:
        UniLassoCVResult containing cross-validation results.
    """
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler

    if lmda_min_ratio is None:
        lmda_min_ratio = _configure_lmda_min_ratio(X.shape[0], X.shape[1])

    _check_lmda_min_ratio(lmda_min_ratio)

    # Step 1: Prepare input and compute LOO fits
    X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, _, zero_var_idx = _prepare_unilasso_input(X, y, family, None)
    fit_intercept = False if family == "cox" else True

    n_folds = len(cv_splits)

    # Step 2: Manual CV using sklearn LassoCV with provided splits
    # Generate alpha path
    alpha_max = np.max(np.abs(loo_fits.T @ y)) / len(y)
    alpha_min = alpha_max * lmda_min_ratio
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), 100)[::-1]

    # Standardize features for CV
    scaler = StandardScaler()
    loo_fits_scaled = scaler.fit_transform(loo_fits)

    # Manual CV with provided splits
    mse_path = np.full((len(alphas), n_folds), np.inf)
    n_nonzero_path = np.zeros((len(alphas), n_folds))

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_tr = loo_fits_scaled[train_idx]
        X_va = loo_fits_scaled[val_idx]
        y_tr = y[train_idx]
        y_va = y[val_idx]

        # Fit lasso path on training fold
        lasso_path_model = LassoCV(
            alphas=alphas,
            cv=None,  # We'll do manual CV
            fit_intercept=False,
            max_iter=5000,
            tol=1e-7,
            random_state=seed,
            precompute=False
        )
        lasso_path_model.fit(X_tr, y_tr)

        # Predict on validation fold for all alphas
        # We need to manually compute MSE for each alpha
        _, coefs_path, _ = Lasso(
            alpha=alphas[0],  # placeholder
            fit_intercept=False,
            max_iter=1,  # just to get the path
            warm_start=True
        ).path(X_tr, y_tr, alphas=alphas)

        # Actually, let's use a simpler approach - fit on full path
        lasso_refit = Lasso(fit_intercept=False, max_iter=5000, tol=1e-7, random_state=seed)
        lasso_refit.fit(X_tr, y_tr)

        # Get predictions for all alphas using the path
        # For simplicity, let's just compute MSE for each alpha manually
        for alpha_idx, alpha in enumerate(alphas):
            lasso_tmp = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000, tol=1e-7, random_state=seed)
            lasso_tmp.fit(X_tr, y_tr)
            y_pred = lasso_tmp.predict(X_va)
            mse_path[alpha_idx, fold_idx] = np.mean((y_va - y_pred) ** 2)
            n_nonzero_path[alpha_idx, fold_idx] = np.sum(lasso_tmp.coef_ != 0)

    # Compute mean MSE across folds
    mean_mse = np.mean(mse_path, axis=1)
    std_mse = np.std(mse_path, axis=1) / np.sqrt(n_folds)

    # Find best alpha (min MSE)
    best_alpha_idx = np.argmin(mean_mse)
    best_alpha = alphas[best_alpha_idx]

    # 1-SE rule: select most sparse model within 1-SE of best
    threshold = mean_mse[best_alpha_idx] + std_mse[best_alpha_idx]
    candidates_mask = mean_mse <= threshold
    if np.any(candidates_mask):
        # Among candidates, select most sparse (fewest non-zeros)
        masked_nonzero = np.where(candidates_mask, np.mean(n_nonzero_path, axis=1), np.inf)
        best_alpha_idx = np.argmin(masked_nonzero)
        best_alpha = alphas[best_alpha_idx]

    if verbose:
        print(f"[cv_unilasso_with_splits] Best alpha: {best_alpha:.6f}, MSE: {mean_mse[best_alpha_idx]:.6f}")

    # Step 3: Refit on full data with best alpha
    scaler_final = StandardScaler()
    loo_fits_final = scaler_final.fit_transform(loo_fits)

    lasso_final = Lasso(alpha=best_alpha, fit_intercept=fit_intercept, max_iter=5000, tol=1e-7, random_state=seed)
    lasso_final.fit(loo_fits_final, y)

    # Get coefficients in original scale
    gamma_hat = lasso_final.coef_ / scaler_final.scale_
    gamma_0 = lasso_final.intercept_ if fit_intercept else 0.0

    # Adjust intercept for standardization
    if fit_intercept:
        gamma_0 = gamma_0 - np.sum(scaler_final.mean_ * gamma_hat)

    # Format output
    gamma_hat, gamma_0, beta_coefs = _format_output(
        lasso_final, beta_coefs_fit, beta_intercepts, zero_var_idx, X, fit_intercept
    )

    unilasso_result = UniLassoCVResult(
        coefs=gamma_hat,
        intercept=gamma_0,
        family=family,
        gamma=gamma_hat,
        gamma_intercept=gamma_0,
        beta=beta_coefs,
        beta_intercepts=beta_intercepts,
        lasso_model=lasso_final,
        lmdas=alphas,
        avg_losses=mean_mse,
        cv_plot=None,
        best_idx=int(best_alpha_idx),
        best_lmda=best_alpha
    )

    return unilasso_result


# ------------------------------------------------------------------------------
# Fit UniLasso for a specified regularization path
# ------------------------------------------------------------------------------
def fit_unilasso(
            X: np.ndarray,
            y: np.ndarray,
            family: str = "gaussian",
            lmdas: Optional[Union[float, List[float], np.ndarray]] = None,
            n_lmdas: Optional[int] = 100,
            lmda_min_ratio: Optional[float] = 1e-2,
            verbose: bool = False
) -> UniLassoResult:
    """
    Perform UniLasso with specified regularization parameters.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'cox').
        lmdas: Lasso regularization parameter(s).
        n_lmdas: Number of regularization parameters to use if `lmdas` is None.
        lmda_min_ratio: Minimum ratio of the largest to smallest regularization parameter. 
        verbose: Whether to print results.

    Returns:
        Dictionary containing UniLasso results.
    """
    X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, lmdas, zero_var_idx = _prepare_unilasso_input(X, y, family, lmdas)

    fit_intercept = False if family == "cox" else True


    lasso_model = ad.grpnet(
        X=loo_fits,
        glm=glm_family,
        groups=None,
        intercept=fit_intercept,
        lmda_path=lmdas, # Regularization path, if unspecified, will be generated
        constraints=constraints,
        lmda_path_size=n_lmdas,
        min_ratio=lmda_min_ratio,
        tol=1e-7
    )

    glm_lmdas = np.array(lasso_model.lmdas)

    if lmdas is not None:
        if not np.all(np.isin(lmdas, glm_lmdas)):
            removed_lmdas = np.setdiff1d(lmdas, glm_lmdas)
            removed_lmdas = np.round(removed_lmdas, 3)
            warn_removed_lmdas(removed_lmdas)

        matching_idx = np.where(np.isin(lmdas, glm_lmdas))[0]
        lmdas = lmdas[matching_idx]
    else:
        lmdas = glm_lmdas

    if len(lmdas) == 0:
        raise ValueError("No regularization strengths remain after removing invalid values")

    reverse_indices = np.arange(len(glm_lmdas))
    reverse_indices = reverse_indices[::-1]

    # Apply the same reversal to lmdas to maintain correspondence with coefs
    lmdas = lmdas[reverse_indices]

    gamma_hat, gamma_0, beta_coefs = _format_output(lasso_model,
                                                    beta_coefs_fit,
                                                    beta_intercepts,
                                                    zero_var_idx,
                                                    X,
                                                    fit_intercept,
                                                    reverse_indices)

    if verbose:
        _print_unilasso_results(gamma_hat, lmdas)

    unilasso_result = UniLassoResult(
        coefs=gamma_hat,
        intercept=gamma_0,
        family=family,
        gamma=gamma_hat,
        gamma_intercept=gamma_0,
        beta=beta_coefs,
        beta_intercepts=beta_intercepts,
        lasso_model=lasso_model,
        lmdas=lmdas
    )

    return unilasso_result


def predict(result: UniLassoResult,
            X: np.ndarray, 
            lmda_idx: Optional[int] = None) -> np.ndarray:
    """
    Predict response variable using UniLasso model.

    Args:
        result: UniLasso result object.
        X: Feature matrix of shape (n, p).
        lmda_idx: Index of the regularization parameter to use for prediction.

    Returns:
        Predicted response variable.
    """

    if not type(result) == UniLassoResult:
        raise ValueError("`result` must be a UniLassoResult object.")
    
    if len(result.coefs.shape) == 1:
        result.coefs = np.expand_dims(result.coefs, axis=0)
    assert result.coefs.shape[1] == X.shape[1], "Feature matrix must have the same number of columns as the fitted model."

    X, _ = _format_unilasso_feature_matrix(X, remove_zero_var=False)
    
    if lmda_idx is not None:
        assert lmda_idx >= 0 and lmda_idx < len(result.lmdas), "Invalid regularization parameter index."
        y_hat = X @ result.coefs[lmda_idx] + result.intercept[lmda_idx]
    else:
        y_hat = X @ result.coefs.T + result.intercept 

    y_hat = y_hat.squeeze()
          
    return y_hat




import numpy as np
import torch
from typing import Optional, Callable
# 假设已经导入了 _format_output, _prepare_unilasso_input, UniLassoCVResult 等原有函数

class PyTorchGrpnetAdapter:
    """
    严肃工程实现：适配器模式。
    作用：将 PyTorch 计算出的权重矩阵和截距数组，包装成类似 ad.grpnet 的对象接口。
    """
    def __init__(self, betas_matrix: np.ndarray, intercepts_array: np.ndarray, lmdas: np.ndarray):
        # 参数 betas_matrix: 形状为 (n_lmdas, n_features) 的特征系数矩阵
        # 参数 intercepts_array: 形状为 (n_lmdas,) 的截距数组
        self._betas = betas_matrix
        self.intercepts = intercepts_array
        self.lmdas = lmdas

    @property
    def betas(self):
        """
        模拟 adelie 中稀疏矩阵的 .toarray() 方法。
        返回一个具有 toarray 方法的内部匿名类，巧妙满足 _format_output 的调用链。
        """
        class MockSparseMatrix:
            def __init__(self, mat):
                self.mat = mat
            def toarray(self):
                return self.mat
                
        return MockSparseMatrix(self._betas)
    
    
def _fit_pytorch_lasso_path(
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
    沿着正则化路径 (lambda_path) 训练模型，利用 Warm Start 加速收敛。
    CUDA-optimized implementation with vectorized operations.

    使用双层非对称软阈值算子 (Double Asymmetric Soft-Thresholding)，支持：
    1. 特征级自适应权重惩罚
    2. 组级符号一致性约束

    Parameters
    ----------
    alpha : float
        XLasso parameter: penalty coefficient for the $\frac{1}{w_j}$ term
        For significant variables (small w_j), this increases penalty on negative coefficients
    beta : float
        XLasso parameter: penalty coefficient for the $w_j$ term
        For insignificant variables (large w_j), this increases penalty on both signs
    negative_penalty : float
        Backward compatibility: Additional penalty strength for negative coefficients
        (used when alpha=0, this reduces to the old simplified form)
    feature_weights : np.ndarray, optional
        Feature-level significance weights w_j, default all ones
    group_signs : np.ndarray, optional
        Dominant sign for each feature's group, default all ones
    group_penalty : float
        Global group penalty strength for sign inconsistency
    group_weights : np.ndarray, optional
        Group-level penalty weights, default all ones
    """
    # Auto-detect CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move all tensors to GPU if available
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)

    n_features = X_train.shape[1]
    n_lmdas = len(lmdas)
    n_samples = X_train.shape[0]

    # Default values for optional parameters
    if feature_weights is None:
        feature_weights = np.ones(n_features)
    if group_signs is None:
        group_signs = np.ones(n_features)
    if group_weights is None:
        group_weights = np.ones(n_features)

    # Convert to torch tensors (on device)
    fw_t = torch.tensor(feature_weights, dtype=torch.float32, device=device)
    gs_t = torch.tensor(group_signs, dtype=torch.float32, device=device)
    gw_t = torch.tensor(group_weights, dtype=torch.float32, device=device)

    # 预分配结果存储空间
    betas_matrix = np.zeros((n_lmdas, n_features))
    intercepts = np.zeros(n_lmdas)

    # 初始化可训练参数 (on device)
    weights = torch.zeros((n_features, 1), dtype=torch.float32, device=device, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)

    # Precompute matrices for Gaussian family to accelerate gradient computation
    precomputed = {}
    if family == "gaussian":
        XtX = torch.matmul(X_t.T, X_t) / n_samples
        Xty = torch.matmul(X_t.T, y_t) / n_samples
        y_mean = torch.mean(y_t)
        X_mean = torch.mean(X_t, dim=0)
        precomputed = {"XtX": XtX, "Xty": Xty, "y_mean": y_mean, "X_mean": X_mean}

    # Use Nesterov only when momentum > 0
    use_nesterov = momentum > 0
    optimizer = torch.optim.SGD([weights, bias], lr=lr, momentum=momentum, nesterov=use_nesterov)

    for i, lmda in enumerate(lmdas):
        # Reset momentum when lambda changes (warm start) if momentum is enabled
        if momentum > 0:
            optimizer.param_groups[0]['momentum'] = 0.5
            for param_group in optimizer.param_groups:
                param_group['momentum_buffer'] = None

        # For early stopping: track last 3 changes
        last_changes = torch.zeros(3, device=device)
        change_idx = 0

        # Precompute thresholds for all features (vectorized)
        fw_safe = torch.clamp(fw_t, min=1e-10)
        tau_pos_base = lr * lmda * fw_t
        tau_neg_base = lr * lmda * (alpha / fw_safe + beta * fw_t)

        if negative_penalty > 0:
            tau_neg_base += lr * negative_penalty * fw_t

        # Group penalty adjustment (vectorized)
        group_penalty_t = lr * group_penalty * gw_t
        tau_pos = torch.where(gs_t > 0, tau_pos_base, tau_pos_base + group_penalty_t)
        tau_neg = torch.where(gs_t > 0, tau_neg_base + group_penalty_t, tau_neg_base)

        # Reshape for broadcasting
        tau_pos = tau_pos.view(-1, 1)
        tau_neg = tau_neg.view(-1, 1)

        # 针对当前的 lambda 进行梯度下降更新 (利用了上一轮的 weights 作为起点)
        for epoch in range(max_epochs):
            optimizer.zero_grad()

            if family == "gaussian" and precomputed:
                # Fast gradient computation using precomputed matrices
                weights_lookahead = weights + momentum * optimizer.state[weights].get('momentum_buffer', 0) if momentum > 0 else weights
                bias_lookahead = bias + momentum * optimizer.state[bias].get('momentum_buffer', 0) if momentum > 0 else bias

                grad_weights = torch.matmul(precomputed["XtX"], weights_lookahead) + bias_lookahead * precomputed["X_mean"].view(-1, 1) - precomputed["Xty"]
                grad_bias = torch.dot(precomputed["X_mean"], weights_lookahead.flatten()) + bias_lookahead - precomputed["y_mean"]

                # Manually set gradients
                weights.grad = grad_weights
                bias.grad = grad_bias
            else:
                # Standard forward pass for other families
                y_pred = torch.matmul(X_t, weights) + bias

                if family == "binomial" or family == "multinomial":
                    # Binomial/Multinomial: Logistic loss
                    mu = torch.sigmoid(torch.clamp(y_pred, -50, 50))
                    loss = -torch.mean(y_t * torch.log(mu + 1e-15) + (1 - y_t) * torch.log(1 - mu + 1e-15))
                elif family == "poisson":
                    # Poisson: log-linear loss
                    mu = torch.exp(torch.clamp(y_pred, -50, 50))
                    loss = torch.mean(mu - y_t * torch.log(mu + 1e-15))
                else:
                    # Default to Gaussian
                    loss = torch.mean((y_pred - y_t) ** 2)

                loss.backward()

            # 走普通梯度下降的一步
            optimizer.step()

            # 第二步：双层非对称近端算子 (完全向量化，无Python循环)
            with torch.no_grad():
                w = weights
                # Apply soft thresholding vectorized
                w_new = torch.where(w > tau_pos, w - tau_pos, torch.where(w < -tau_neg, w + tau_neg, 0.0))

                # 检查收敛
                max_change = torch.max(torch.abs(w_new - weights))
                last_changes[change_idx] = max_change
                change_idx = (change_idx + 1) % 3

                # Converge only if last 3 changes are all below tol
                converged = epoch > 3 and torch.all(last_changes < tol)

                if converged:
                    break

                weights.copy_(w_new)

        # Move results back to CPU for storage
        betas_matrix[i, :] = weights.detach().cpu().numpy().flatten()
        intercepts[i] = bias.detach().cpu().item()
    return betas_matrix, intercepts


from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def cv_uni(
    X: np.ndarray,
    y: np.ndarray,
    family: str = "gaussian",
    n_folds: int = 5,
    lmdas: Optional[np.ndarray] = None,
    n_lmdas: int = 30,
    lmda_min_ratio: Optional[float] = 1e-1,
    alpha: float = 1.0,
    beta: float = 1.0,
    negative_penalty: float = 0.0,
    verbose: bool = False,
    seed: Optional[int] = None,
    backend: str = "numba",
    lr: Optional[float] = None,
    # 新增：自适应惩罚参数 (与 fit_uni 一致)
    adaptive_weighting: bool = False,
    weight_method: str = "t_statistic",
    weight_max_scale: float = None,
    gamma: float = 1.0,
    sharp_scale: float = 0.0,
    k: float = 1.0,  # 非对称惩罚权重调节系数
    # 新增：组约束参数 (与 fit_uni 一致)
    enable_group_constraint: bool = False,
    corr_threshold: float = 0.7,
    group_penalty: float = 5.0,
    max_group_size: int = 20,
    # 新增：非线性单变量模型参数
    univariate_model: str = "linear",
    spline_df: int = 5,
    spline_degree: int = 3,
    tree_max_depth: int = 2,
    # 新增：高相关变量组优化参数（支持成对/成组反符号变量场景）
    enable_group_decomp: bool = False,  # 开启组级正交分解，兼容原有enable_orthogonal_decomp
    enable_orthogonal_decomp: bool = False,  # 向后兼容别名
    group_corr_threshold: float = 0.7,  # 组相关系数阈值
    orthogonal_corr_threshold: float = 0.7,  # 向后兼容别名
    enable_group_aware_filter: bool = False,  # 开启组感知过滤，兼容原有enable_pair_aware_filter
    enable_pair_aware_filter: bool = False,  # 向后兼容别名
    group_filter_k: Optional[int] = None,  # 组过滤保留的变量数，兼容原有pair_filter_k
    pair_filter_k: Optional[int] = None  # 向后兼容别名
) -> UniLassoCVResult:
    """
    GroupAdaUniLasso: 交叉验证版的全GLM家族单变量引导 Lasso 回归。

    支持所有 fit_uni 的新功能：特征级自适应惩罚、组级符号一致性约束、
    非线性单变量模型（样条、决策树）。

    Parameters
    ----------
    family : str, optional
        GLM家族: 'gaussian' (线性回归), 'binomial' (二分类),
        'multinomial' (多分类), 'poisson' (泊松回归), 'cox' (生存分析)
    alpha : float, optional
        XLasso parameter: penalty coefficient for the $\frac{1}{w_j}$ term
        For significant variables (small w_j), this increases penalty on negative coefficients
        Default: 1.0
    beta : float, optional
        XLasso parameter: penalty coefficient for the $w_j$ term
        For insignificant variables (large w_j), this increases penalty on both signs
        Default: 1.0
    negative_penalty : float, optional
        Backward compatibility: Additional penalty strength for negative coefficients
        (used when alpha=0, this reduces to the old simplified form). Default: 0.0
    backend : str, optional
        Solver backend to use: 'numba' (default, fast) or 'pytorch' (original).
    adaptive_weighting : bool, optional
        开启特征级自适应惩罚 (默认关闭，与原版行为一致)
    weight_method : str, optional
        显著性权重计算方法: 't_statistic', 'p_value', 'correlation'
    weight_max_scale : float, optional
        如果提供，权重将归一化到 [1, weight_max_scale] 范围内。
        默认None，返回原始权重不做归一化。
    gamma : float, optional
        权重对比度调整参数：
        - gamma < 1: 平滑权重差异，适合高相关降维打击场景
        - gamma = 1: 默认线性映射
        - gamma > 1: 锐化权重差异，适合高频噪声场景
    sharp_scale : float, optional
        指数锐化参数（仅对p_value权重方法生效）：
        - sharp_scale = 0: 禁用指数锐化（默认）
        - sharp_scale > 0: 对p值大的非显著变量施加指数级惩罚增强
        推荐值：5~20，值越大非显著变量的惩罚增长越快
    enable_group_constraint : bool, optional
        开启组级一致性约束 (默认关闭，与原版行为一致)
    corr_threshold : float, optional
        共线性分组阈值 (默认0.7)
    group_penalty : float, optional
        全局组惩罚强度 (默认5.0)
    max_group_size : int, optional
        最大组大小 (默认20)
    univariate_model : str, optional
        单变量模型类型: 'linear' (线性，默认), 'spline' (样条), 'tree' (决策树)
    spline_df : int, optional
        样条回归自由度 (univariate_model='spline' 时使用，默认5)
    spline_degree : int, optional
        样条多项式次数 (默认3为三次样条)
    tree_max_depth : int, optional
        决策树最大深度 (univariate_model='tree' 时使用，默认2)
    """
    # 校验 univariate_model 参数
    if univariate_model not in VALID_UNIVARIATE_MODELS:
        raise ValueError(f"univariate_model must be one of {VALID_UNIVARIATE_MODELS}")

    # 向后兼容：当新功能都关闭时，回退到原有行为
    use_original = (not adaptive_weighting) and (not enable_group_constraint) and (univariate_model == "linear")

    # 1. 前置校验
    if backend not in ["numba", "pytorch"]:
        raise ValueError("backend must be either 'numba' or 'pytorch'")

    # 如果新功能关闭且需要用原始adelie（保持向后兼容）
    # 注意：cox模型仍然使用原始adelie，因为需要特殊处理
    if use_original and family == "cox":
        # 回退到原始adelie实现（cox在原始实现中有完整支持）
        return cv_unilasso(X, y, family, n_folds, lmda_min_ratio, verbose, seed)

    # Select solver based on backend
    if backend == "numba":
        from unilasso.solvers import _fit_numba_lasso_path_accelerated
        _fit_lasso_path = _fit_numba_lasso_path_accelerated
    else:
        _fit_lasso_path = _fit_pytorch_lasso_path

    # 2. 严格复用原版的数据准备逻辑，获取至关重要的 loo_fits
    X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, _, original_lmdas, zero_var_idx = _prepare_unilasso_input(
        X, y, family, lmdas,
        univariate_model=univariate_model,
        spline_df=spline_df,
        spline_degree=spline_degree,
        tree_max_depth=tree_max_depth
    )

    fit_intercept = False if family == "cox" else True

    # 设置默认学习率（根据GLM家族）
    if lr is None:
        if family == "gaussian":
            lr = 0.1  # 增大学习率，解决正则化过强问题
        elif family == "binomial":
            lr = 0.1
        elif family == "poisson":
            lr = 0.05
        elif family == "multinomial":
            lr = 0.1
        else:
            lr = 0.01

    # 3. 计算自适应权重和分组约束 (新功能，在全量数据上计算一次)
    feature_weights = None
    group_signs = None
    group_weights_arr = None
    groups = None

    if adaptive_weighting or enable_group_constraint:
        # 构建单变量结果字典
        univariate_results = {
            'beta': beta_coefs_fit,
            't_stats': np.ones_like(beta_coefs_fit),
            'p_values': np.ones_like(beta_coefs_fit),
            'correlations': np.zeros_like(beta_coefs_fit)
        }

        # Vectorized correlation computation
        if X.shape[1] > 0:
            n = X.shape[0]
            X_centered = X - np.mean(X, axis=0)
            y_centered = y - np.mean(y)
            X_std = np.std(X, axis=0)
            y_std = np.std(y)
            univariate_results['correlations'] = (X_centered.T @ y_centered) / (X_std * y_std * n)

    if adaptive_weighting:
        feature_weights = _compute_feature_significance_weights(
            univariate_results, weight_method, gamma, sharp_scale, weight_max_scale
        )
    else:
        feature_weights = np.ones(len(beta_coefs_fit))

    if enable_group_constraint:
        # 计算特征相关矩阵
        if len(beta_coefs_fit) > 1:
            corr_matrix = _parallel_corr_matrix(X)
        else:
            corr_matrix = np.array([[1.0]])

        # 贪心分组
        groups = _greedy_correlation_grouping(
            corr_matrix, corr_threshold, max_group_size
        )

        # 组惩罚增强模块已永久移除，所有组权重恒为1.0
        group_signs = np.ones(len(beta_coefs_fit))
        group_weights_arr = np.ones(len(beta_coefs_fit))
    else:
        group_signs = np.ones(len(beta_coefs_fit))
        group_weights_arr = np.ones(len(beta_coefs_fit))

    # 4. 确定正则化路径
    if original_lmdas is not None:
        lambda_path = np.sort(np.array(original_lmdas))[::-1]
    else:
        n_samples, n_features = loo_fits.shape
        p = n_features

        # 中心化数据，计算X^T y
        X_center = loo_fits - np.mean(loo_fits, axis=0)
        y_center = y - np.mean(y)
        Xty = X_center.T @ y_center / n_samples  # 每个特征的相关系数

        # 预计算标准化后的w_plus和w_minus（新的非对称惩罚公式）
        w_plus = np.zeros(n_features)
        w_minus = np.zeros(n_features)
        S_plus = 0.0
        S_minus = 0.0

        # 临时修复：k参数未定义，使用k=1.0
        k = 1.0
        for j in range(n_features):
            p_j = univariate_results['p_values'][j]
            # 新权重公式：w_j = 0.5 * p_j^k
            wj = 0.5 * (p_j ** k)
            wp = wj
            wm = 1.0 - wj
            w_plus[j] = wp
            w_minus[j] = wm
            S_plus += wp
            S_minus += wm

        # 标准化并乘以p
        S_plus_safe = max(S_plus, 1e-10)
        S_minus_safe = max(S_minus, 1e-10)
        for j in range(n_features):
            w_plus[j] = (w_plus[j] / S_plus_safe) * p
            w_minus[j] = (w_minus[j] / S_minus_safe) * p

        # 方案A：锚定原始梯度的lambda_max，只和数据本身有关，和权重无关
        # 这样lambda路径是固定的，权重变化会直接反映到惩罚强度上
        # 当信号权重w→0时，阻力λ*w真的变小；噪声权重变大时，阻力真的变大
        lambda_max = np.max(np.abs(Xty))

        # 生成lambda路径
        lambda_min = lambda_max * lmda_min_ratio
        lambda_path = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), n_lmdas))

        if not (backend == "numba" and 'accelerated' in str(_fit_lasso_path)):
            # 梯度下降需要放大100倍匹配lr尺度
            lambda_path *= 100.0

    # 5. 交叉验证过程
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    avg_losses = np.zeros(len(lambda_path))

    for train_idx, val_idx in kf.split(loo_fits):
        X_train, X_val = loo_fits[train_idx], loo_fits[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        b_mat, int_arr = _fit_lasso_path(
            X_train, y_train, lambda_path, alpha, beta, negative_penalty, fit_intercept,
            lr=lr,
            max_epochs=200,  # 坐标下降收敛快，200次足够
            tol=1e-4,  # 放宽收敛阈值，提升速度
            feature_weights=feature_weights,
            group_signs=group_signs,
            group_penalty=group_penalty,
            group_weights=group_weights_arr,
            family=family
        )

        for i in range(len(lambda_path)):
            preds = X_val @ b_mat[i] + int_arr[i]
            if family == "gaussian":
                avg_losses[i] += np.mean((preds - y_val) ** 2) / n_folds
            elif family == "binomial":
                # Logistic loss
                mu = 1.0 / (1.0 + np.exp(-np.clip(preds, -50, 50)))
                log_loss = -np.mean(y_val * np.log(mu + 1e-15) + (1 - y_val) * np.log(1 - mu + 1e-15))
                avg_losses[i] += log_loss / n_folds
            elif family == "poisson":
                # Poisson loss
                mu = np.exp(np.clip(preds, -50, 50))
                poisson_loss = np.mean(mu - y_val * np.log(mu + 1e-15))
                avg_losses[i] += poisson_loss / n_folds
            elif family == "multinomial":
                # Treat as binomial for now
                mu = 1.0 / (1.0 + np.exp(-np.clip(preds, -50, 50)))
                log_loss = -np.mean(y_val * np.log(mu + 1e-15) + (1 - y_val) * np.log(1 - mu + 1e-15))
                avg_losses[i] += log_loss / n_folds
            else:
                # Default to MSE
                avg_losses[i] += np.mean((preds - y_val) ** 2) / n_folds

    best_idx = int(np.argmin(avg_losses))
    best_lmda = lambda_path[best_idx]

    # 6. 在全量 LOO 特征上进行最终拟合
    final_betas, final_intercepts = _fit_lasso_path(
        loo_fits, y, lambda_path, alpha, beta, negative_penalty, fit_intercept,
        lr=lr,
        max_epochs=200,  # 坐标下降收敛快，200次足够
        tol=1e-4,  # 放宽收敛阈值，提升速度
        feature_weights=feature_weights,
        group_signs=group_signs,
        group_penalty=group_penalty,
        group_weights=group_weights_arr,
        family=family
    )

    # 将最终结果装入我们的适配器
    adapter_model = PyTorchGrpnetAdapter(final_betas, final_intercepts, lambda_path)

    # 7. 调用原生格式化函数
    gamma_hat, gamma_0, beta_coefs = _format_output(
        lasso_model=adapter_model,
        beta_coefs_fit=beta_coefs_fit,
        beta_intercepts=beta_intercepts,
        zero_var_idx=zero_var_idx,
        X=X,
        fit_intercept=fit_intercept
    )

    # 定义一个伪装的可视化函数
    def mock_cv_plot():
        plt.plot(np.log(lambda_path), avg_losses)
        plt.xlabel("Log(Lambda)")
        plt.ylabel("CV MSE Loss")
        plt.title("Custom UniLasso CV Plot")
        plt.show()

    if verbose:
        _print_unilasso_results(gamma_hat, lambda_path, best_idx)
        mock_cv_plot()

    # 8. 返回结果
    result = UniLassoCVResult(
        coefs=gamma_hat,  # 已经转换好的原始特征空间系数，直接X @ coefs[i]即可预测
        intercept=gamma_0,
        family=family,
        gamma=gamma_hat,
        gamma_intercept=gamma_0,
        beta=beta_coefs,  # 原始单变量系数，全局
        beta_intercepts=beta_intercepts,
        lasso_model=adapter_model,
        lmdas=lambda_path,
        avg_losses=avg_losses,
        cv_plot=mock_cv_plot,
        best_idx=best_idx,
        best_lmda=best_lmda
    )

    # 附加分组信息 (可选)
    if enable_group_constraint and groups is not None:
        result.groups = groups
        result.group_signs = group_signs

    return result



from typing import List, Optional, Union, Tuple, Dict
import numpy as np
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# 假设已经导入了 _prepare_unilasso_input, _format_output, _print_unilasso_results, UniLassoResult
# 以及我们之前编写的 _fit_pytorch_lasso_path 和 PyTorchGrpnetAdapter

def detect_high_correlation_groups(X: np.ndarray, corr_threshold: float = 0.7,
                                   max_group_size: int = 20) -> List[List[int]]:
    """
    检测高相关变量组，使用层次聚类
    Args:
        X: 特征矩阵 (n_samples, n_features)
        corr_threshold: 相关系数阈值，超过此值的变量会被聚为一组
        max_group_size: 最大组大小
    Returns:
        groups: 分组列表，每个元素是组内变量的索引列表
    """
    n_features = X.shape[1]
    if n_features <= 1:
        return []

    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(X.T)
    # 强制对称，解决浮点数精度问题
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    # 对角线强制为1，保证距离为0
    np.fill_diagonal(corr_matrix, 1.0)
    # 转换为距离矩阵：距离 = 1 - |相关系数|
    dist_matrix = 1 - np.abs(corr_matrix)
    # 对角线强制为0，满足距离矩阵要求
    np.fill_diagonal(dist_matrix, 0.0)
    # 层次聚类
    linkage_matrix = linkage(squareform(dist_matrix), method='average')
    # 聚类
    clusters = fcluster(linkage_matrix, t=1 - corr_threshold, criterion='distance')

    # 整理分组
    groups_dict: Dict[int, List[int]] = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in groups_dict:
            groups_dict[cluster_id] = []
        groups_dict[cluster_id].append(idx)

    # 过滤掉单变量组和超过最大大小的组
    groups = []
    for group in groups_dict.values():
        if 2 <= len(group) <= max_group_size:
            groups.append(group)

    return groups

def fit_uni(
    X: np.ndarray,
    y: np.ndarray,
    family: str = "gaussian",
    lmdas: Optional[Union[float, List[float], np.ndarray]] = None,
    n_lmdas: Optional[int] = 100,
    lmda_min_ratio: Optional[float] = 1e-2,
    lmda_scale: float = 1.0,  # 全局lambda缩放系数
    k: float = 1.0,  # 非对称惩罚权重调节系数
    alpha: float = 1.0,  # 向后兼容参数
    beta: float = 1.0,  # 向后兼容参数
    negative_penalty: float = 0.0,  # Backward compatibility: additional penalty on negative coefficients
    verbose: bool = False,
    backend: str = "numba",
    lr: Optional[float] = None,
    # 新增：Nesterov动量加速
    momentum: float = 0.0,
    # 新增：自适应惩罚参数
    adaptive_weighting: bool = False,
    weight_method: str = "t_statistic",
    weight_max_scale: float = None,
    gamma: float = 1.0,
    sharp_scale: float = 0.0,
    # 新增：组约束参数
    enable_group_constraint: bool = False,
    corr_threshold: float = 0.7,
    group_penalty: float = 5.0,
    max_group_size: int = 20,
    # 新增：非线性单变量模型参数
    univariate_model: str = "linear",
    spline_df: int = 5,
    spline_degree: int = 3,
    tree_max_depth: int = 2,
    # 新增：高相关变量组优化参数（支持成对/成组反符号变量场景）
    enable_group_decomp: bool = False,  # 开启组级正交分解，兼容原有enable_orthogonal_decomp
    enable_orthogonal_decomp: bool = False,  # 向后兼容别名
    group_corr_threshold: float = 0.7,  # 组相关系数阈值
    orthogonal_corr_threshold: float = 0.7,  # 向后兼容别名
    enable_group_aware_filter: bool = False,  # 开启组感知过滤，兼容原有enable_pair_aware_filter
    enable_pair_aware_filter: bool = False,  # 向后兼容别名
    group_filter_k: Optional[int] = None,  # 组过滤保留的变量数，兼容原有pair_filter_k
    pair_filter_k: Optional[int] = None  # 向后兼容别名
) -> UniLassoResult:
    """
    GroupAdaUniLasso: 全GLM家族升级的单变量引导 Lasso 回归。

    在指定的正则化路径上拟合模型，支持对负系数的自定义软惩罚、
    特征级自适应惩罚、组级符号一致性约束以及非线性单变量模型。

    This implementation matches the new XLasso weight design:
    $w_j = 0.5 \cdot p_j^k$, $w_j^+ = w_j$, $w_j^- = 1 - w_j$
    where $p_j$ is the p-value of the univariate regression for feature j,
    $k$ is the weight adjustment coefficient (default 1.0).

    Parameters
    ----------
    family : str, optional
        GLM家族: 'gaussian' (线性回归), 'binomial' (二分类),
        'multinomial' (多分类), 'poisson' (泊松回归), 'cox' (生存分析)
    alpha : float, optional
        XLasso parameter: penalty coefficient for the $\frac{1}{w_j}$ term
        For significant variables (small w_j), this increases penalty on negative coefficients
        Default: 1.0
    beta : float, optional
        XLasso parameter: penalty coefficient for the $w_j$ term
        For insignificant variables (large w_j), this increases penalty on both signs
        Default: 1.0
    negative_penalty : float, optional
        Backward compatibility: Additional penalty strength for negative coefficients
        (used when alpha=0, this reduces to the old simplified form). Default: 0.0
    backend : str, optional
        Solver backend to use: 'numba' (default, fast) or 'pytorch' (original).
    adaptive_weighting : bool, optional
        开启特征级自适应惩罚 (默认关闭，与原版行为一致)
    weight_method : str, optional
        显著性权重计算方法: 't_statistic', 'p_value', 'correlation'
    weight_max_scale : float, optional
        如果提供，权重将归一化到 [1, weight_max_scale] 范围内。
        默认None，返回原始权重不做归一化。
    gamma : float, optional
        权重对比度调整参数：
        - gamma < 1: 平滑权重差异，适合高相关降维打击场景
        - gamma = 1: 默认线性映射
        - gamma > 1: 锐化权重差异，适合高频噪声场景
    sharp_scale : float, optional
        指数锐化参数（仅对p_value权重方法生效）：
        - sharp_scale = 0: 禁用指数锐化（默认）
        - sharp_scale > 0: 对p值大的非显著变量施加指数级惩罚增强
        推荐值：5~20，值越大非显著变量的惩罚增长越快
    enable_group_constraint : bool, optional
        开启组级一致性约束 (默认关闭，与原版行为一致)
    corr_threshold : float, optional
        共线性分组阈值 (默认0.7)
    group_penalty : float, optional
        全局组惩罚强度 (默认5.0)
    max_group_size : int, optional
        最大组大小 (默认20)
    univariate_model : str, optional
        单变量模型类型: 'linear' (线性，默认), 'spline' (样条), 'tree' (决策树)
    spline_df : int, optional
        样条回归自由度 (univariate_model='spline' 时使用，默认5)
    spline_degree : int, optional
        样条多项式次数 (默认3为三次样条)
    tree_max_depth : int, optional
        决策树最大深度 (univariate_model='tree' 时使用，默认2)
    """
    # 校验 univariate_model 参数
    if univariate_model not in VALID_UNIVARIATE_MODELS:
        raise ValueError(f"univariate_model must be one of {VALID_UNIVARIATE_MODELS}")

    # 向后兼容：当新功能都关闭时，回退到原有行为
    use_original = (not adaptive_weighting) and (not enable_group_constraint) and (univariate_model == "linear")

    # 1. 前置校验
    if backend not in ["numba", "pytorch"]:
        raise ValueError("backend must be either 'numba' or 'pytorch'")

    # 如果新功能关闭且需要用原始adelie（保持向后兼容）
    # 注意：cox模型仍然使用原始adelie，因为需要特殊处理
    if use_original and family == "cox":
        # 回退到原始adelie实现（cox在原始实现中有完整支持）
        return fit_unilasso(X, y, family, lmdas, n_lmdas, lmda_min_ratio, verbose)

    # Select solver based on backend
    if backend == "numba":
        from unilasso.solvers import _fit_numba_lasso_path_accelerated
        _fit_lasso_path = _fit_numba_lasso_path_accelerated
    else:
        _fit_lasso_path = _fit_pytorch_lasso_path

    # 组级正交分解预处理（支持成对/成组反符号变量优化）
    transformation_info = []
    X_original = X.copy()
    # 向后兼容：两个开关任意一个开启都启用
    decomp_enabled = enable_group_decomp or enable_orthogonal_decomp
    if decomp_enabled:
        n_samples, n_features = X.shape
        X_transformed = X.copy()

        # 参数兼容
        corr_threshold = group_corr_threshold if group_corr_threshold != 0.7 else orthogonal_corr_threshold

        # 检测高相关变量组
        groups = detect_high_correlation_groups(
            X,
            corr_threshold=corr_threshold,
            max_group_size=max_group_size
        )

        # 对每个组做PCA正交变换
        for group in groups:
            group_size = len(group)
            if group_size < 2:
                continue

            # 提取组内特征
            X_group = X[:, group]
            # 中心化
            X_group_centered = X_group - np.mean(X_group, axis=0)
            # PCA变换
            cov_matrix = np.cov(X_group_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # 按特征值降序排列
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvectors_sorted = eigenvectors[:, sorted_indices]
            # 变换到正交空间
            X_group_transformed = X_group_centered @ eigenvectors_sorted

            # 替换原特征为变换后的特征
            for idx_in_group, original_idx in enumerate(group):
                X_transformed[:, original_idx] = X_group_transformed[:, idx_in_group]

            # 记录变换信息，用于逆变换
            transformation_info.append({
                "type": "group",
                "group_indices": group,
                "eigenvectors": eigenvectors_sorted,
                "mean": np.mean(X_group, axis=0)
            })

        # 替换X为变换后的数据
        X = X_transformed

    # 2. 数据准备与预处理
    X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, _, original_lmdas, zero_var_idx = _prepare_unilasso_input(
        X, y, family, lmdas,
        univariate_model=univariate_model,
        spline_df=spline_df,
        spline_degree=spline_degree,
        tree_max_depth=tree_max_depth
    )

    fit_intercept = False if family == "cox" else True

    # 设置默认学习率（根据GLM家族）
    if lr is None:
        if family == "gaussian":
            lr = 0.1  # 增大学习率，解决正则化过强问题
        elif family == "binomial":
            lr = 0.1
        elif family == "poisson":
            lr = 0.05
        elif family == "multinomial":
            lr = 0.1
        else:
            lr = 0.01

    # 3. 确定正则化路径 (Lambda Path)
    if original_lmdas is not None:
        lambda_path = np.sort(np.array(original_lmdas))[::-1]
    else:
        lambda_path = _configure_lmda_path(
                    X=loo_fits,
                    y=y,
                    family=family,
                    n_lmdas=n_lmdas,
                    lmda_min_ratio=lmda_min_ratio
                )

    # 全局lambda缩放
    lambda_path *= lmda_scale

    # 4. 计算自适应权重和分组约束 (新功能)
    feature_weights = None
    group_signs = None
    group_weights_arr = None
    groups = None

    if adaptive_weighting or enable_group_constraint:
        # 构建单变量结果字典
        # Try to compute real t-statistics and p-values for Gaussian linear regression
        univariate_results = {
            'beta': beta_coefs_fit,
            't_stats': np.ones_like(beta_coefs_fit),
            'p_values': np.ones_like(beta_coefs_fit),
            'correlations': np.zeros_like(beta_coefs_fit)
        }

        n, p = X.shape
        # Vectorized correlation computation
        if X.shape[1] > 0:
            X_centered = X - np.mean(X, axis=0)
            y_centered = y - np.mean(y)
            X_std = np.std(X, axis=0)
            y_std = np.std(y)
            univariate_results['correlations'] = (X_centered.T @ y_centered) / (X_std * y_std * n)

        # Compute t-statistics for Gaussian linear case
        if family == "gaussian" and univariate_model == "linear":
            # For each feature, compute t-statistic: beta / se(beta)
            y_mean = np.mean(y)
            y_ss = np.sum((y - y_mean) ** 2)
            for j in range(p):
                beta_j = beta_coefs_fit[j]
                if beta_j != 0:
                    x_j = X[:, j]
                    x_mean_j = np.mean(x_j)
                    ss_xj = np.sum((x_j - x_mean_j) ** 2)
                    if ss_xj > 1e-10:
                        # R-squared for this univariate model
                        r2 = beta_j ** 2 * ss_xj / y_ss if y_ss > 0 else 0
                        # Residual variance estimate
                        df_resid = n - 2
                        mse = y_ss * (1 - r2) / df_resid if df_resid > 0 else 0
                        # Standard error of beta
                        se_beta = np.sqrt(mse / ss_xj)
                        # t-statistic
                        t_j = beta_j / se_beta if se_beta > 0 else 0
                        univariate_results['t_stats'][j] = np.abs(t_j)
                        # Two-tailed p-value (approximate using normal distribution)
                        # We'll leave p_values for computation in _compute_feature_significance_weights
                        # but fill with 1/(1 + |t_j|) as an approximation of p-value
                        import scipy.stats
                        if abs(t_j) < 10:
                            p_j = 2 * (1 - scipy.stats.norm.cdf(abs(t_j)))
                        else:
                            p_j = 1e-10  # Approximation for large t
                        # 对p值做安全截断，防止极端值
                        p_j = np.clip(p_j, 1e-4, 0.95)
                        univariate_results['p_values'][j] = p_j

    if adaptive_weighting:
        feature_weights = _compute_feature_significance_weights(
            univariate_results, weight_method, gamma, sharp_scale, weight_max_scale
        )
    else:
        feature_weights = np.ones(len(beta_coefs_fit))

    if enable_group_constraint:
        # 计算特征相关矩阵
        if len(beta_coefs_fit) > 1:
            corr_matrix = _parallel_corr_matrix(X)
        else:
            corr_matrix = np.array([[1.0]])

        # 贪心分组
        groups = _greedy_correlation_grouping(
            corr_matrix, corr_threshold, max_group_size
        )

        # 组惩罚增强模块已永久移除，所有组权重恒为1.0
        group_signs = np.ones(len(beta_coefs_fit))
        group_weights_arr = np.ones(len(beta_coefs_fit))
    else:
        group_signs = np.ones(len(beta_coefs_fit))
        group_weights_arr = np.ones(len(beta_coefs_fit))

    # 5. 调用核心求解器
    betas_matrix, intercepts_array = _fit_lasso_path(
        X_train=loo_fits,
        y_train=y,
        lmdas=lambda_path,
        alpha=alpha,
        beta=beta,
        negative_penalty=negative_penalty,
        fit_intercept=fit_intercept,
        lr=lr,
        momentum=momentum,
        feature_weights=feature_weights,
        group_signs=group_signs,
        group_penalty=group_penalty,
        group_weights=group_weights_arr,
        family=family
    )

    # 6. 适配器模式启动
    adapter_model = PyTorchGrpnetAdapter(betas_matrix, intercepts_array, lambda_path)

    # 7. 参数还原与格式化
    gamma_hat, gamma_0, beta_coefs = _format_output(
        lasso_model=adapter_model,
        beta_coefs_fit=beta_coefs_fit,
        beta_intercepts=beta_intercepts,
        zero_var_idx=zero_var_idx,
        X=X,
        fit_intercept=fit_intercept,
        reverse_indices=None
    )

    if verbose:
        _print_unilasso_results(gamma_hat, lambda_path)

    # 正交分量逆变换（如果开启了组分解/正交分解）
    decomp_enabled = enable_group_decomp or enable_orthogonal_decomp
    if decomp_enabled and len(transformation_info) > 0:
        n_lmdas, n_features = gamma_hat.shape
        gamma_hat_original = gamma_hat.copy()

        for info in transformation_info:
            if info["type"] == "group":
                # 组级逆变换
                group_indices = info["group_indices"]
                eigenvectors = info["eigenvectors"]
                group_size = len(group_indices)

                for lmda_idx in range(n_lmdas):
                    # 提取变换空间的系数
                    beta_transformed = gamma_hat[lmda_idx, group_indices]
                    # 逆PCA变换：beta_original = eigenvectors @ beta_transformed + mean？
                    # 注意：我们只变换了X，系数的逆变换是 beta_original = eigenvectors @ beta_transformed
                    beta_original = eigenvectors @ beta_transformed
                    # 赋值回原始变量
                    for idx_in_group, original_idx in enumerate(group_indices):
                        gamma_hat_original[lmda_idx, original_idx] = beta_original[idx_in_group]

            elif "pair" in info:
                # 兼容旧版本的成对变换
                i, j = info["pair"]
                for lmda_idx in range(n_lmdas):
                    beta_S = gamma_hat[lmda_idx, i]
                    beta_D = gamma_hat[lmda_idx, j]
                    # 逆变换回原始空间
                    gamma_hat_original[lmda_idx, i] = beta_S + beta_D
                    gamma_hat_original[lmda_idx, j] = beta_S - beta_D

        gamma_hat = gamma_hat_original

    # 组感知过滤（如果开启，兼容原有成对感知过滤）
    filter_enabled = enable_group_aware_filter or enable_pair_aware_filter
    if filter_enabled:
        n_lmdas, n_features = gamma_hat.shape
        # 参数兼容
        corr_threshold = group_corr_threshold if group_corr_threshold != 0.7 else orthogonal_corr_threshold
        filter_k = group_filter_k if group_filter_k is not None else pair_filter_k

        # 预检测所有高相关组
        all_groups = detect_high_correlation_groups(
            X_original,
            corr_threshold=corr_threshold,
            max_group_size=max_group_size
        )

        for lmda_idx in range(n_lmdas):
            beta = gamma_hat[lmda_idx, :]
            beta_filtered = beta.copy()

            # 对每个组进行组感知过滤
            for group in all_groups:
                group_size = len(group)
                if group_size == 0:
                    continue

                # 统计组内被选中的变量数量（绝对值 > 1e-8）
                count_selected = 0
                for j in group:
                    if np.abs(beta[j]) > 1e-8:
                        count_selected += 1

                # 确定保留阈值
                if filter_k is None:
                    # 使用比例阈值：默认保留组大小的50%
                    threshold = int(group_size * 0.5)
                else:
                    # 使用绝对数量阈值
                    threshold = filter_k

                # 组感知过滤规则：
                # 如果组内选中变量数 >= 阈值，保留组内所有变量
                # 否则，过滤掉整个组
                if count_selected < threshold:
                    for j in group:
                        beta_filtered[j] = 0.0

            gamma_hat[lmda_idx, :] = beta_filtered

    # 8. 返回标准结果对象 (附加分组信息)
    unilasso_result = UniLassoResult(
        coefs=gamma_hat,
        intercept=gamma_0,
        family=family,
        gamma=gamma_hat,
        gamma_intercept=gamma_0,
        beta=beta_coefs,
        beta_intercepts=beta_intercepts,
        lasso_model=adapter_model,
        lmdas=lambda_path
    )

    # 附加分组信息 (可选)
    if enable_group_constraint and groups is not None:
        unilasso_result.groups = groups
        unilasso_result.group_signs = group_signs

    return unilasso_result

def plot_uni(unilasso_fit) -> None:
    """
    创新版 Lasso 路径可视化。
    使用行业标准 -log(lambda) 作为横轴，使得从左到右代表模型复杂度增加（正则化减弱）。
    """
    assert hasattr(unilasso_fit, "coefs") and hasattr(unilasso_fit, "lmdas"), \
        "Input must have 'coefs' and 'lmdas' attributes."

    coefs, lambdas = unilasso_fit.coefs, unilasso_fit.lmdas
    
    if coefs.ndim == 1 or len(lambdas) == 1:
        print("Only one regularization parameter was used. No path to plot.")
        return

    plt.figure(figsize=(8, 6))
    
    # --- 核心改动点 ---
    # 改为使用 -log(lambda)，这是业界更通用的做法
    neg_log_lambdas = -np.log(lambdas) 

    # Compute the number of nonzero coefficients at each lambda
    n_nonzero = np.sum(coefs != 0, axis=1)

    # Plot coefficient paths
    for i in range(coefs.shape[1]):  
        plt.plot(neg_log_lambdas, coefs[:, i], lw=2)

    # --- 标签更新 ---
    plt.xlabel(r"$-\log(\lambda)$", fontsize=12)  # 更新 LaTeX 标签
    plt.ylabel("Coefficients", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # 添加顶部的第二 X 轴（用于显示非零特征数量）
    ax1 = plt.gca()  
    ax2 = ax1.twiny()  
    ax2.set_xlim(ax1.get_xlim())  
    
    # 动态计算刻度，保持美观
    tick_indices = np.linspace(0, len(neg_log_lambdas) - 1, min(6, len(neg_log_lambdas)), dtype=int)
    ax2.set_xticks(neg_log_lambdas[tick_indices])  
    ax2.set_xticklabels(n_nonzero[tick_indices]) 
    
    ax2.set_xlabel("Number of Active Coefficients", fontsize=12)

    plt.show()
    
    
def plot_cv_uni(cv_result) -> None:
    """
    创新版交叉验证损失曲线可视化。
    彻底接管绘图逻辑，使用 -log(lambda) 作为横轴，并高亮最佳参数点。
    """
    # 1. 前置契约校验 (Contract Check)
    assert hasattr(cv_result, "lmdas") and hasattr(cv_result, "avg_losses"), \
        "输入对象必须包含 'lmdas' 和 'avg_losses' 属性"
    
    # 2. 从对象中提取原始数据 (打破原先的黑盒调用机制)
    lambdas = cv_result.lmdas
    avg_losses = cv_result.avg_losses
    best_lmda = cv_result.best_lmda
    
    # --- 核心改动：转换为 -log ---
    neg_log_lambdas = -np.log(lambdas)
    
    plt.figure(figsize=(8, 6))
    
    # 绘制交叉验证平均损失曲线
    plt.plot(
        neg_log_lambdas, 
        avg_losses, 
        marker='o', 
        linestyle='-', 
        color='royalblue', 
        markersize=5, 
        linewidth=2,
        label='CV MSE Loss'
    )
    
    # --- 工业级体验增强：标出最佳点 ---
    if best_lmda is not None:
        best_neg_log = -np.log(best_lmda)
        # 获取最佳 lambda 对应的 loss 值用于画点
        best_idx = cv_result.best_idx
        best_loss = avg_losses[best_idx]
        
        # 画一条垂直虚线辅助对齐
        plt.axvline(x=best_neg_log, color='crimson', linestyle='--', alpha=0.7, label=f'Best $-\\log(\\lambda)$ = {best_neg_log:.2f}')
        # 在最低点画一个红星
        plt.plot(best_neg_log, best_loss, marker='*', color='crimson', markersize=12)

    # 设置符合行业标准的标签
    plt.xlabel(r"$-\log(\lambda)$", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.title("Cross-Validation Error Curve", fontsize=14, pad=15)
    
    # 优化网格和图例，提升图表可读性
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', frameon=True)
    
    plt.show()