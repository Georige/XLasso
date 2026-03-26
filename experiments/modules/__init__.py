"""
Experiments Modules - Unified Algorithm Interface
================================================
Provides unified access to sparse feature selection algorithms:

NLasso Family:
- NLasso, NLassoClassifier, NLassoCV, NLassoClassifierCV
  Ridge-based first stage + asymmetric penalty (from NLasso/)

Other Lasso Variants (from other_lasso/):
- AdaptiveLasso, AdaptiveLassoCV
- FusedLasso, FusedLassoCV
- GroupLasso, GroupLassoCV
- AdaptiveSparseGroupLasso, AdaptiveSparseGroupLassoCV

XLasso (from xlasso/):
- XLasso, XLassoCV (wrap fit_uni/cv_uni)

UniLasso (from unilasso/):
- UniLasso, UniLassoCV (wrap fit_unilasso/cv_unilasso)
- fit_loo_univariate_models
- simulate_gaussian_data, simulate_binomial_data, simulate_cox_data

Standard Lasso (from sklearn):
- Lasso, LassoCV

Base Utilities:
- MetricCalculator: Unified metrics computation
- CrossValidator: K-fold cross-validation wrapper
- DataGenerator: Synthetic data generation
"""

import numpy as np
from sklearn.model_selection import KFold
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# =============================================================================
# NLasso Family (from NLasso/)
# =============================================================================
from .NLasso import NLasso, NLassoClassifier, NLassoCV, NLassoClassifierCV
from .NLasso import metrics as nlasso_metrics

# =============================================================================
# Other Lasso Variants (from other_lasso/)
# =============================================================================
from .other_lasso import (
    AdaptiveLasso,
    AdaptiveLassoCV,
    FusedLasso,
    FusedLassoCV,
    GroupLasso,
    GroupLassoCV,
    AdaptiveSparseGroupLasso,
    AdaptiveSparseGroupLassoCV,
)

# =============================================================================
# UniLasso (from unilasso/)
# =============================================================================
from unilasso import (
    fit_unilasso,
    cv_unilasso,
    fit_loo_univariate_models,
    simulate_gaussian_data,
    simulate_binomial_data,
    simulate_cox_data,
)

# =============================================================================
# Base Utilities
# =============================================================================

class MetricCalculator:
    """
    Unified metrics calculator for sparse regression evaluation.
    Computes: MSE, R2, F1, TPR, FDR, Precision, Recall, Sparsity
    """

    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        beta_true: np.ndarray,
        beta_est: np.ndarray,
        threshold: float = 1e-6,
    ) -> Dict[str, float]:
        """Calculate all metrics."""
        from .NLasso.metrics import (
            mean_squared_error,
            r2_score,
        )

        # Regression metrics
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Feature selection metrics
        true_nonzero = np.where(np.abs(beta_true) > threshold)[0]
        selected = np.where(np.abs(beta_est) > threshold)[0]
        n_selected = len(selected)

        # TPR: How many true nonzero features were selected
        true_nonzero_set = set(true_nonzero)
        selected_set = set(selected)
        tp = len(true_nonzero_set & selected_set)
        fn = len(true_nonzero_set - selected_set)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # FDR: How many selected features are false positives
        fp = len(selected_set - true_nonzero_set)
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0

        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Sparsity
        sparsity = 1.0 - n_selected / len(beta_true) if len(beta_true) > 0 else 0.0

        return {
            "mse": mse,
            "r2": r2,
            "f1": f1,
            "tpr": tpr,
            "fdr": fdr,
            "precision": precision,
            "recall": recall,
            "sparsity": sparsity,
            "n_selected": n_selected,
        }


class CrossValidator:
    """K-fold cross-validation wrapper."""
    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.kfold = KFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_state,
        )

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Generate train/test splits."""
        yield from self.kfold.split(X)


@dataclass
class DataGenerator:
    """
    Unified data generator for sparse regression experiments.
    Supports: pairwise, ar1, twin correlation structures.
    """
    random_state: int = 42

    def __post_init__(self):
        self.rng = np.random.RandomState(self.random_state)

    def generate(
        self,
        n_samples: int = 300,
        n_features: int = 500,
        n_nonzero: int = 20,
        sigma: float = 1.0,
        correlation_type: str = "pairwise",
        rho: float = 0.5,
        family: str = "gaussian",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic sparse regression data."""
        if correlation_type == "pairwise":
            return self._generate_pairwise(n_samples, n_features, n_nonzero, sigma, family, rho)
        elif correlation_type == "ar1":
            return self._generate_ar1(n_samples, n_features, n_nonzero, sigma, family, rho)
        elif correlation_type == "twin":
            return self._generate_twin(n_samples, n_features, n_nonzero, sigma, family, rho)
        else:
            raise ValueError(f"Unknown correlation_type: {correlation_type}")

    def _generate_pairwise(self, n, p, k, sigma, family, rho) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate equicorrelated data."""
        cov = np.full((p, p), rho)
        np.fill_diagonal(cov, 1.0)
        X = self.rng.multivariate_normal(np.zeros(p), cov, size=n)
        beta_true = np.zeros(p)
        beta_true[:k] = 1.0
        if family == "gaussian":
            y = X @ beta_true + self.rng.randn(n) * sigma
        else:
            z = X @ beta_true + self.rng.randn(n) * sigma
            y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
        return X, y, beta_true

    def _generate_ar1(self, n, p, k, sigma, family, rho) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate AR(1) autocorrelated data."""
        cov = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                cov[i, j] = rho ** abs(i - j)
        X = self.rng.multivariate_normal(np.zeros(p), cov, size=n)
        beta_true = np.zeros(p)
        beta_true[1 : 2 * k : 2] = 1.0
        if family == "gaussian":
            y = X @ beta_true + self.rng.randn(n) * sigma
        else:
            z = X @ beta_true + self.rng.randn(n) * sigma
            y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
        return X, y, beta_true

    def _generate_twin(self, n, p, k, sigma, family, rho) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate twin variables (pairs with opposite signs)."""
        X = self.rng.randn(n, p)
        beta_true = np.zeros(p)
        for i in range(k):
            beta_true[2 * i] = 2.0
            beta_true[2 * i + 1] = -2.5
        if family == "gaussian":
            y = X @ beta_true + self.rng.randn(n) * sigma
        else:
            z = X @ beta_true + self.rng.randn(n) * sigma
            y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
        return X, y, beta_true


# =============================================================================
# Algorithm Registry (for factory/run.py)
# =============================================================================

# Try to import XLasso and UniLasso wrappers if available
try:
    from .xlasso import XLasso, XLassoCV
    HAS_XLASSO = True
except ImportError:
    HAS_XLASSO = False
    XLasso = None
    XLassoCV = None

try:
    from .lasso import Lasso, LassoCV
    HAS_LASSO = True
except ImportError:
    HAS_LASSO = False
    Lasso = None
    LassoCV = None

try:
    from .uniLasso import UniLasso, UniLassoCV
    from .postlasso import PostLasso
    HAS_UNILASSO = True
except ImportError:
    HAS_UNILASSO = False
    UniLasso = None
    UniLassoCV = None
    PostLasso = None

ALGO_REGISTRY = {
    # NLasso family
    "nlasso": NLasso,
    "nlclassifier": NLassoClassifier,
    # XLasso (fit_uni/cv_uni)
    "xlasso": XLasso,
    "xlasso_cv": XLassoCV,
    # UniLasso (fit_unilasso/cv_unilasso)
    "unilasso": UniLasso,
    "unilasso_cv": UniLassoCV,
    # Standard Lasso
    "lasso": Lasso,
    "lasso_cv": LassoCV,
    # Other Lasso variants
    "adaptive_lasso": AdaptiveLasso,
    "fused_lasso": FusedLasso,
    "group_lasso": GroupLasso,
    "adaptive_sparse_group_lasso": AdaptiveSparseGroupLasso,
    # PostLasso
    "postlasso": PostLasso,
}


__all__ = [
    # NLasso family
    "NLasso",
    "NLassoClassifier",
    "NLassoCV",
    "NLassoClassifierCV",
    "nlasso_metrics",
    # Other Lasso variants
    "AdaptiveLasso",
    "AdaptiveLassoCV",
    "FusedLasso",
    "FusedLassoCV",
    "GroupLasso",
    "GroupLassoCV",
    "AdaptiveSparseGroupLasso",
    "AdaptiveSparseGroupLassoCV",
    # UniLasso functions
    "fit_unilasso",
    "cv_unilasso",
    "fit_loo_univariate_models",
    "simulate_gaussian_data",
    "simulate_binomial_data",
    "simulate_cox_data",
    # XLasso (fit_uni/cv_uni)
    "XLasso",
    "XLassoCV",
    # UniLasso (fit_unilasso/cv_unilasso)
    "UniLasso",
    "UniLassoCV",
    # Standard Lasso
    "Lasso",
    "LassoCV",
    # PostLasso
    "PostLasso",
    # Base utilities
    "MetricCalculator",
    "CrossValidator",
    "DataGenerator",
    # Registry
    "ALGO_REGISTRY",
]