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
from .NLasso import (
    NLasso, NLassoClassifier, NLassoCV, NLassoClassifierCV,
    AdaptiveFlippedLasso, AdaptiveFlippedLassoClassifier,
    AdaptiveFlippedLassoCV, AdaptiveFlippedLassoClassifierEBIC,
    AdaptiveFlippedLassoEBIC, AdaptiveFlippedLassoCV_EN,
    AdaptiveFlippedLassoCV_ENClassifier,
    AdaptiveFlippedLassoEBIC_Simple,
)
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
    Supports: pairwise, ar1, twin, block correlation structures.
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
        block_size: int = 10,
        n_blocks: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic sparse regression data."""
        if correlation_type == "pairwise":
            return self._generate_pairwise(n_samples, n_features, n_nonzero, sigma, family, rho)
        elif correlation_type == "ar1":
            return self._generate_ar1(n_samples, n_features, n_nonzero, sigma, family, rho)
        elif correlation_type == "twin":
            return self._generate_twin(n_samples, n_features, n_nonzero, sigma, family, rho)
        elif correlation_type == "block":
            return self._generate_block(n_samples, n_features, n_nonzero, sigma, family, rho, block_size, n_blocks)
        elif correlation_type == "experiment1":
            return self._generate_experiment1(n_samples, n_features, n_nonzero, sigma, family, rho)
        elif correlation_type == "experiment2":
            return self._generate_experiment2(n_samples, n_features, n_nonzero, sigma, family, rho)
        elif correlation_type == "experiment3":
            return self._generate_experiment3(n_samples, n_features, sigma, family, rho)
        elif correlation_type == "experiment4":
            return self._generate_experiment4(n_samples, n_features, sigma, family, rho)
        elif correlation_type == "experiment5":
            return self._generate_experiment5(n_samples, n_features, sigma, family, rho)
        elif correlation_type == "experiment6":
            return self._generate_experiment6(n_samples, n_features, sigma, family, rho)
        elif correlation_type == "experiment7":
            return self._generate_experiment7(n_samples, n_features, sigma, family, rho)
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

    def _generate_block(self, n, p, k, sigma, family, rho, block_size, n_blocks) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate block diagonal correlation structure."""
        cov = np.zeros((p, p))
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, p)
            block_cov = np.full((end - start, end - start), rho)
            np.fill_diagonal(block_cov, 1.0)
            cov[start:end, start:end] = block_cov
        X = self.rng.multivariate_normal(np.zeros(p), cov, size=n)
        beta_true = np.zeros(p)
        beta_true[:k] = 1.0
        if family == "gaussian":
            y = X @ beta_true + self.rng.randn(n) * sigma
        else:
            z = X @ beta_true + self.rng.randn(n) * sigma
            y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
        return X, y, beta_true

    def _generate_experiment1(self, n, p, k, sigma, family, rho) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate experiment 1 data: pairwise correlation, first k vars β=1.0.

        n=300, p=500, X~N(0,Σ) with Σ_ij=rho
        first k variables β=1.0, remaining p-k variables β=0
        """
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

    def _generate_experiment2(self, n, p, k, sigma, family, rho) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate experiment 2 data: AR(1) correlation, odd-indexed first k vars β=1.0.

        n=300, p=500, X~N(0,Σ) with Σ_ij=rho^|i-j|
        odd-indexed first k variables β=1.0 (j=1,3,5,...,39)
        """
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

    def _generate_experiment3(self, n, p, sigma, family, rho) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate experiment 3 data: binomial with offset, AR(1) correlation.

        n=300, p=500, X~N(0,Σ) with AR(1) ρ=0.8
        first 20 variables β=1.0
        y=1 samples have offset 0.6 on first 20 variables, ~150 each class
        """
        k = 20
        cov = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                cov[i, j] = rho ** abs(i - j)
        X = self.rng.multivariate_normal(np.zeros(p), cov, size=n)
        beta_true = np.zeros(p)
        beta_true[:k] = 1.0

        if family == "binomial":
            z = X @ beta_true + self.rng.randn(n) * sigma
            y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
            # Ensure class balance: ~150 each
            y1_idx = y == 1
            if np.sum(y1_idx) > 160:
                drop_idx = self.rng.choice(np.where(y1_idx)[0], int(np.sum(y1_idx) - 150), replace=False)
                y[drop_idx] = 0
            elif np.sum(y1_idx) < 140:
                add_idx = self.rng.choice(np.where(y == 0)[0], int(150 - np.sum(y1_idx)), replace=False)
                y[add_idx] = 1
            # Offset 0.6 on y=1 samples
            y1_idx = y == 1
            X[y1_idx, :k] += 0.6
        else:
            y = X @ beta_true + self.rng.randn(n) * sigma
        return X, y, beta_true

    def _generate_experiment4(self, n, p, sigma, family, rho) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate experiment 4 data: opposite-sign twin variables with correlation.

        n=300, p=1000, 10 pairs twin variables with ρ=0.85
        β_{2t-1}=2.0, β_{2t}=-2.5, remaining 980 variables are noise
        """
        X = self.rng.randn(n, p)
        beta_true = np.zeros(p)

        # Generate 10 pairs of twin variables with correlation rho
        for i in range(10):
            common = self.rng.randn(n)
            X[:, 2*i] = common * np.sqrt(rho) + self.rng.randn(n) * np.sqrt(1-rho)
            X[:, 2*i+1] = -common * np.sqrt(rho) + self.rng.randn(n) * np.sqrt(1-rho)
            beta_true[2*i] = 2.0
            beta_true[2*i+1] = -2.5

        if family == "gaussian":
            y = X @ beta_true + self.rng.randn(n) * sigma
        else:
            z = X @ beta_true + self.rng.randn(n) * sigma
            y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
        return X, y, beta_true

    def _generate_experiment5(self, n, p, sigma, family, rho) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate experiment 5 data: Perfect Masking (Absolute Invisible Trap).

        n=300, p=1000, 10 pairs twin variables with ρ=0.8
        β_{2t-1}=2.0, β_{2t}=-2.5, remaining 980 variables are independent noise
        Core challenge: Design X such that marginal correlation of true signals → 0
        """
        X = self.rng.randn(n, p)
        beta_true = np.zeros(p)

        # Generate 10 pairs of twin variables (same structure as exp4)
        for i in range(10):
            common = self.rng.randn(n)
            X[:, 2*i] = common * np.sqrt(rho) + self.rng.randn(n) * np.sqrt(1-rho)
            X[:, 2*i+1] = -common * np.sqrt(rho) + self.rng.randn(n) * np.sqrt(1-rho)
            beta_true[2*i] = 2.0
            beta_true[2*i+1] = -2.5

        # Perfect masking: rotate twin variables to have zero marginal correlation with y
        y_base = X @ beta_true + self.rng.randn(n) * sigma

        for i in range(10):
            v1 = X[:, 2*i].copy()
            v2 = X[:, 2*i+1].copy()

            # Make v1 orthogonal to y_base (marginal correlation → 0)
            proj_y = np.dot(v1, y_base) / (np.dot(y_base, y_base) + 1e-10)
            v1_ortho = v1 - proj_y * y_base
            v1_ortho = v1_ortho / (np.linalg.norm(v1_ortho) + 1e-10)

            # Same for v2
            proj_y2 = np.dot(v2, y_base) / (np.dot(y_base, y_base) + 1e-10)
            v2_ortho = v2 - proj_y2 * y_base
            v2_ortho = v2_ortho / (np.linalg.norm(v2_ortho) + 1e-10)

            X[:, 2*i] = v1_ortho * np.std(X[:, 2*i]) + np.mean(X[:, 2*i])
            X[:, 2*i+1] = v2_ortho * np.std(X[:, 2*i+1]) + np.mean(X[:, 2*i+1])

        if family == "gaussian":
            y = X @ beta_true + self.rng.randn(n) * sigma
        else:
            z = X @ beta_true + self.rng.randn(n) * sigma
            y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
        return X, y, beta_true

    def _generate_experiment6(self, n, p, sigma, family, rho) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate experiment 6 data: The Decoy Trap.

        n=300, p=500, 5 groups of 3 variables each
        True signal: first 2 vars in each group β=1.0
        Noise decoy: 3rd var in each group, correlated with group signals (ρ=0.8)
        Remaining 482 vars are independent noise β=0
        """
        X = self.rng.randn(n, p)
        beta_true = np.zeros(p)

        # 5 groups of 3 variables
        groups = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (12, 13, 14), (15, 16, 17)]

        for a, b, c in groups:
            # a,b are true signals (independent), c is noise decoy (correlated with a,b)
            common_ab = self.rng.randn(n) * np.sqrt(0.5)
            indep_a = self.rng.randn(n) * np.sqrt(0.5)
            indep_b = self.rng.randn(n) * np.sqrt(0.5)
            indep_c = self.rng.randn(n) * np.sqrt(1 - rho)

            X[:, a] = common_ab + indep_a
            X[:, b] = common_ab + indep_b
            # c correlated with a,b at level rho
            common_ac = self.rng.randn(n) * np.sqrt(rho)
            X[:, c] = common_ac + indep_c

            beta_true[a] = 1.0
            beta_true[b] = 1.0
            # beta_true[c] = 0 (remains 0)

        if family == "gaussian":
            y = X @ beta_true + self.rng.randn(n) * sigma
        else:
            z = X @ beta_true + self.rng.randn(n) * sigma
            y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
        return X, y, beta_true

    def _generate_experiment7(self, n, p, sigma, family, rho) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate experiment 7 data: AR(1) Sign Avalanche.

        n=300, p=500, first 20 variables AR(1) chain Σ_ij=0.9^|i-j|
        β_j = (-1)^(j+1) * 2.0 * 0.9^((j-1)/2), j=1..20
        e.g. β1=2.0, β2=-1.8, β3=1.62, β4=-1.458, ...
        Remaining 480 vars β=0
        """
        cov = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                cov[i, j] = rho ** abs(i - j)
        X = self.rng.multivariate_normal(np.zeros(p), cov, size=n)

        beta_true = np.zeros(p)
        for j in range(1, 21):
            beta_true[j-1] = ((-1) ** (j + 1)) * 2.0 * (0.9 ** ((j - 1) / 2))

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
    "nlasso_cv": NLassoCV,
    # AdaptiveFlippedLasso family
    "adaptive_flipped_lasso": AdaptiveFlippedLasso,
    "aflclassifier": AdaptiveFlippedLassoClassifier,
    "aflclassifier_cv": AdaptiveFlippedLassoCV,
    "aflclassifier_ebic": AdaptiveFlippedLassoClassifierEBIC,
    "adaptive_flipped_lasso_cv_en": AdaptiveFlippedLassoCV_EN,
    "adaptive_flipped_lasso_cv_en_classifier": AdaptiveFlippedLassoCV_ENClassifier,
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
    # AdaptiveFlippedLasso family
    "AdaptiveFlippedLasso",
    "AdaptiveFlippedLassoClassifier",
    "AdaptiveFlippedLassoCV",
    "AdaptiveFlippedLassoClassifierEBIC",
    "AdaptiveFlippedLassoEBIC",
    "AdaptiveFlippedLassoCV_EN",
    "AdaptiveFlippedLassoCV_ENClassifier",
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