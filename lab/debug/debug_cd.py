"""
Debug coordinate descent implementation
"""
import numpy as np
from unilasso.solvers import _fit_numba_lasso_path_accelerated

# Generate simple test data
np.random.seed(42)
n = 100
p = 10
X = np.random.randn(n, p)
beta_true = np.zeros(p)
beta_true[:3] = [1.0, 2.0, -1.5]  # 3 true features
y = X @ beta_true + np.random.randn(n) * 0.1

print("True coefficients:", beta_true)

# Test coordinate descent solver
lmdas = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0])
betas, intercepts = _fit_numba_lasso_path_accelerated(
    X, y, lmdas,
    alpha=0.0,  # No asymmetric penalty for test
    beta=1.0,
    family="gaussian",
    fit_intercept=True
)

print("\nResults:")
for i, lmda in enumerate(lmdas):
    print(f"\nLambda = {lmda}:")
    print(f"  Coefficients: {betas[i].round(3)}")
    print(f"  Intercept: {intercepts[i].round(3)}")
    print(f"  Non-zero count: {np.sum(np.abs(betas[i]) > 1e-8)}")
