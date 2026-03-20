"""
Single test run after fixing lambda scaling issue
"""
import numpy as np
from unilasso.uni_lasso import cv_unilasso, cv_uni, extract_cv
from experiment_ar1 import generate_data

# Generate test data
np.random.seed(42)
X_train, y_train, X_test, y_test, beta_true, true_indices = generate_data(
    n=300, p=1000, sigma=1.0, rho=0.8
)

print(f"True non-zero coefficients: {len(true_indices)}")
print(f"True beta range: [{beta_true.min():.2f}, {beta_true.max():.2f}]")

# Test 1: Original UniLasso for baseline
print("\n=== 1. Original UniLasso (Baseline) ===")
cv_result_uni = cv_unilasso(
    X_train, y_train, family="gaussian", n_folds=5, verbose=False
)
fit_uni = extract_cv(cv_result_uni)
y_pred_uni = X_test @ fit_uni.coefs + fit_uni.intercept
mse_uni = np.mean((y_pred_uni - y_test) ** 2)
selected_uni = np.sum(np.abs(fit_uni.coefs) > 1e-8)
tpr_uni = len(np.intersect1d(np.where(np.abs(fit_uni.coefs) > 1e-8)[0], true_indices)) / len(true_indices)
print(f"Test MSE: {mse_uni:.3f}")
print(f"Selected variables: {selected_uni}")
print(f"TPR (True Positive Rate): {tpr_uni:.3f}")
print(f"Best lambda: {cv_result_uni.best_lmda:.6f}")

# Test 2: XLasso - Soft Constraint only (no adaptive, no group)
print("\n=== 2. XLasso - Soft Constraint ===")
cv_result_soft = cv_uni(
    X_train, y_train, family="gaussian", n_folds=5,
    adaptive_weighting=False,
    enable_group_constraint=False,
    alpha=0.0,
    beta=1.0,
    backend="numba",
    verbose=False
)
fit_soft = extract_cv(cv_result_soft)
y_pred_soft = X_test @ fit_soft.coefs + fit_soft.intercept
mse_soft = np.mean((y_pred_soft - y_test) ** 2)
selected_soft = np.sum(np.abs(fit_soft.coefs) > 1e-8)
tpr_soft = len(np.intersect1d(np.where(np.abs(fit_soft.coefs) > 1e-8)[0], true_indices)) / len(true_indices)
print(f"Test MSE: {mse_soft:.3f}")
print(f"Selected variables: {selected_soft}")
print(f"TPR: {tpr_soft:.3f}")
print(f"Best lambda: {cv_result_soft.best_lmda:.6f}")

# Test 3: XLasso - Soft + Adaptive weighting
print("\n=== 3. XLasso - Soft + Adaptive ===")
cv_result_adapt = cv_uni(
    X_train, y_train, family="gaussian", n_folds=5,
    adaptive_weighting=True,
    enable_group_constraint=False,
    alpha=1.0,
    beta=1.0,
    backend="numba",
    verbose=False
)
fit_adapt = extract_cv(cv_result_adapt)
y_pred_adapt = X_test @ fit_adapt.coefs + fit_adapt.intercept
mse_adapt = np.mean((y_pred_adapt - y_test) ** 2)
selected_adapt = np.sum(np.abs(fit_adapt.coefs) > 1e-8)
tpr_adapt = len(np.intersect1d(np.where(np.abs(fit_adapt.coefs) > 1e-8)[0], true_indices)) / len(true_indices)
print(f"Test MSE: {mse_adapt:.3f}")
print(f"Selected variables: {selected_adapt}")
print(f"TPR: {tpr_adapt:.3f}")
print(f"Best lambda: {cv_result_adapt.best_lmda:.6f}")

# Test 4: XLasso - Soft + Group constraint
print("\n=== 4. XLasso - Soft + Group ===")
cv_result_group = cv_uni(
    X_train, y_train, family="gaussian", n_folds=5,
    adaptive_weighting=False,
    enable_group_constraint=True,
    alpha=0.0,
    beta=1.0,
    group_penalty=2.0,
    backend="numba",
    verbose=False
)
fit_group = extract_cv(cv_result_group)
y_pred_group = X_test @ fit_group.coefs + fit_group.intercept
mse_group = np.mean((y_pred_group - y_test) ** 2)
selected_group = np.sum(np.abs(fit_group.coefs) > 1e-8)
tpr_group = len(np.intersect1d(np.where(np.abs(fit_group.coefs) > 1e-8)[0], true_indices)) / len(true_indices)
print(f"Test MSE: {mse_group:.3f}")
print(f"Selected variables: {selected_group}")
print(f"TPR: {tpr_group:.3f}")
print(f"Best lambda: {cv_result_group.best_lmda:.6f}")

# Test 5: XLasso - Full (Soft + Adaptive + Group)
print("\n=== 5. XLasso - Full Version ===")
cv_result_full = cv_uni(
    X_train, y_train, family="gaussian", n_folds=5,
    adaptive_weighting=True,
    enable_group_constraint=True,
    alpha=1.0,
    beta=1.0,
    group_penalty=2.0,
    backend="numba",
    verbose=False
)
fit_full = extract_cv(cv_result_full)
y_pred_full = X_test @ fit_full.coefs + fit_full.intercept
mse_full = np.mean((y_pred_full - y_test) ** 2)
selected_full = np.sum(np.abs(fit_full.coefs) > 1e-8)
tpr_full = len(np.intersect1d(np.where(np.abs(fit_full.coefs) > 1e-8)[0], true_indices)) / len(true_indices)
print(f"Test MSE: {mse_full:.3f}")
print(f"Selected variables: {selected_full}")
print(f"TPR: {tpr_full:.3f}")
print(f"Best lambda: {cv_result_full.best_lmda:.6f}")

# Summary table
print("\n" + "="*80)
print("SUMMARY COMPARISON (after lambda fix)")
print("="*80)
print(f"{'Method':<25} {'MSE':>8} {'TPR':>8} {'Selected':>10}")
print("-"*80)
print(f"{'UniLasso (baseline)':<25} {mse_uni:>8.2f} {tpr_uni:>8.2f} {selected_uni:>10d}")
print(f"{'Soft Constraint':<25} {mse_soft:>8.2f} {tpr_soft:>8.2f} {selected_soft:>10d}")
print(f"{'Soft + Adaptive':<25} {mse_adapt:>8.2f} {tpr_adapt:>8.2f} {selected_adapt:>10d}")
print(f"{'Soft + Group':<25} {mse_group:>8.2f} {tpr_group:>8.2f} {selected_group:>10d}")
print(f"{'Full XLasso':<25} {mse_full:>8.2f} {tpr_full:>8.2f} {selected_full:>10d}")
