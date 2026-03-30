"""
Test: Min-Anchored vs Mean-Anchored Weight Normalization
========================================================
对比 AdaptiveFlippedLassoCV 在 min_w vs mean_w 归一化下的表现
使用 exp1 数据配置：n=300, p=500, rho=0.5, first 20 vars β=1.0
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.modules import DataGenerator, MetricCalculator
from experiments.modules.NLasso.adaptive_flipped_lasso.base import AdaptiveFlippedLassoCV


class AdaptiveFlippedLassoCV_MeanAnchored(AdaptiveFlippedLassoCV):
    """
    使用 Mean-Anchored 归一化的 AdaptiveFlippedLassoCV
    所有权重除以 mean_w，而非 min_w
    """

    def fit(self, X, y, sample_weight=None, cv_splits=None):
        # 完全复用父类 fit，只在权重计算处覆盖
        # 由于 fit() 内部分阶段，我们需要在调用前修改行为
        # 最简单的方式：临时修改实例属性来控制行为
        raise NotImplementedError("Use the standalone function approach instead")


def run_min_anchored(X, y, config, cv_splits):
    """运行 Min-Anchored 版本"""
    model = AdaptiveFlippedLassoCV(
        lambda_ridge_list=config.get('lambda_ridge_list', (1.0, 5.0, 10.0, 50.0)),
        gamma_list=config.get('gamma_list', (0.3, 0.5, 0.7, 1.0)),
        cv=config.get('cv_folds', 10),
        alpha_min_ratio=config.get('alpha_min_ratio', 0.0001),
        n_alpha=config.get('n_alpha', 30),
        max_iter=config.get('max_iter', 5000),
        tol=config.get('tol', 0.0001),
        standardize=True,
        fit_intercept=True,
        random_state=config.get('random_state', 2026),
        verbose=False,
        use_post_ols_debiasing=False,
        auto_tune_collinearity=True,
        weight_clip_max=config.get('weight_clip_max', 100.0),
    )
    model.fit(X, y, cv_splits=cv_splits)
    return model


def run_mean_anchored(X, y, config, cv_splits):
    """
    运行 Mean-Anchored 版本
    通过 monkey-patch 在 fit 内部临时替换权重计算逻辑
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import RidgeCV, Lasso, lasso_path
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.validation import check_X_y
    import numpy as np

    eps = 1e-5

    # 复制父类 fit 的完整逻辑，但将 min_w 改为 mean_w
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], dtype=np.float64,
                     copy=False, ensure_2d=True, ensure_min_samples=2, ensure_min_features=2)
    n_features = X.shape[1]

    # Auto-detect collinearity
    from experiments.modules.NLasso.adaptive_flipped_lasso.base import _detect_collinearity_mode
    data_mode, energy_ratio = _detect_collinearity_mode(X)
    gamma_list = config.get('gamma_list', (0.3, 0.5, 0.7, 1.0))
    if data_mode == "dense_collinear":
        gamma_list = (1.0, 2.0, 3.0)
    else:
        gamma_list = (0.3, 0.5, 1.0)

    standardize = True
    if standardize:
        scaler = StandardScaler(copy=False)
        X = scaler.fit_transform(X)

    n = X.shape[0]
    n_gamma = len(gamma_list)

    # CV splits
    if cv_splits is not None:
        n_folds = len(cv_splits)
        splits = cv_splits
    else:
        n_folds = config.get('cv_folds', 10)
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=config.get('random_state', 2026))
        splits = list(kfold.split(X))

    error_matrix = np.full((n_gamma, config.get('n_alpha', 30), n_folds), np.inf)
    nselected_matrix = np.zeros((n_gamma, config.get('n_alpha', 30), n_folds))

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        ridge_cv = RidgeCV(alphas=config.get('lambda_ridge_list', (1.0, 5.0, 10.0, 50.0)), cv=3)
        ridge_cv.fit(X_tr, y_tr)
        beta_ridge_fold = ridge_cv.coef_

        signs_fold = np.sign(beta_ridge_fold)
        signs_fold[signs_fold == 0] = 1.0

        for gamma_idx, gamma in enumerate(gamma_list):
            # ========== MEAN-ANCHORED (修改点) ==========
            raw_weights = 1.0 / (np.abs(beta_ridge_fold) + eps) ** gamma
            mean_w = np.mean(raw_weights)  # 用 mean 而非 min
            w_normalized = raw_weights / mean_w  # 除以均值
            clip_max = config.get('weight_clip_max', 100.0)
            weights = np.clip(w_normalized, 1.0, clip_max)
            # ===========================================

            X_adaptive_tr = (X_tr * signs_fold) / weights
            X_adaptive_va = (X_va * signs_fold) / weights

            alpha_max = np.max(np.abs(X_adaptive_tr.T @ y_tr)) / len(y_tr)
            alpha_min = alpha_max * config.get('alpha_min_ratio', 0.0001)
            alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), config.get('n_alpha', 30))[::-1]

            _, coefs_path, _ = lasso_path(
                X_adaptive_tr, y_tr,
                alphas=alphas,
                positive=True,
                max_iter=config.get('max_iter', 5000),
                tol=config.get('tol', 0.0001),
            )

            preds = X_adaptive_va @ coefs_path
            mse_path = np.mean((y_va[:, np.newaxis] - preds) ** 2, axis=0)
            error_matrix[gamma_idx, :, fold_idx] = mse_path
            nselected_path = np.sum(coefs_path != 0, axis=0)
            nselected_matrix[gamma_idx, :, fold_idx] = nselected_path

    # Stage 2: 1-SE rule
    mean_error = np.mean(error_matrix, axis=2)
    std_error = np.std(error_matrix, axis=2) / np.sqrt(n_folds)
    mean_nselected = np.mean(nselected_matrix, axis=2)

    min_mse = np.min(mean_error)
    min_mse_idx = np.unravel_index(np.argmin(mean_error), mean_error.shape)
    min_std = std_error[min_mse_idx]
    threshold = min_mse + min_std

    candidates_mask = mean_error <= threshold
    if not np.any(candidates_mask):
        best_gamma_idx, best_alpha_idx = min_mse_idx
    else:
        masked_nselected = np.where(candidates_mask, mean_nselected, np.inf)
        best_flat_idx = np.argmin(masked_nselected)
        best_gamma_idx, best_alpha_idx = np.unravel_index(best_flat_idx, mean_error.shape)

    # High-dimensional fallback
    if n_features > n * 2:
        best_gamma_idx, best_alpha_idx = np.unravel_index(np.argmin(mean_error), mean_error.shape)

    best_gamma = gamma_list[best_gamma_idx]

    # Stage 3: Full data fit with mean anchoring
    ridge_final = RidgeCV(alphas=config.get('lambda_ridge_list', (1.0, 5.0, 10.0, 50.0)), cv=config.get('cv_folds', 10))
    ridge_final.fit(X, y)
    beta_ridge = ridge_final.coef_
    best_lambda_ridge = ridge_final.alpha_

    signs_final = np.sign(beta_ridge)
    signs_final[signs_final == 0] = 1.0

    # ========== MEAN-ANCHORED (修改点) ==========
    raw_weights_final = 1.0 / (np.abs(beta_ridge) + eps) ** best_gamma
    mean_w_final = np.mean(raw_weights_final)
    weights_final = np.clip(raw_weights_final / mean_w_final, 1.0, config.get('weight_clip_max', 100.0))
    # ===========================================

    X_adaptive_final = (X * signs_final) / weights_final

    alpha_max_tmp = np.max(np.abs(X_adaptive_final.T @ y)) / len(y)
    alpha_min_tmp = alpha_max_tmp * config.get('alpha_min_ratio', 0.0001)
    alphas_final = np.logspace(np.log10(alpha_min_tmp), np.log10(alpha_max_tmp), config.get('n_alpha', 30))[::-1]
    best_alpha = alphas_final[best_alpha_idx]

    lasso_final = Lasso(alpha=best_alpha, positive=True, fit_intercept=True,
                        max_iter=config.get('max_iter', 5000), tol=config.get('tol', 0.0001),
                        random_state=config.get('random_state', 2026))
    lasso_final.fit(X_adaptive_final, y)

    coef_final = (lasso_final.coef_ / weights_final) * signs_final

    if standardize:
        coef_final = coef_final / scaler.scale_
        intercept_final = scaler.mean_[0] - np.sum(coef_final * scaler.mean_)
    else:
        intercept_final = lasso_final.intercept_

    # 构建结果对象
    class Result:
        pass
    result = Result()
    result.coef_ = coef_final
    result.intercept_ = intercept_final
    result.best_gamma_ = best_gamma
    result.best_alpha_ = best_alpha
    result.best_lambda_ridge_ = best_lambda_ridge
    result.cv_score_ = -mean_error[best_gamma_idx, best_alpha_idx]
    result.is_fitted_ = True
    result.n_features_in_ = n_features
    result.scaler_ = scaler
    result.weights_ = weights_final
    result.signs_ = signs_final
    result.beta_ridge_ = beta_ridge
    return result


def predict(model, X):
    if hasattr(model, 'scaler_') and model.scaler_ is not None:
        X = model.scaler_.transform(X)
    return X @ model.coef_ + model.intercept_


def main():
    from sklearn.model_selection import KFold

    # Exp1 配置
    config = {
        'n_samples': 300,
        'n_features': 500,
        'n_nonzero': 20,
        'correlation_type': 'experiment1',
        'rho': 0.5,
        'random_state': 42,
        'sigma': 1.0,
        'n_repeats': 5,
        'cv_folds': 10,
        'lambda_ridge_list': (1.0, 5.0, 10.0, 50.0),
        'gamma_list': (0.3, 0.5, 0.7, 1.0),
        'alpha_min_ratio': 0.0001,
        'n_alpha': 30,
        'max_iter': 5000,
        'tol': 0.0001,
        'weight_clip_max': 100.0,
    }

    n_repeats = config['n_repeats']
    n_folds = config['cv_folds']
    random_state_base = config['random_state']

    results = []

    for repeat in range(n_repeats):
        print(f"\n{'='*60}")
        print(f"Repeat {repeat+1}/{n_repeats}")
        print(f"{'='*60}")

        # 生成数据
        gen = DataGenerator(random_state=random_state_base + repeat)
        X, y, beta_true = gen.generate(
            n_samples=config['n_samples'],
            n_features=config['n_features'],
            n_nonzero=config['n_nonzero'],
            sigma=config['sigma'],
            correlation_type=config['correlation_type'],
            rho=config['rho'],
        )

        # 80/20 分割
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            np.arange(len(y)), test_size=0.2,
            random_state=random_state_base + repeat, shuffle=True
        )
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 生成共享的 CV splits
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state_base + repeat)
        cv_splits = list(kfold.split(X_train))

        # ========== Min-Anchored ==========
        print(f"\n[Repeat {repeat}] Running Min-Anchored...")
        t0 = time.time()
        model_min = run_min_anchored(X_train, y_train, config, cv_splits)
        t_min = time.time() - t0

        y_pred_min = predict(model_min, X_val)
        metrics_min = MetricCalculator().calculate(y_val, y_pred_min, beta_true, model_min.coef_)
        metrics_min['repeat'] = repeat
        metrics_min['norm_type'] = 'min_anchored'
        metrics_min['train_time'] = t_min
        metrics_min['best_gamma'] = getattr(model_min, 'best_gamma_', np.nan)
        metrics_min['best_alpha'] = getattr(model_min, 'best_alpha_', np.nan)
        metrics_min['best_lambda_ridge'] = getattr(model_min, 'best_lambda_ridge_', np.nan)
        metrics_min['n_selected'] = np.sum(np.abs(model_min.coef_) > 1e-6)
        print(f"  Min-Anchored: F1={metrics_min['f1']:.4f}, TPR={metrics_min['tpr']:.4f}, n_sel={metrics_min['n_selected']}, gamma={metrics_min['best_gamma']}")

        # ========== Mean-Anchored ==========
        print(f"[Repeat {repeat}] Running Mean-Anchored...")
        t0 = time.time()
        model_mean = run_mean_anchored(X_train, y_train, config, cv_splits)
        t_mean = time.time() - t0

        y_pred_mean = predict(model_mean, X_val)
        metrics_mean = MetricCalculator().calculate(y_val, y_pred_mean, beta_true, model_mean.coef_)
        metrics_mean['repeat'] = repeat
        metrics_mean['norm_type'] = 'mean_anchored'
        metrics_mean['train_time'] = t_mean
        metrics_mean['best_gamma'] = getattr(model_mean, 'best_gamma_', np.nan)
        metrics_mean['best_alpha'] = getattr(model_mean, 'best_alpha_', np.nan)
        metrics_mean['best_lambda_ridge'] = getattr(model_mean, 'best_lambda_ridge_', np.nan)
        metrics_mean['n_selected'] = np.sum(np.abs(model_mean.coef_) > 1e-6)
        print(f"  Mean-Anchored: F1={metrics_mean['f1']:.4f}, TPR={metrics_mean['tpr']:.4f}, n_sel={metrics_mean['n_selected']}, gamma={metrics_mean['best_gamma']}")

        # 对比
        print(f"  Delta F1: {metrics_mean['f1'] - metrics_min['f1']:+.4f}")

        results.append(metrics_min)
        results.append(metrics_mean)

    # 汇总
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for norm_type in ['min_anchored', 'mean_anchored']:
        subset = df[df['norm_type'] == norm_type]
        print(f"\n{norm_type}:")
        print(f"  F1:   {subset['f1'].mean():.4f} ± {subset['f1'].std():.4f}")
        print(f"  TPR:  {subset['tpr'].mean():.4f} ± {subset['tpr'].std():.4f}")
        print(f"  FDR:  {subset['fdr'].mean():.4f} ± {subset['fdr'].std():.4f}")
        print(f"  MSE:  {subset['mse'].mean():.4f} ± {subset['mse'].std():.4f}")
        print(f"  R2:   {subset['r2'].mean():.4f} ± {subset['r2'].std():.4f}")
        print(f"  n_sel: {subset['n_selected'].mean():.1f} ± {subset['n_selected'].std():.1f}")
        print(f"  gamma: {subset['best_gamma'].mode().iloc[0] if len(subset) > 0 else 'N/A'}")

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/home/lili/lyn/clear/NLasso/XLasso/experiments/results/afl补/norm_compare__{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'results.csv'}")
    return df


if __name__ == "__main__":
    main()