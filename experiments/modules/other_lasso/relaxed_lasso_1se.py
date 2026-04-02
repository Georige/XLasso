"""
Relaxed Lasso with 1-SE Rule Implementation

Stage 1: LassoCV (1-SE) 进行严格的特征筛选
Stage 2: OLS 对选中的特征进行无偏估计（彻底去偏）

支持回归 (family="gaussian") 和二分类 (family="binomial")。
"""
import numpy as np
from sklearn.linear_model import LassoCV, Lasso, LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from joblib import Parallel, delayed
import warnings


class RelaxedLassoCV1SE(BaseEstimator, ClassifierMixin):
    """
    基于 1-SE 规则的 Relaxed Lasso (Lasso-OLS 极端去偏版)

    Stage 1: LassoCV (1-SE) 进行严格的特征筛选
    Stage 2: OLS 对选中的特征进行无偏估计（回归）或 LogisticRegression（分类）

    参数:
        cv: 交叉验证折数
        random_state: 随机种子
        eps: alpha 路径长度参数
        n_alphas: alpha 候选数量
        verbose: 是否打印详细信息
        standardize: 是否标准化特征
        fit_intercept: 是否拟合截距
        family: 模型家族，"gaussian"(回归) 或 "binomial"(二分类)
        scoring: 分类的评分指标，"roc_auc" 或 "log_loss"
    """

    def __init__(
        self,
        cv: int = 5,
        random_state: int = 42,
        eps: float = 1e-3,
        n_alphas: int = 100,
        verbose: bool = False,
        standardize: bool = True,
        fit_intercept: bool = True,
        family: str = "gaussian",
        scoring: str = "roc_auc",
        n_jobs: int = -1,
    ):
        self.cv = cv
        self.random_state = random_state
        self.eps = eps
        self.n_alphas = n_alphas
        self.verbose = verbose
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.family = family
        self.scoring = scoring
        self.n_jobs = n_jobs

        # 拟合后填充的属性
        self.coef_: np.ndarray = None
        self.intercept_: float = None
        self.alpha_1se_: float = None
        self.support_: np.ndarray = None
        self.n_selected_: int = None

    def fit(self, X, y, cv_splits=None):
        """
        使用 Relaxed Lasso (1-SE) 拟合模型。

        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标向量 (n_samples,)
            cv_splits: list of tuples, optional
                Pre-generated CV splits (list of (train_idx, val_idx) tuples).
                If provided, uses these splits instead of creating new KFold.

        返回:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if cv_splits is not None:
            self._fit_with_splits(X, y, cv_splits)
        else:
            self._fit_internal_cv(X, y)

        return self

    def _fit_with_splits(self, X, y, cv_splits):
        """使用外部提供的 cv_splits 进行 1-SE 寻优。
        Fold 串行，fold 内部 alpha 并行 (joblib)。
        """
        n_folds = len(cv_splits)
        is_classification = self.family.lower() == "binomial"

        if is_classification:
            y_lr = np.where(y > 0.5, 1, -1).astype(np.float64)

        if self.verbose:
            print(f"Relaxed Lasso Stage 1: 运行 {'LogisticRegression' if is_classification else 'Lasso'}CV 寻找 1-SE 阈值 (external splits, {n_folds} folds)...")

        # Compute alpha grid similar to LassoCV
        lasso_tmp = Lasso(alpha=0.001, max_iter=20000, random_state=self.random_state)
        lasso_tmp.fit(X, y)
        alpha_max = np.abs(lasso_tmp.coef_).max() if np.abs(lasso_tmp.coef_).max() > 0 else 1.0
        alpha_min = alpha_max * self.eps
        alphas = np.linspace(alpha_max, alpha_min, self.n_alphas)

        # score_path shape: (n_alphas, n_folds)
        score_path = np.zeros((len(alphas), n_folds))

        # === Fold 串行，alpha 并行 ===
        def _eval_gaussian(alpha, X_tr, y_tr, X_va, y_va):
            model = Lasso(alpha=alpha, max_iter=20000, random_state=self.random_state)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            return float(np.mean((y_va - y_pred) ** 2))

        def _eval_classifier(alpha, X_tr, y_tr, X_va, y_va):
            # alpha in Lasso sense -> C for LogisticRegression
            C = 1.0 / alpha if alpha > 0 else 1e10
            model = LogisticRegression(C=C, penalty='l1', solver='liblinear',
                                      fit_intercept=self.fit_intercept, max_iter=5000, tol=1e-4)
            model.fit(X_tr, y_tr)
            if self.scoring == "log_loss":
                proba_va = model.predict_proba(X_va)[:, 1]
                # Convert y_va from {1,-1} to {1,0} for log_loss
                y_va_binary = np.where(y_va > 0, 1, 0)
                return float(log_loss(y_va_binary, proba_va))
            else:  # roc_auc
                proba_va = model.predict_proba(X_va)[:, 1]
                return -float(roc_auc_score(y_va, proba_va))  # negative because we minimize

        eval_fn = _eval_classifier if is_classification else _eval_gaussian

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            if is_classification:
                y_train_lr = y_lr[train_idx]
                y_val_lr = y_lr[val_idx]
            else:
                y_train_lr = y_train
                y_val_lr = y_val

            # 并行：对当前 fold 的所有 alpha 候选求 score
            score_results = Parallel(n_jobs=self.n_jobs, prefer="threads", verbose=0)(
                delayed(eval_fn)(alpha, X_train_s, y_train_lr, X_val_s, y_val_lr)
                for alpha in alphas
            )
            for alpha_idx, score_val in enumerate(score_results):
                score_path[alpha_idx, fold_idx] = score_val

        # 1-SE logic
        mean_score = score_path.mean(axis=1)
        se_score = score_path.std(axis=1) / np.sqrt(n_folds)

        if is_classification and self.scoring == "roc_auc":
            # Larger is better
            best_idx = np.argmax(mean_score)
            best_score = mean_score[best_idx]
            best_se = se_score[best_idx]
            threshold_1se = best_score - best_se
            candidates_mask = mean_score >= threshold_1se
        elif is_classification and self.scoring == "log_loss":
            # Smaller is better
            best_idx = np.argmin(mean_score)
            best_score = mean_score[best_idx]
            best_se = se_score[best_idx]
            threshold_1se = best_score + best_se
            candidates_mask = mean_score <= threshold_1se
        else:
            # MSE: smaller is better
            best_idx = np.argmin(mean_score)
            best_score = mean_score[best_idx]
            best_se = se_score[best_idx]
            threshold_1se = best_score + best_se
            candidates_mask = mean_score <= threshold_1se

        candidate_indices = np.where(candidates_mask)[0]

        if len(candidate_indices) > 0:
            max_check = min(20, len(candidate_indices))
            sampled_indices = candidate_indices[:max_check]

            n_nonzero_candidates = []
            for idx in sampled_indices:
                alpha = alphas[idx]
                if is_classification:
                    C = 1.0 / alpha if alpha > 0 else 1e10
                    model_tmp = LogisticRegression(C=C, penalty='l1', solver='liblinear',
                                                    fit_intercept=self.fit_intercept, max_iter=5000, tol=1e-4)
                    model_tmp.fit(X, y_lr)
                else:
                    model_tmp = Lasso(alpha=alpha, max_iter=20000, random_state=self.random_state)
                    model_tmp.fit(X, y)
                n_nonzero_candidates.append(np.sum(model_tmp.coef_ != 0))

            best_local_idx = np.argmin(n_nonzero_candidates)
            alpha_1se = alphas[sampled_indices[best_local_idx]]
        else:
            alpha_1se = alphas[best_idx]

        self.alpha_1se_ = alpha_1se

        if self.verbose:
            print(f"Stage 1: 全局最小 score = {best_score:.4f}, 1-SE threshold = {threshold_1se:.4f}")

        # Stage 1 final: fit on all data with alpha_1se
        if is_classification:
            C_1se = 1.0 / alpha_1se if alpha_1se > 0 else 1e10
            scaler_X = StandardScaler()
            X_s = scaler_X.fit_transform(X)
            stage1_final = LogisticRegression(C=C_1se, penalty='l1', solver='liblinear',
                                              fit_intercept=self.fit_intercept, max_iter=5000, tol=1e-4)
            stage1_final.fit(X_s, y_lr)
            support_mask = (stage1_final.coef_[0] != 0)
            self.intercept_ = stage1_final.intercept_[0]
            # Store for predict_proba later
            self.scaler_X_ = scaler_X
            self.stage1_coef_ = stage1_final.coef_[0]
        else:
            lasso_stage1 = Lasso(alpha=alpha_1se, max_iter=20000, random_state=self.random_state)
            lasso_stage1.fit(X, y)
            support_mask = (lasso_stage1.coef_ != 0)
            self.intercept_ = lasso_stage1.intercept_

        self.support_ = support_mask
        self.n_selected_ = np.sum(support_mask)

        if self.verbose:
            print(f"Stage 1 结束. 1-SE Alpha: {alpha_1se:.6f}, 选出特征数: {self.n_selected_}")

        # Stage 2: debiasing
        self.coef_ = np.zeros(X.shape[1])

        if self.n_selected_ == 0:
            if self.verbose:
                print("警告：Stage 1 砍掉了所有特征！退化为常数截距模型。")
            self.intercept_ = np.mean(y) if not is_classification else self.intercept_
            return self

        X_active = X[:, support_mask]
        if is_classification:
            scaler_active = StandardScaler()
            X_active_s = scaler_active.fit_transform(X_active)
            stage2_final = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                                              fit_intercept=self.fit_intercept, max_iter=5000, tol=1e-4)
            stage2_final.fit(X_active_s, y_lr)
            self.coef_[support_mask] = stage2_final.coef_[0]
            # Adjust intercept for the active-only scaler
            self.intercept_ = stage2_final.intercept_[0] - np.sum(stage2_final.coef_[0] * scaler_active.mean_)
        else:
            ols = LinearRegression()
            ols.fit(X_active, y)
            self.coef_[support_mask] = ols.coef_
            self.intercept_ = ols.intercept_

        if self.verbose:
            print(f"Stage 2 结束. {'LogisticRegression' if is_classification else 'OLS'} 系数已填回 {self.n_selected_} 个选中特征")

    def _fit_internal_cv(self, X, y):
        """使用 sklearn LassoCV 的内部 CV (原始行为)。"""
        is_classification = self.family.lower() == "binomial"

        if is_classification:
            y_lr = np.where(y > 0.5, 1, -1).astype(np.float64)

        if self.verbose:
            print(f"Relaxed Lasso Stage 1: 运行 {'LogisticRegression' if is_classification else 'Lasso'}CV 寻找 1-SE 阈值...")

        if is_classification:
            # Manual CV for LogisticRegression with L1 penalty
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            cv_splits = list(kfold.split(X))

            lasso_tmp = Lasso(alpha=0.001, max_iter=20000, random_state=self.random_state)
            lasso_tmp.fit(X, y)
            alpha_max = np.abs(lasso_tmp.coef_).max() if np.abs(lasso_tmp.coef_).max() > 0 else 1.0
            alpha_min = alpha_max * self.eps
            alphas = np.linspace(alpha_max, alpha_min, self.n_alphas)

            score_path = np.zeros((len(alphas), self.cv))

            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train_lr = y_lr[train_idx]
                y_val_lr = y_lr[val_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)

                for alpha_idx, alpha in enumerate(alphas):
                    C = 1.0 / alpha if alpha > 0 else 1e10
                    model = LogisticRegression(C=C, penalty='l1', solver='liblinear',
                                               fit_intercept=self.fit_intercept, max_iter=5000, tol=1e-4)
                    model.fit(X_train_s, y_train_lr)
                    if self.scoring == "log_loss":
                        proba_va = model.predict_proba(X_val_s)[:, 1]
                        y_val_binary = np.where(y_val_lr > 0, 1, 0)
                        score_path[alpha_idx, fold_idx] = log_loss(y_val_binary, proba_va)
                    else:
                        proba_va = model.predict_proba(X_val_s)[:, 1]
                        score_path[alpha_idx, fold_idx] = -roc_auc_score(y_val_lr, proba_va)

            mean_score = score_path.mean(axis=1)
            se_score = score_path.std(axis=1) / np.sqrt(self.cv)

            if self.scoring == "roc_auc":
                best_idx = np.argmax(mean_score)
                best_score = mean_score[best_idx]
                best_se = se_score[best_idx]
                threshold_1se = best_score - best_se
                candidates_mask = mean_score >= threshold_1se
            else:
                best_idx = np.argmin(mean_score)
                best_score = mean_score[best_idx]
                best_se = se_score[best_idx]
                threshold_1se = best_score + best_se
                candidates_mask = mean_score <= threshold_1se

            candidate_indices = np.where(candidates_mask)[0]

            if len(candidate_indices) > 0:
                max_check = min(20, len(candidate_indices))
                sampled_indices = candidate_indices[:max_check]
                n_nonzero_candidates = []
                for idx in sampled_indices:
                    alpha = alphas[idx]
                    C = 1.0 / alpha if alpha > 0 else 1e10
                    model_tmp = LogisticRegression(C=C, penalty='l1', solver='liblinear',
                                                   fit_intercept=self.fit_intercept, max_iter=5000, tol=1e-4)
                    model_tmp.fit(X, y_lr)
                    n_nonzero_candidates.append(np.sum(model_tmp.coef_ != 0))
                best_local_idx = np.argmin(n_nonzero_candidates)
                alpha_1se = alphas[sampled_indices[best_local_idx]]
            else:
                alpha_1se = alphas[best_idx]

            self.alpha_1se_ = alpha_1se

            if self.verbose:
                print(f"Stage 1: 全局最小 score = {best_score:.4f}, 1-SE threshold = {threshold_1se:.4f}")

            C_1se = 1.0 / alpha_1se if alpha_1se > 0 else 1e10
            scaler_X = StandardScaler()
            X_s = scaler_X.fit_transform(X)
            stage1_final = LogisticRegression(C=C_1se, penalty='l1', solver='liblinear',
                                              fit_intercept=self.fit_intercept, max_iter=5000, tol=1e-4)
            stage1_final.fit(X_s, y_lr)
            support_mask = (stage1_final.coef_[0] != 0)
            self.support_ = support_mask
            self.n_selected_ = np.sum(support_mask)
            self.scaler_X_ = scaler_X
            self.stage1_coef_ = stage1_final.coef_[0]
            self.intercept_ = stage1_final.intercept_[0]

            if self.verbose:
                print(f"Stage 1 结束. 1-SE Alpha: {alpha_1se:.6f}, 选出特征数: {self.n_selected_}")

            self.coef_ = np.zeros(X.shape[1])
            if self.n_selected_ == 0:
                if self.verbose:
                    print("警告：Stage 1 砍掉了所有特征！退化为常数截距模型。")
                return self

            X_active = X[:, support_mask]
            scaler_active = StandardScaler()
            X_active_s = scaler_active.fit_transform(X_active)
            stage2_final = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                                              fit_intercept=self.fit_intercept, max_iter=5000, tol=1e-4)
            stage2_final.fit(X_active_s, y_lr)
            self.coef_[support_mask] = stage2_final.coef_[0]
            self.intercept_ = stage2_final.intercept_[0] - np.sum(stage2_final.coef_[0] * scaler_active.mean_)

            if self.verbose:
                print(f"Stage 2 结束. LogisticRegression 系数已填回 {self.n_selected_} 个选中特征")
        else:
            # Gaussian regression: original logic
            precompute = False if X.shape[0] < X.shape[1] else 'auto'
            lasso_cv = LassoCV(
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=-1,
                max_iter=20000,
                eps=self.eps,
                n_alphas=self.n_alphas,
                precompute=precompute
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lasso_cv.fit(X, y)

            mean_mse = lasso_cv.mse_path_.mean(axis=1)
            se_mse = lasso_cv.mse_path_.std(axis=1) / np.sqrt(self.cv)

            best_idx = np.argmin(mean_mse)
            best_mse = mean_mse[best_idx]
            best_se = se_mse[best_idx]

            threshold_1se = best_mse + best_se

            candidates_mask = mean_mse <= threshold_1se
            candidate_indices = np.where(candidates_mask)[0]

            if len(candidate_indices) > 0:
                max_check = min(20, len(candidate_indices))
                sampled_indices = candidate_indices[:max_check]

                n_nonzero_candidates = []
                for idx in sampled_indices:
                    alpha = lasso_cv.alphas_[idx]
                    model_tmp = Lasso(alpha=alpha, max_iter=20000, random_state=self.random_state)
                    model_tmp.fit(X, y)
                    n_nonzero_candidates.append(np.sum(model_tmp.coef_ != 0))

                best_local_idx = np.argmin(n_nonzero_candidates)
                alpha_1se = lasso_cv.alphas_[sampled_indices[best_local_idx]]
            else:
                alpha_1se = lasso_cv.alphas_[best_idx]

            self.alpha_1se_ = alpha_1se

            if self.verbose:
                print(f"Stage 1: 全局最小 MSE = {best_mse:.4f}, 1-SE MSE = {threshold_1se:.4f}")

            lasso_stage1 = Lasso(
                alpha=alpha_1se,
                max_iter=20000,
                random_state=self.random_state
            )
            lasso_stage1.fit(X, y)

            support_mask = (lasso_stage1.coef_ != 0)
            self.support_ = support_mask
            self.n_selected_ = np.sum(support_mask)

            if self.verbose:
                print(f"Stage 1 结束. 1-SE Alpha: {alpha_1se:.6f}, 选出特征数: {self.n_selected_}")

            self.coef_ = np.zeros(X.shape[1])

            if self.n_selected_ == 0:
                if self.verbose:
                    print("警告：Stage 1 砍掉了所有特征！退化为常数截距模型。")
                self.intercept_ = np.mean(y)
                return self

            X_active = X[:, support_mask]
            ols = LinearRegression()
            ols.fit(X_active, y)

            self.coef_[support_mask] = ols.coef_
            self.intercept_ = ols.intercept_

            if self.verbose:
                print(f"Stage 2 结束. OLS 系数已填回 {self.n_selected_} 个选中特征")

    def predict(self, X):
        """使用拟合的模型进行预测。"""
        if self.family.lower() == "binomial":
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        """预测概率（仅在 family="binomial" 时可用）。"""
        if self.family.lower() != "binomial":
            raise ValueError("predict_proba 仅在 family='binomial' 时可用")
        if self.n_selected_ == 0:
            # All zero: return prior probability
            proba = np.full((X.shape[0], 2), 0.5)
            return proba
        z = X @ self.coef_ + self.intercept_
        proba_1 = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - proba_1, proba_1])

    def score(self, X, y):
        """计算准确率（分类）或 R²（回归）。"""
        if self.family.lower() == "binomial":
            from sklearn.metrics import accuracy_score
            return accuracy_score(y, self.predict(X))
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
