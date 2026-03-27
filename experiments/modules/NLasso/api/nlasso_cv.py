"""
API层：NLasso交叉验证自动调参版本
自动搜索最优超参数组合，支持CV-1-SE规则选择最稀疏模型
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from itertools import product
from .nlasso import NLasso, NLassoClassifier
from ..base import _DTYPE


class NLassoCV(BaseEstimator, RegressorMixin):
    """
    NLasso 交叉验证版本（回归）
    自动搜索最优超参数组合，支持 CV-1-SE 规则选择最稀疏模型
    """
    def __init__(
        self,
        # 超参数搜索空间
        param_grid: dict = None,
        # CV配置
        cv: int = 5,
        scoring: str = 'r2',
        n_jobs: int = -1,
        verbose: bool = False,
        random_state: int = 2026,
        use_1se: bool = True,
        # 其他固定参数
        **kwargs
    ):
        # 默认超参数搜索空间（paper推荐范围）
        if param_grid is None:
            self.param_grid = {
                'lambda_ridge': [1.0, 5.0, 10.0, 20.0, 50.0],
                'gamma': [0.1, 0.3, 0.5, 0.7, 1.0],
                's': [0.1, 0.5, 1.0, 2.0, 5.0],
                'group_threshold': [0.6, 0.7, 0.8, 0.9],
            }
        else:
            self.param_grid = param_grid

        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.use_1se = use_1se
        self.fixed_params = kwargs

        # 拟合后属性
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None, **fit_params):
        """
        拟合模型并执行交叉验证调参
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # 保存用于后续 1-SE 计算
        self._X_for_cv = X
        self._y_for_cv = y

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # 构建完整参数网格
        lambda_ridges = self.param_grid.get('lambda_ridge', [10.0])
        gammas = self.param_grid.get('gamma', [1.0])
        s_list = self.param_grid.get('s', [1.0])
        group_thresholds = self.param_grid.get('group_threshold', [0.8])

        param_combinations = list(product(lambda_ridges, gammas, s_list, group_thresholds))
        n_combinations = len(param_combinations)
        n_folds = self.cv

        if self.verbose:
            print(f"[NLassoCV] {n_combinations} param combos × {n_folds} folds = {n_combinations * n_folds} fits")

        # 存储每组参数的 fold-level 分数
        all_scores = np.full((n_combinations, n_folds), np.nan)

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        fold_idx = 0

        for train_idx, val_idx in kfold.split(X):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            for combo_idx, (lr, gamma, s, gt) in enumerate(param_combinations):
                try:
                    model = NLasso(
                        lambda_ridge=lr, gamma=gamma, s=s,
                        group_threshold=gt,
                        random_state=self.random_state,
                        verbose=False,
                        **self.fixed_params
                    )
                    model.fit(X_tr, y_tr)

                    y_pred = model.predict(X_va)
                    if self.scoring == 'r2':
                        score = r2_score(y_va, y_pred)
                    elif self.scoring == 'neg_mean_squared_error':
                        score = -mean_squared_error(y_va, y_pred)
                    else:
                        score = r2_score(y_va, y_pred)
                    all_scores[combo_idx, fold_idx] = score
                except Exception:
                    all_scores[combo_idx, fold_idx] = np.nan

            fold_idx += 1

        # 计算均值和标准误
        mean_scores = np.nanmean(all_scores, axis=1)
        std_scores = np.nanstd(all_scores, axis=1)
        se_scores = std_scores / np.sqrt(n_folds)

        # 保存 CV 结果
        self.cv_results_ = {
            'mean_test_score': mean_scores,
            'std_test_score': std_scores,
            'se_test_score': se_scores,
            'param_combinations': param_combinations,
            'all_scores': all_scores,
        }

        # 确定最优参数（min-MSE baseline）
        best_local_idx = np.nanargmax(mean_scores)
        best_score = mean_scores[best_local_idx]
        best_combo = param_combinations[best_local_idx]
        self.best_params_ = {
            'lambda_ridge': best_combo[0],
            'gamma': best_combo[1],
            's': best_combo[2],
            'group_threshold': best_combo[3],
        }
        self.best_score_ = best_score

        # 1-SE 规则：选择最稀疏的模型，其分数在 best - 1*SE 范围内
        if self.use_1se:
            best_combo_1se, best_score_1se = self._select_1se_params(
                mean_scores, se_scores, param_combinations
            )
            if self.verbose:
                print(f"[NLassoCV] Best (lambda_ridge, gamma, s, group_threshold): ({best_combo[0]}, {best_combo[1]}, {best_combo[2]}, {best_combo[3]})")
                print(f"[NLassoCV] 1-SE  (lambda_ridge, gamma, s, group_threshold): ({best_combo_1se[0]}, {best_combo_1se[1]}, {best_combo_1se[2]}, {best_combo_1se[3]})")
            self.best_params_ = {
                'lambda_ridge': best_combo_1se[0],
                'gamma': best_combo_1se[1],
                's': best_combo_1se[2],
                'group_threshold': best_combo_1se[3],
            }
            self.best_score_ = best_score_1se

        # 用最优参数在全量数据上终极拟合
        self.best_estimator_ = NLasso(
            **self.best_params_,
            random_state=self.random_state,
            verbose=self.verbose,
            **self.fixed_params
        )
        self.best_estimator_.fit(X, y)
        self.coef_ = self.best_estimator_.coef_
        self.intercept_ = self.best_estimator_.intercept_
        self.is_fitted_ = True

        return self

    def _select_1se_params(self, mean_scores, se_scores, param_combinations):
        """
        1-SE 规则：选择最稀疏的模型，其分数在 best - 1*SE 范围内

        在候选中选择非零系数数量最少的（最稀疏的）
        """
        best_idx = np.nanargmax(mean_scores)
        best_score = mean_scores[best_idx]
        threshold = best_score - se_scores[best_idx]

        # 找出所有通过 1-SE 检验的候选
        candidates_mask = mean_scores >= threshold
        candidate_indices = np.where(candidates_mask)[0]

        if len(candidate_indices) == 0:
            return param_combinations[best_idx], mean_scores[best_idx]

        # 计算每个候选的非零系数数量，选择最稀疏的
        n_nonzero_list = []
        for idx in candidate_indices:
            lr, gamma, s, gt = param_combinations[idx]
            model = NLasso(
                lambda_ridge=lr, gamma=gamma, s=s,
                group_threshold=gt,
                random_state=self.random_state,
                verbose=False,
                **self.fixed_params
            )
            model.fit(self._X_for_cv, self._y_for_cv)
            n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
            n_nonzero_list.append(n_nonzero)

        best_candidate_idx = candidate_indices[np.argmin(n_nonzero_list)]
        return param_combinations[best_candidate_idx], mean_scores[best_candidate_idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> float:
        """评分"""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.score(X, y, sample_weight=sample_weight)

    @property
    def coef_(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.coef_

    @property
    def intercept_(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.intercept_

    @property
    def groups_(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.groups_


class NLassoClassifierCV(BaseEstimator, ClassifierMixin):
    """
    NLasso 交叉验证版本（分类）
    自动搜索最优超参数组合，支持 CV-1-SE 规则选择最稀疏模型
    """
    def __init__(
        self,
        # 超参数搜索空间
        param_grid: dict = None,
        # CV配置
        cv: int = 5,
        scoring: str = 'accuracy',
        n_jobs: int = -1,
        verbose: bool = False,
        random_state: int = 2026,
        use_1se: bool = True,
        # 其他固定参数
        **kwargs
    ):
        # 默认超参数搜索空间（paper推荐范围）
        if param_grid is None:
            self.param_grid = {
                'lambda_ridge': [1.0, 5.0, 10.0, 20.0],
                'gamma': [0.2, 0.3, 0.5, 0.7],
                's': [0.5, 1.0, 2.0],
                'group_threshold': [0.7, 0.8, 0.9],
            }
        else:
            self.param_grid = param_grid

        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.use_1se = use_1se
        self.fixed_params = kwargs

        # 拟合后属性
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.is_fitted_ = False
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None, **fit_params):
        """
        拟合模型并执行交叉验证调参
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("NLassoClassifierCV only supports binary classification")

        # 保存用于后续 1-SE 计算
        self._X_for_cv = X
        self._y_for_cv = y

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # 构建完整参数网格
        lambda_ridges = self.param_grid.get('lambda_ridge', [10.0])
        gammas = self.param_grid.get('gamma', [1.0])
        s_list = self.param_grid.get('s', [1.0])
        group_thresholds = self.param_grid.get('group_threshold', [0.8])

        param_combinations = list(product(lambda_ridges, gammas, s_list, group_thresholds))
        n_combinations = len(param_combinations)
        n_folds = self.cv

        if self.verbose:
            print(f"[NLassoClassifierCV] {n_combinations} param combos × {n_folds} folds = {n_combinations * n_folds} fits")

        # 存储每组参数的 fold-level 分数
        all_scores = np.full((n_combinations, n_folds), np.nan)

        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        fold_idx = 0

        for train_idx, val_idx in kfold.split(X, y):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            for combo_idx, (lr, gamma, s, gt) in enumerate(param_combinations):
                try:
                    model = NLassoClassifier(
                        lambda_ridge=lr, gamma=gamma, s=s,
                        group_threshold=gt,
                        random_state=self.random_state,
                        verbose=False,
                        **self.fixed_params
                    )
                    model.fit(X_tr, y_tr)

                    y_pred = model.predict(X_va)
                    if self.scoring == 'accuracy':
                        score = accuracy_score(y_va, y_pred)
                    else:
                        score = accuracy_score(y_va, y_pred)
                    all_scores[combo_idx, fold_idx] = score
                except Exception:
                    all_scores[combo_idx, fold_idx] = np.nan

            fold_idx += 1

        # 计算均值和标准误
        mean_scores = np.nanmean(all_scores, axis=1)
        std_scores = np.nanstd(all_scores, axis=1)
        se_scores = std_scores / np.sqrt(n_folds)

        # 保存 CV 结果
        self.cv_results_ = {
            'mean_test_score': mean_scores,
            'std_test_score': std_scores,
            'se_test_score': se_scores,
            'param_combinations': param_combinations,
            'all_scores': all_scores,
        }

        # 确定最优参数
        best_local_idx = np.nanargmax(mean_scores)
        best_score = mean_scores[best_local_idx]
        best_combo = param_combinations[best_local_idx]
        self.best_params_ = {
            'lambda_ridge': best_combo[0],
            'gamma': best_combo[1],
            's': best_combo[2],
            'group_threshold': best_combo[3],
        }
        self.best_score_ = best_score

        # 1-SE 规则
        if self.use_1se:
            best_combo_1se, best_score_1se = self._select_1se_params(
                mean_scores, se_scores, param_combinations
            )
            if self.verbose:
                print(f"[NLassoClassifierCV] Best (lambda_ridge, gamma, s, group_threshold): ({best_combo[0]}, {best_combo[1]}, {best_combo[2]}, {best_combo[3]})")
                print(f"[NLassoClassifierCV] 1-SE  (lambda_ridge, gamma, s, group_threshold): ({best_combo_1se[0]}, {best_combo_1se[1]}, {best_combo_1se[2]}, {best_combo_1se[3]})")
            self.best_params_ = {
                'lambda_ridge': best_combo_1se[0],
                'gamma': best_combo_1se[1],
                's': best_combo_1se[2],
                'group_threshold': best_combo_1se[3],
            }
            self.best_score_ = best_score_1se

        # 用最优参数在全量数据上终极拟合
        self.best_estimator_ = NLassoClassifier(
            **self.best_params_,
            random_state=self.random_state,
            verbose=self.verbose,
            **self.fixed_params
        )
        self.best_estimator_.fit(X, y)
        self.coef_ = self.best_estimator_.coef_
        self.intercept_ = self.best_estimator_.intercept_
        self.is_fitted_ = True

        return self

    def _select_1se_params(self, mean_scores, se_scores, param_combinations):
        """
        1-SE 规则：选择最稀疏的模型，其分数在 best - 1*SE 范围内
        """
        best_idx = np.nanargmax(mean_scores)
        best_score = mean_scores[best_idx]
        threshold = best_score - se_scores[best_idx]

        candidates_mask = mean_scores >= threshold
        candidate_indices = np.where(candidates_mask)[0]

        if len(candidate_indices) == 0:
            return param_combinations[best_idx], mean_scores[best_idx]

        n_nonzero_list = []
        for idx in candidate_indices:
            lr, gamma, s, gt = param_combinations[idx]
            model = NLassoClassifier(
                lambda_ridge=lr, gamma=gamma, s=s,
                group_threshold=gt,
                random_state=self.random_state,
                verbose=False,
                **self.fixed_params
            )
            model.fit(self._X_for_cv, self._y_for_cv)
            n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
            n_nonzero_list.append(n_nonzero)

        best_candidate_idx = candidate_indices[np.argmin(n_nonzero_list)]
        return param_combinations[best_candidate_idx], mean_scores[best_candidate_idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> float:
        """评分"""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.score(X, y, sample_weight=sample_weight)

    @property
    def coef_(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.coef_

    @property
    def intercept_(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.intercept_

    @property
    def groups_(self):
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        return self.best_estimator_.groups_
