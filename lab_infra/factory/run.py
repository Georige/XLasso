"""
Factory run.py - Single experiment entry point for lab_infra
Supports: XLasso, UniLasso, Lasso, Adaptive Lasso, Group Lasso, Fused Lasso

Usage: python run.py --scope {pilot,stage1,stage2} --config CONFIG.yaml --algorithm ALGO
"""
import argparse
import os
import sys
import json
import datetime
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExperimentResult:
    """Single experiment result"""
    repeat_id: int
    config: Dict[str, Any]
    metrics: Dict[str, float]
    coef_: Optional[np.ndarray] = None
    runtime: float = 0.0
    status: str = "pending"
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        if self.coef_ is not None:
            d['coef_'] = self.coef_.tolist()
        return d


@dataclass
class Status:
    """Experiment status for checkpoint/resume"""
    experiment_name: str
    stage: str
    scope: str
    current_repeat: int = 0
    total_repeats: int = 3
    current_config_idx: int = 0
    total_configs: int = 0
    completed: bool = False
    results: List[Dict] = field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error: Optional[str] = None

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Status':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='XLasso experiment runner')
    parser.add_argument('--scope', type=str, required=True,
                        choices=['pilot', 'stage1', 'stage2'],
                        help='Experiment scope')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--experiment', type=str, default='exp1',
                        help='Experiment name (e.g., exp1, exp2)')
    parser.add_argument('--algorithm', type=str, default='xlasso',
                        choices=['xlasso', 'unilasso', 'lasso', 'adaptive_lasso',
                                'group_lasso', 'fused_lasso', 'asgl'],
                        help='Algorithm to use')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if exists')
    parser.add_argument('--repeat-id', type=int, default=None,
                        help='Specific repeat ID to run (0-based)')
    parser.add_argument('--sigma', type=float, default=None,
                        help='Override sigma value')
    # Override grid parameters with single values
    parser.add_argument('--k', type=float, default=None,
                        help='Override k (gamma) parameter')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=None,
                        help='Override lambda parameter')
    parser.add_argument('--threshold', type=str, default=None,
                        help='Override threshold parameter')
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# Data Generation
# =============================================================================

def generate_experiment_data(exp_config: Dict[str, Any], random_state: int,
                            sigma_override: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic experiment data based on experiment type.

    Returns: X, y, beta_true, true_indices
    """
    n = exp_config.get('n', 200)
    p = exp_config.get('p', 100)
    true_sparsity = exp_config.get('true_sparsity', 20)
    sigma = sigma_override if sigma_override is not None else exp_config.get('sigma', 1.0)
    correlation = exp_config.get('correlation', 'pairwise')
    family = exp_config.get('family', 'gaussian')

    rng = np.random.RandomState(random_state)

    # Generate coefficient vector
    beta_true = np.zeros(p)
    if exp_config.get('anti_sign', False):
        # For twin variables with opposite signs
        for i in range(0, min(true_sparsity * 2, p), 2):
            beta_true[i] = 2.0 if i % 4 == 0 else -2.5
    else:
        beta_true[:true_sparsity] = rng.randn(true_sparsity)

    # Generate design matrix with correlation structure
    if correlation == 'ar1':
        rho = exp_config.get('rho', 0.5)
        Sigma = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = rho ** abs(i - j)
        L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(p))
        X = rng.randn(n, p) @ L.T
    elif correlation == 'pairwise':
        rho = exp_config.get('rho', 0.5)
        Sigma = np.full((p, p), rho)
        np.fill_diagonal(Sigma, 1.0)
        L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(p))
        X = rng.randn(n, p) @ L.T
    else:
        X = rng.randn(n, p)

    # Generate response
    noise = rng.randn(n) * sigma
    y = X @ beta_true + noise

    if family == 'binomial':
        # Convert to binary classification
        y_binary = (1 / (1 + np.exp(-(X @ beta_true + noise) / sigma)) > 0.5).astype(int)
        return X, y_binary, beta_true, np.arange(true_sparsity)

    return X, y, beta_true, np.arange(true_sparsity)


# =============================================================================
# Algorithm Wrappers
# =============================================================================

class BaseAlgorithmWrapper:
    """Base class for algorithm wrappers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.is_fitted_: bool = False
        self.family = config.get('family', 'gaussian')

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseAlgorithmWrapper':
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted")
        if self.family == 'gaussian':
            return X @ self.coef_ + self.intercept_
        else:
            prob = 1 / (1 + np.exp(-(X @ self.coef_ + self.intercept_)))
            return (prob >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted")
        return 1 / (1 + np.exp(-(X @ self.coef_ + self.intercept_)))

    def get_metrics(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        y_pred = self.predict(X)
        metrics = {}

        if self.family == 'gaussian':
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
        else:
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['f1'] = f1_score(y, y_pred, average='binary', zero_division=0)
            try:
                metrics['auc'] = roc_auc_score(y, self.predict_proba(X))
            except ValueError:
                metrics['auc'] = 0.5

        # Selection metrics
        if 'true_indices' in kwargs:
            true_idx = kwargs['true_indices']
            selected = np.where(np.abs(self.coef_) > 1e-8)[0]
            tp = len(set(selected) & set(true_idx))
            fp = len(set(selected) - set(true_idx))
            fn = len(set(true_idx) - set(selected))
            metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['fdr'] = fp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics['n_selected'] = len(selected)

        return metrics


class XLassoWrapper(BaseAlgorithmWrapper):
    """XLasso wrapper using lab_infra.modules.unilasso"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.k = config.get('k', 1.0)
        self.threshold = config.get('threshold', 'OFF')
        self.filter_ = config.get('filter', False)
        self.enable_group_decomp = config.get('enable_group_decomp', False)
        self.adaptive_weighting = config.get('adaptive_weighting', True)
        self.cv_folds = config.get('cv_folds', 3)
        self.lambda_ = config.get('lambda', config.get('lambda_', 1.0))
        self.lmda_min_ratio = config.get('lmda_min_ratio', 1e-2)
        self.n_lmdas = config.get('n_lmdas', 100)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'XLassoWrapper':
        from lab_infra.modules.unilasso.soft_unilasso import cv_uni
        from lab_infra.modules.unilasso.uni_lasso import fit_unilasso

        params = {
            'family': self.family,
            'k': self.k,
            'lambda': self.lambda_,
            'threshold': self.threshold,
            'filter': self.filter_,
            'enable_group_decomp': self.enable_group_decomp,
            'adaptive_weighting': self.adaptive_weighting,
        }

        if self.cv_folds > 1:
            result = cv_uni(
                X, y,
                family=self.family,
                n_folds=self.cv_folds,
                lmdas=np.logspace(-1, -4, 20),
                negative_penalty=self.k,
                seed=2026
            )
            self.coef_ = result["coefs"]
            self.intercept_ = result.get("intercept", 0.0)
        else:
            result = fit_unilasso(X, y, lmdas=[self.lambda_], **params)
            self.coef_ = result.coefs
            self.intercept_ = result.intercept if hasattr(result, 'intercept') else 0.0

        self.is_fitted_ = True
        return self


class UniLassoWrapper(BaseAlgorithmWrapper):
    """UniLasso wrapper using lab_infra.modules.unilasso"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lambda_ = config.get('lambda', config.get('lambda_', 1.0))
        self.cv_folds = config.get('cv_folds', 3)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'UniLassoWrapper':
        from lab_infra.modules.unilasso.soft_unilasso import cv_uni
        from lab_infra.modules.unilasso.uni_lasso import fit_unilasso

        if self.cv_folds > 1:
            result = cv_uni(
                X, y,
                family=self.family,
                n_folds=self.cv_folds,
                lmdas=np.logspace(-1, -4, 20),
                negative_penalty=1.0,
                seed=2026
            )
            self.coef_ = result["coefs"]
            self.intercept_ = result.get("intercept", 0.0)
        else:
            result = fit_unilasso(
                X, y,
                lmdas=[self.lambda_],
                family=self.family,
                k=1.0,
                adaptive_weighting=False,
                enable_group_decomp=False
            )
            self.coef_ = result.coefs
            self.intercept_ = result.intercept if hasattr(result, 'intercept') else 0.0

        self.is_fitted_ = True
        return self


class LassoWrapper(BaseAlgorithmWrapper):
    """Standard Lasso wrapper using sklearn"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lambda_ = config.get('lambda', config.get('lambda_', 1.0))
        self.cv_folds = config.get('cv_folds', 3)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LassoWrapper':
        from sklearn.linear_model import LassoCV, LogisticRegressionCV

        if self.family == 'gaussian':
            if self.cv_folds > 1:
                model = LassoCV(cv=self.cv_folds, max_iter=5000, n_jobs=-1)
            else:
                from sklearn.linear_model import Lasso
                model = Lasso(alpha=self.lambda_, max_iter=5000)
            model.fit(X, y)
            self.coef_ = model.coef_
            self.intercept_ = model.intercept_
        else:
            if self.cv_folds > 1:
                model = LogisticRegressionCV(
                    cv=self.cv_folds, max_iter=5000,
                    penalty='l1', solver='liblinear', n_jobs=-1
                )
            else:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(
                    penalty='l1', solver='liblinear', max_iter=5000, C=1.0/self.lambda_
                )
            model.fit(X, y)
            self.coef_ = model.coef_.ravel()
            self.intercept_ = model.intercept_[0]

        self.is_fitted_ = True
        return self


class AdaptiveLassoWrapper(BaseAlgorithmWrapper):
    """Adaptive Lasso wrapper using lab_infra.modules.other_lasso"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.gamma = config.get('gamma', 1.0)
        self.cv_folds = config.get('cv_folds', 3)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'AdaptiveLassoWrapper':
        from lab_infra.modules.other_lasso.adaptive_lasso import AdaptiveLassoCV

        model = AdaptiveLassoCV(
            gammas=[0.5, 1.0, 2.0],
            cv=self.cv_folds,
            n_jobs=-1,
            max_iter=1000
        )
        model.fit(X, y)
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_ if hasattr(model, 'intercept_') else 0.0
        self.is_fitted_ = True
        return self


class GroupLassoWrapper(BaseAlgorithmWrapper):
    """Group Lasso wrapper using lab_infra.modules.other_lasso"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cv_folds = config.get('cv_folds', 3)
        self.groups = config.get('groups', None)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'GroupLassoWrapper':
        from lab_infra.modules.other_lasso.group_lasso import GroupLassoCV, GroupLasso
        from sklearn.utils.validation import check_X_y

        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        # Auto-generate groups if not provided
        if self.groups is None:
            self.groups = GroupLasso.group_features_by_correlation(X, corr_threshold=0.7)

        model = GroupLassoCV(
            cv=self.cv_folds,
            n_jobs=-1,
            max_iter=1000
        )
        # Set groups before fitting
        model.groups = self.groups
        model.fit(X, y)
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_ if hasattr(model, 'intercept_') else 0.0
        self.is_fitted_ = True
        return self


class FusedLassoWrapper(BaseAlgorithmWrapper):
    """Fused Lasso wrapper using lab_infra.modules.other_lasso"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cv_folds = config.get('cv_folds', 3)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'FusedLassoWrapper':
        from lab_infra.modules.other_lasso.fused_lasso import FusedLassoCV

        model = FusedLassoCV(
            lambda_fused_ratios=[0.1, 0.5, 1.0, 2.0],
            cv=self.cv_folds,
            n_jobs=-1,
            max_iter=1000
        )
        model.fit(X, y)
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_ if hasattr(model, 'intercept_') else 0.0
        self.is_fitted_ = True
        return self


class ASGLWrapper(BaseAlgorithmWrapper):
    """Adaptive Sparse Group Lasso wrapper"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cv_folds = config.get('cv_folds', 3)
        self.groups = config.get('groups', None)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'ASGLWrapper':
        from lab_infra.modules.other_lasso.adaptive_sparse_group_lasso import AdaptiveSparseGroupLassoCV
        from lab_infra.modules.other_lasso.group_lasso import GroupLasso

        if self.groups is None:
            self.groups = GroupLasso.group_features_by_correlation(X, corr_threshold=0.7)

        model = AdaptiveSparseGroupLassoCV(
            l1_ratios=[0.1, 0.5, 0.9],
            cv=self.cv_folds,
            n_jobs=-1,
            max_iter=500
        )
        model.groups = self.groups
        model.fit(X, y)
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_ if hasattr(model, 'intercept_') else 0.0
        self.is_fitted_ = True
        return self


# Algorithm registry
ALGORITHM_REGISTRY = {
    'xlasso': XLassoWrapper,
    'unilasso': UniLassoWrapper,
    'lasso': LassoWrapper,
    'adaptive_lasso': AdaptiveLassoWrapper,
    'group_lasso': GroupLassoWrapper,
    'fused_lasso': FusedLassoWrapper,
    'asgl': ASGLWrapper,
}


# =============================================================================
# Experiment Running
# =============================================================================

def run_single_experiment(
    config: Dict[str, Any],
    algo_name: str,
    repeat_id: int,
    exp_config: Dict[str, Any],
    sigma_override: float = None
) -> ExperimentResult:
    """Run a single experiment with given configuration"""
    start_time = time.time()

    # Generate data
    X, y, beta_true, true_indices = generate_experiment_data(
        exp_config, random_state=2026 + repeat_id, sigma_override=sigma_override
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=2026 + repeat_id
    )

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize algorithm
    algo_class = ALGORITHM_REGISTRY.get(algo_name)
    if algo_class is None:
        return ExperimentResult(
            repeat_id=repeat_id,
            config={'algorithm': algo_name},
            metrics={},
            runtime=time.time() - start_time,
            status='failed',
            error=f"Unknown algorithm: {algo_name}"
        )

    algo_config = config.copy()
    algo_config['family'] = exp_config.get('family', 'gaussian')
    algo = algo_class(algo_config)

    # Fit model
    try:
        algo.fit(X_train_scaled, y_train)
        y_pred = algo.predict(X_test_scaled)

        # Compute metrics
        metrics = algo.get_metrics(X_test_scaled, y_test, true_indices=true_indices)

        result = ExperimentResult(
            repeat_id=repeat_id,
            config=algo_config,
            metrics=metrics,
            coef_=algo.coef_,
            runtime=time.time() - start_time,
            status='completed'
        )
    except Exception as e:
        result = ExperimentResult(
            repeat_id=repeat_id,
            config=algo_config,
            metrics={},
            runtime=time.time() - start_time,
            status='failed',
            error=str(e)
        )

    return result


def save_results(results: List[ExperimentResult], output_dir: str,
                  experiment_name: str, config: Dict[str, Any], algo_name: str):
    """Save results to output directory"""
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(output_dir, 'config.yaml')
    import yaml
    full_config = {**config, 'algorithm': algo_name}
    with open(config_path, 'w') as f:
        yaml.dump(full_config, f)

    # Save status
    status = Status(
        experiment_name=experiment_name,
        stage=config.get('stage', 'unknown'),
        scope=config.get('scope', 'unknown'),
        total_repeats=len(results),
        completed=True,
        results=[r.to_dict() for r in results]
    )
    status.end_time = datetime.datetime.now().isoformat()
    status.save(os.path.join(output_dir, 'status.json'))

    # Save raw results
    raw_data = []
    for r in results:
        row = {
            'repeat_id': r.repeat_id,
            'status': r.status,
            'runtime': r.runtime,
        }
        row.update(r.metrics)
        raw_data.append(row)
    raw_df = pd.DataFrame(raw_data)
    raw_df.to_csv(os.path.join(output_dir, 'raw.csv'), index=False)

    # Save summary
    if raw_data:
        summary = {}
        for key in raw_data[0].keys():
            if key not in ['repeat_id', 'status']:
                values = [r[key] for r in raw_data if r.get(key) is not None]
                if values:
                    summary[f'{key}_mean'] = np.mean(values)
                    summary[f'{key}_std'] = np.std(values)
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.join(
            PROJECT_ROOT, 'lab_infra', 'results', args.scope,
            f"{args.algorithm}_{args.experiment}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    else:
        output_dir = args.output_dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check for resume
    status_path = os.path.join(output_dir, 'status.json')
    if args.resume and os.path.exists(status_path):
        status = Status.load(status_path)
        print(f"Resuming from repeat {status.current_repeat}/{status.total_repeats}")
        start_repeat = status.current_repeat
    else:
        start_repeat = 0

    # Get experiment config
    experiments = config.get('experiments', {})
    exp_config = experiments.get(args.experiment, experiments.get('default', {}))

    # Apply command-line parameter overrides
    # If a grid value is provided in config, use the first value or the override
    for param in ['k', 'lambda', 'threshold', 'filter']:
        if hasattr(args, param) and getattr(args, param) is not None:
            config[param] = getattr(args, param)
        elif param in config and isinstance(config[param], list):
            config[param] = config[param][0]

    # Also handle lambda_ alias
    if args.lambda_ is not None:
        config['lambda'] = args.lambda_

    # Get sigma values to iterate
    sigmas = [args.sigma] if args.sigma else exp_config.get('sigma', [0.5, 1.0, 2.5])
    if not isinstance(sigmas, list):
        sigmas = [sigmas]

    # Run experiments
    n_repeats = config.get('n_repeats', 3)
    results = []

    total_runs = n_repeats * len(sigmas)
    run_idx = 0

    for repeat_id in range(start_repeat, n_repeats):
        for sigma in sigmas:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] Running {args.algorithm} repeat={repeat_id} sigma={sigma}...", end=" ", flush=True)

            result = run_single_experiment(config, args.algorithm, repeat_id, exp_config, sigma_override=sigma)
            results.append(result)

            if result.status == 'completed':
                metrics_str = " ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in list(result.metrics.items())[:3]])
                print(f"OK | {metrics_str}")
            else:
                print(f"FAILED | {result.error}")

            # Save checkpoint
            temp_status = Status(
                experiment_name=args.experiment,
                stage=config.get('stage', 'unknown'),
                scope=args.scope,
                current_repeat=repeat_id + 1,
                total_repeats=n_repeats,
                completed=(run_idx == total_runs)
            )
            temp_status.results = [r.to_dict() for r in results]
            temp_status.save(status_path)

    # Save final results
    save_results(results, output_dir, args.experiment, config, args.algorithm)
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
