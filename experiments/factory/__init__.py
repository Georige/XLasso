"""
Factory Module - Experiment Execution Layer
==========================================
Provides unified interfaces for running experiments:
- run.py: Single experiment runner
- sweep.py: Two-stage hyperparameter tuning
- compare.py: Multi-algorithm comparison hub
"""

from .run import run_experiment
from .sweep import run_stage1, run_stage2

__all__ = [
    "run_experiment",
    "run_stage1",
    "run_stage2",
]
