"""
UniLasso Simulation Experiments Package
========================================

This package contains the experiment framework for systematically evaluating
the UniLasso algorithm with:
- Group constraint (promoting grouped selection of correlated features)
- Adaptive weighting (penalty adjustment based on univariate significance)

The experiments cover three settings:
1. Linear Gaussian sparse regression
2. GLM extensions (all GLM families)
3. Nonlinear modeling with splines and trees
"""

from .base_experiment import BaseSimulationExperiment

__version__ = "0.1.0"
__all__ = ["BaseSimulationExperiment"]
