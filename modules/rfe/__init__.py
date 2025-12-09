"""
Recursive Feature Elimination (RFE) Module.

This package contains the logic for the iterative circular RFE pipeline, including:
- RFEController: Orchestrates the multi-round execution.
- FeatureEvaluator: Handles LOFO (Leave-One-Feature-Out) evaluation.
- ComparisonEngine: Compares baseline vs. dropped models.
- StoppingCriteria: Evaluates when to halt the RFE loop.
"""

from .rfe_controller import RFEController
from .feature_evaluator import FeatureEvaluator
from .comparison_engine import ComparisonEngine
from .stopping_criteria import StoppingCriteria

__all__ = [
    'RFEController',
    'FeatureEvaluator',
    'ComparisonEngine',
    'StoppingCriteria'
]