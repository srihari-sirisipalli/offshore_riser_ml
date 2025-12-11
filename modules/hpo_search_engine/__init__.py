"""
HPO Search Engine
=================

Responsibility:
- Hyperparameter Optimization using Grid Search.
- Cross-Validation with stratified sampling.
- Snapshot generation for Global Error Tracking.
- Memory-efficient processing of large configuration grids.
"""

from .hpo_search_engine import HPOSearchEngine

__all__ = ['HPOSearchEngine']