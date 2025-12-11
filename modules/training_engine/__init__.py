"""
Training Engine Module
======================

Responsibility:
- Prepares feature matrices (X) and target vectors (y) from DataFrames.
- Instantiates models via ModelFactory.
- Executes model training (fitting).
- Persists trained models (.pkl) and training metadata (.json).
- Manages memory cleanup post-training.
"""

from .training_engine import TrainingEngine

__all__ = ['TrainingEngine']