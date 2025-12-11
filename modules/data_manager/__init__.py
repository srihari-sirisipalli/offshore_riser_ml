"""
Data Manager Module
===================

Responsibility:
- Secure loading of raw data files (CSV, Excel).
- Validation of schema, types, and geometric constraints.
- Preprocessing and feature engineering (stratification bins).
- Persistence of validated data for downstream consumption.
"""

from .data_manager import DataManager
from .data_persistence import DataPersistence

__all__ = ['DataManager', 'DataPersistence']