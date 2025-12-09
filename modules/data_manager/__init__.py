"""
Data Module.

Handles loading, validation, splitting, and memory-optimized persistence.
"""

from .data_manager import DataManager
from .data_persistence import DataPersistence

__all__ = [
    'DataManager',
    'DataPersistence'
]