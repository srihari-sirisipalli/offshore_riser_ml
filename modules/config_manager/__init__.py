"""
Configuration Manager Module
============================

Responsibility:
- Centralized loading and validation of JSON configuration files.
- Enforcement of schema constraints and logical business rules.
- Resource usage guardrails (DoD/Memory protection).
- Deterministic seed propagation for reproducibility.
"""

from .config_manager import ConfigurationManager

__all__ = ['ConfigurationManager']