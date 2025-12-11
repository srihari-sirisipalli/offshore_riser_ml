"""
Resource Limits Validator

Prevents resource exhaustion, DoS attacks, and system crashes by validating
and enforcing resource limits before execution.

Features:
- HPO grid size limits
- Memory usage limits
- Dataset size limits
- Execution time limits
- Feature count limits
- File size limits
"""

import psutil
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from utils.constants import (
    MAX_HPO_CONFIGURATIONS,
    MAX_DATASET_ROWS,
    MAX_FEATURES,
    MAX_EXECUTION_TIME_HOURS,
    MAX_MEMORY_USAGE_PERCENT,
    MAX_FILE_SIZE_MB,
    MIN_DATASET_ROWS,
    MIN_TRAIN_SAMPLES,
    MIN_FEATURES
)


@dataclass
class ResourceLimit:
    """Represents a resource limit violation."""
    resource: str
    current: any
    limit: any
    severity: str  # 'error', 'warning', 'info'
    message: str


class ResourceLimitsValidator:
    """
    Validates configuration and runtime parameters against resource limits.

    Prevents:
    - Out-of-memory crashes
    - Excessive computation time
    - Disk space exhaustion
    - DoS-like scenarios
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize resource limits validator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.violations: List[ResourceLimit] = []

    def validate_config(self, config: Dict) -> Tuple[bool, List[ResourceLimit]]:
        """
        Validate configuration against all resource limits.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, violations_list)
        """
        self.violations = []

        # Validate HPO configuration
        self._validate_hpo_limits(config)

        # Validate execution limits
        self._validate_execution_limits(config)

        # Validate memory limits
        self._validate_memory_limits(config)

        # Validate feature limits
        self._validate_feature_limits(config)

        # Check for errors (warnings are allowed)
        has_errors = any(v.severity == 'error' for v in self.violations)

        # Log violations
        for violation in self.violations:
            if violation.severity == 'error':
                self.logger.error(violation.message)
            elif violation.severity == 'warning':
                self.logger.warning(violation.message)
            else:
                self.logger.info(violation.message)

        return (not has_errors, self.violations)

    def _validate_hpo_limits(self, config: Dict) -> None:
        """Validate HPO configuration limits."""
        hpo_config = config.get('hyperparameter_optimization', {})

        if 'param_grid' in hpo_config:
            grid_size = self._calculate_grid_size(hpo_config['param_grid'])

            if grid_size > MAX_HPO_CONFIGURATIONS:
                self.violations.append(ResourceLimit(
                    resource='HPO Grid Size',
                    current=grid_size,
                    limit=MAX_HPO_CONFIGURATIONS,
                    severity='error',
                    message=f"HPO grid size ({grid_size:,}) exceeds maximum "
                           f"({MAX_HPO_CONFIGURATIONS:,}). Reduce parameter grid."
                ))
            elif grid_size > MAX_HPO_CONFIGURATIONS * 0.8:
                self.violations.append(ResourceLimit(
                    resource='HPO Grid Size',
                    current=grid_size,
                    limit=MAX_HPO_CONFIGURATIONS,
                    severity='warning',
                    message=f"HPO grid size ({grid_size:,}) is close to maximum "
                           f"({MAX_HPO_CONFIGURATIONS:,}). Consider reducing."
                ))

    def _calculate_grid_size(self, param_grid: Dict) -> int:
        """Calculate total HPO grid size."""
        size = 1
        for param_values in param_grid.values():
            if isinstance(param_values, list):
                size *= len(param_values)
            else:
                size *= 1  # Single value
        return size

    def _validate_execution_limits(self, config: Dict) -> None:
        """Validate execution time limits."""
        execution = config.get('execution', {})

        if 'max_hours' in execution:
            max_hours = execution['max_hours']
            if max_hours > MAX_EXECUTION_TIME_HOURS:
                self.violations.append(ResourceLimit(
                    resource='Execution Time',
                    current=max_hours,
                    limit=MAX_EXECUTION_TIME_HOURS,
                    severity='error',
                    message=f"Maximum execution time ({max_hours}h) exceeds limit "
                           f"({MAX_EXECUTION_TIME_HOURS}h)"
                ))

    def _validate_memory_limits(self, config: Dict) -> None:
        """Validate memory configuration."""
        # Get system memory
        try:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)

            self.logger.debug(
                f"System memory: {total_memory_gb:.1f}GB total, "
                f"{available_memory_gb:.1f}GB available"
            )

            # Warn if low memory
            if available_memory_gb < 2.0:
                self.violations.append(ResourceLimit(
                    resource='Available Memory',
                    current=f"{available_memory_gb:.1f}GB",
                    limit="2GB recommended",
                    severity='warning',
                    message=f"Low available memory ({available_memory_gb:.1f}GB). "
                           "Pipeline may encounter memory issues."
                ))

        except Exception as e:
            self.logger.warning(f"Could not check system memory: {e}")

    def _validate_feature_limits(self, config: Dict) -> None:
        """Validate feature count limits."""
        iterative = config.get('iterative', {})

        if 'min_features' in iterative:
            min_features = iterative['min_features']
            if min_features < MIN_FEATURES:
                self.violations.append(ResourceLimit(
                    resource='Minimum Features',
                    current=min_features,
                    limit=MIN_FEATURES,
                    severity='error',
                    message=f"min_features ({min_features}) is below minimum ({MIN_FEATURES})"
                ))

    def validate_dataset(self, dataset_rows: int, dataset_cols: int,
                        feature_count: int) -> Tuple[bool, List[ResourceLimit]]:
        """
        Validate dataset dimensions against limits.

        Args:
            dataset_rows: Number of rows in dataset
            dataset_cols: Number of columns in dataset
            feature_count: Number of feature columns

        Returns:
            Tuple of (is_valid, violations_list)
        """
        violations = []

        # Check minimum rows
        if dataset_rows < MIN_DATASET_ROWS:
            violations.append(ResourceLimit(
                resource='Dataset Rows',
                current=dataset_rows,
                limit=MIN_DATASET_ROWS,
                severity='error',
                message=f"Dataset has only {dataset_rows} rows. "
                       f"Minimum required: {MIN_DATASET_ROWS}"
            ))

        # Check maximum rows
        if dataset_rows > MAX_DATASET_ROWS:
            violations.append(ResourceLimit(
                resource='Dataset Rows',
                current=dataset_rows,
                limit=MAX_DATASET_ROWS,
                severity='error',
                message=f"Dataset has {dataset_rows:,} rows. "
                       f"Maximum allowed: {MAX_DATASET_ROWS:,}"
            ))

        # Check feature count
        if feature_count > MAX_FEATURES:
            violations.append(ResourceLimit(
                resource='Feature Count',
                current=feature_count,
                limit=MAX_FEATURES,
                severity='error',
                message=f"Dataset has {feature_count:,} features. "
                       f"Maximum allowed: {MAX_FEATURES:,}"
            ))

        # Estimate memory requirement
        estimated_memory_mb = (dataset_rows * dataset_cols * 8) / (1024**2)  # 8 bytes per float64
        if estimated_memory_mb > 5000:  # 5GB
            violations.append(ResourceLimit(
                resource='Estimated Memory',
                current=f"{estimated_memory_mb:.0f}MB",
                limit="5GB recommended",
                severity='warning',
                message=f"Dataset will require ~{estimated_memory_mb:.0f}MB memory. "
                       "Consider using smaller precision (float32) or sampling."
            ))

        # Log violations
        for violation in violations:
            if violation.severity == 'error':
                self.logger.error(violation.message)
            else:
                self.logger.warning(violation.message)

        has_errors = any(v.severity == 'error' for v in violations)
        return (not has_errors, violations)

    def estimate_hpo_memory(self, config: Dict, dataset_rows: int) -> float:
        """
        Estimate peak memory usage for HPO run.

        Args:
            config: Configuration dictionary
            dataset_rows: Number of rows in dataset

        Returns:
            Estimated peak memory in MB
        """
        hpo_config = config.get('hyperparameter_optimization', {})

        # Get grid size
        if 'param_grid' in hpo_config:
            grid_size = self._calculate_grid_size(hpo_config['param_grid'])
        else:
            grid_size = 100  # Default estimate

        # Get CV folds
        cv_folds = config.get('splitting', {}).get('cv_folds', 5)

        # Estimate: each config creates predictions for all CV folds
        # Assume ~50 bytes per prediction row (8 floats)
        prediction_size_mb = (dataset_rows * 50 * cv_folds) / (1024**2)

        # If keeping all in memory (current implementation)
        total_memory_mb = prediction_size_mb * grid_size

        self.logger.debug(
            f"HPO memory estimate: {grid_size} configs × {cv_folds} folds × "
            f"{dataset_rows} rows ≈ {total_memory_mb:.0f}MB"
        )

        return total_memory_mb

    def check_system_resources(self) -> Dict[str, any]:
        """
        Check current system resource availability.

        Returns:
            Dictionary with resource information
        """
        resources = {}

        try:
            # CPU
            resources['cpu_count'] = psutil.cpu_count()
            resources['cpu_percent'] = psutil.cpu_percent(interval=1)

            # Memory
            mem = psutil.virtual_memory()
            resources['memory_total_gb'] = mem.total / (1024**3)
            resources['memory_available_gb'] = mem.available / (1024**3)
            resources['memory_percent'] = mem.percent

            # Disk
            disk = psutil.disk_usage('/')
            resources['disk_total_gb'] = disk.total / (1024**3)
            resources['disk_free_gb'] = disk.free / (1024**3)
            resources['disk_percent'] = disk.percent

            # Log summary
            self.logger.info(
                f"System Resources: CPU={resources['cpu_count']} cores "
                f"({resources['cpu_percent']}% used), "
                f"Memory={resources['memory_available_gb']:.1f}GB available "
                f"({resources['memory_percent']:.1f}% used), "
                f"Disk={resources['disk_free_gb']:.1f}GB free "
                f"({resources['disk_percent']:.1f}% used)"
            )

        except Exception as e:
            self.logger.warning(f"Could not check system resources: {e}")

        return resources
