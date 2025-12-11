import json
import os
import hashlib
import sys
import logging
import numpy as np
import jsonschema
import psutil  # Required for memory awareness
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.model_selection import ParameterGrid

from utils.exceptions import ConfigurationError
from utils import constants

class ConfigurationManager:
    """
    Manages system configuration loading, validation, and access.
    Acts as the single source of truth and safety guard for the pipeline.
    
    Roadmap Improvements:
    - Task 1.2: Resource Limits (Memory, HPO Grid Size).
    - Task 1.4: Enhanced Logical Validation (Bounds checking).
    - Issue #8: Comprehensive Validation.
    """
    
    # Default Resource Limits (Safety Guardrails)
    DEFAULT_MAX_HPO_CONFIGS = 1000  # Prevent accidental combinatoric explosions
    DEFAULT_MAX_EXECUTION_TIME_SEC = 86400  # 24 hours
    
    def __init__(self, config_path: str = "config/config.json",
                 schema_path: str = "config/schema.json"):
        """
        Initialize the ConfigurationManager.

        Args:
            config_path (str): Path to the user configuration JSON.
            schema_path (str): Path to the JSON schema definition.
        """
        self.config_path = config_path
        self.schema_path = schema_path
        self.config: Dict[str, Any] = {}
        self.schema: Dict[str, Any] = {}
        self.run_id: Optional[str] = None
        self.logger = logging.getLogger("config_manager")
        
    def load_and_validate(self) -> Dict[str, Any]:
        """
        Main entry point. Loads config, validates schema/logic/resources, 
        applies defaults, and propagates seeds.
        
        Returns:
            Dict[str, Any]: The fully validated and hydrated configuration.
            
        Raises:
            ConfigurationError: If any validation step fails.
        """
        # 1. Load Files
        self.config = self._load_json(self.config_path)
        self.schema = self._load_json(self.schema_path)
        
        # 2. Structural Validation (Schema)
        self._validate_schema()
        
        # 3. Logical Validation (Business Rules & Bounds)
        self._validate_logic()
        
        # 4. Resource Validation (Prevent Exhaustion/DoS)
        self._validate_resources()
        
        # 5. Internal Seed Propagation (Reproducibility)
        self._propagate_seeds()
        
        return self.config

    def generate_run_id(self) -> str:
        """
        Generate or retrieve a unique run identifier based on timestamp.
        Used for directory naming and metadata.
        """
        if not self.run_id:
            timestamp = datetime.now()
            # Format: YYYYMMDD_HHMMSS
            self.run_id = timestamp.strftime("%Y%m%d_%H%M%S")
        return self.run_id

    def save_artifacts(self, output_dir: str) -> None:
        """
        Save configuration artifacts to the run directory for full reproducibility.
        
        Saves:
        1. config_used.json: The exact config object in memory.
        2. config_hash.txt: SHA256 hash for versioning.
        3. run_metadata.json: Environment details (Python version, Platform, etc.).
        """
        config_dir = Path(output_dir) / constants.CONFIG_DIR
        config_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save Config
        with open(config_dir / "config_used.json", 'w') as f:
            json.dump(self.config, f, indent=2)
            
        # 2. Calculate and Save Hash
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()

        with open(config_dir / "config_hash.txt", 'w') as f:
            f.write(config_hash)

        # 3. Save Metadata (Environment Capture)
        metadata = {
            'run_id': self.run_id,
            'start_time': datetime.now().isoformat(),
            'python_version': sys.version,
            'platform': sys.platform,
            'config_hash': config_hash,
            'working_directory': os.getcwd()
        }

        with open(config_dir / "run_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_json(self, path: str) -> Dict[str, Any]:
        """Safely load a JSON file."""
        if not os.path.exists(path):
            raise ConfigurationError(f"File not found: {path}")
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {path}: {str(e)}")

    def _validate_schema(self) -> None:
        """Validate config structure against JSON schema."""
        try:
            jsonschema.validate(instance=self.config, schema=self.schema)
        except jsonschema.ValidationError as e:
            raise ConfigurationError(f"Schema validation failed: {e.message}")

    def _validate_logic(self) -> None:
        """Comprehensive logical validation."""
        # --- Data Section ---
        data = self.config.get('data', {})
        if 'hs_column' not in data:
            raise ConfigurationError("Data 'hs_column' must be specified.")
        for key in ['file_path', 'target_sin', 'target_cos']:
            if not data.get(key):
                raise ConfigurationError(f"Data '{key}' must be specified and non-empty.")
        
        if data.get('circle_tolerance', 0.01) < 0:
             raise ConfigurationError("circle_tolerance must be non-negative.")
        # Data quality gates validation
        gates = self.config.get('data_quality_gates', {})
        if gates:
            if gates.get('min_split_rows', 1) <= 0:
                raise ConfigurationError("data_quality_gates.min_split_rows must be > 0")
            max_missing = gates.get('max_missing_pct', 100.0)
            if max_missing < 0 or max_missing > 100:
                raise ConfigurationError("data_quality_gates.max_missing_pct must be between 0 and 100")

        # --- Splitting Section ---
        split = self.config.get('splitting', {})
        test_size = split.get('test_size', 0.2)
        val_size = split.get('val_size', 0.2)
        angle_bins = split.get('angle_bins', 0)
        hs_bins = split.get('hs_bins', 0)
        
        if not (0.0 < test_size < 1.0):
            raise ConfigurationError(f"test_size must be between 0 and 1 (exclusive), got {test_size}")
        if not (0.0 < val_size < 1.0):
            raise ConfigurationError(f"val_size must be between 0 and 1 (exclusive), got {val_size}")
        if test_size + val_size >= 1.0:
            raise ConfigurationError(f"Sum of test_size ({test_size}) and val_size ({val_size}) must be < 1.0 to leave room for training data.")
        
        if split.get('seed', 42) < 0:
            raise ConfigurationError("Splitting seed must be non-negative.")
        if angle_bins and angle_bins <= 0:
            raise ConfigurationError(f"angle_bins must be > 0, got {angle_bins}")
        if hs_bins and hs_bins <= 0:
            raise ConfigurationError(f"hs_bins must be > 0, got {hs_bins}")
        
        # --- HPO Section ---
        hpo = self.config.get('hyperparameters', {})
        if hpo.get('enabled', False):
            grids = hpo.get('grids')
            if not grids:
                raise ConfigurationError("Hyperparameter grids cannot be empty when HPO is enabled.")
            
            cv_folds = hpo.get('cv_folds', 5)
            if cv_folds < 2:
                raise ConfigurationError(f"cv_folds must be >= 2, got {cv_folds}.")
            max_trials = hpo.get('max_trials', None)
            if max_trials is not None and max_trials <= 0:
                raise ConfigurationError(f"max_trials must be > 0 when provided, got {max_trials}.")

        # --- Iterative/RFE Section ---
        rfe = self.config.get('iterative', {})
        if rfe.get('enabled', False):
            min_feat = rfe.get('min_features', 1)
            if min_feat < 1:
                raise ConfigurationError(f"min_features must be >= 1, got {min_feat}.")
            
            max_rounds = rfe.get('max_rounds', 20)
            if max_rounds < 1:
                raise ConfigurationError(f"max_rounds must be >= 1, got {max_rounds}.")
            if max_rounds < min_feat:
                raise ConfigurationError(f"max_rounds ({max_rounds}) must be >= min_features ({min_feat}).")

        # Bootstrap validation
        if self.config.get('bootstrapping', {}).get('enabled', False):
            confidence = self.config['bootstrapping'].get('confidence_level', 0.95)
            if not (0 < confidence < 1):
                raise ConfigurationError(f"confidence_level must be in (0, 1)")
            
            sample_ratio = self.config['bootstrapping'].get('sample_ratio', 1.0)
            if sample_ratio <= 0 or sample_ratio > 1:
                raise ConfigurationError(f"sample_ratio must be in (0, 1], got {sample_ratio}")

        # HPO Analysis validation
        if self.config.get('hpo_analysis', {}):
            top_percent = self.config['hpo_analysis'].get('optimal_top_percent', 10)
            if not (0 < top_percent <= 100):
                raise ConfigurationError(f"optimal_top_percent must be in (0, 100]")

        # Execution validation
        execution = self.config.get('execution', {})
        if 'max_hours' in execution and execution['max_hours'] is not None and execution['max_hours'] <= 0:
            raise ConfigurationError(f"execution.max_hours must be > 0, got {execution['max_hours']}")
        if 'n_jobs' in execution:
            n_jobs = execution['n_jobs']
            if n_jobs == 0 or n_jobs < -1:
                raise ConfigurationError(f"execution.n_jobs must be -1 (all cores) or a positive integer, got {n_jobs}")

    def _validate_resources(self) -> None:
        """
        Validate against system resources (Phase 1 Fixes).
        Calculates total grid size and ensures it fits within safe limits to prevent crashes.
        """
        resources = self.config.get('resources', {})
        
        # 1. HPO Grid Explosion Check
        if self.config.get('hyperparameters', {}).get('enabled', False):
            total_configs = 0
            grids = self.config.get('hyperparameters', {}).get('grids', {})
            
            for model_name, grid_params in grids.items():
                try:
                    # Calculate combinations for this model
                    combinations = list(ParameterGrid(grid_params))
                    total_configs += len(combinations)
                except Exception as e:
                    raise ConfigurationError(f"Invalid parameter grid for {model_name}: {str(e)}")

            max_configs = resources.get('max_hpo_configs', self.DEFAULT_MAX_HPO_CONFIGS)
            
            if total_configs > max_configs:
                raise ConfigurationError(
                    f"HPO Grid Explosion Detected! Total configurations ({total_configs}) exceeds "
                    f"safety limit ({max_configs}). Reduce grid search space or increase 'resources.max_hpo_configs'."
                )
            
            # Log the grid size for visibility
            logging.info(f"HPO Grid Size validated: {total_configs} combinations (Limit: {max_configs})")

        # 2. Memory Limits Check
        # Get system total memory in MB
        system_ram_mb = int(psutil.virtual_memory().total / (1024 * 1024))
        # Default safety buffer: 80% of system RAM
        safe_ram_limit = int(system_ram_mb * 0.8)
        
        config_max_ram = resources.get('max_memory_mb', safe_ram_limit)
        
        if config_max_ram > system_ram_mb:
            logging.warning(
                f"Configured max_memory_mb ({config_max_ram}MB) exceeds physical system RAM ({system_ram_mb}MB). "
                "This may lead to instability."
            )
        
        # Inject the safe limit back into config if not present, for other modules to use
        if 'resources' not in self.config:
            self.config['resources'] = {}
        self.config['resources']['max_memory_mb'] = config_max_ram

    def _propagate_seeds(self) -> None:
        """
        Propagate master seed to internal components to ensure full pipeline reproducibility.
        Uses large, non-overlapping offsets to avoid correlation between components.
        """
        master_seed = self.config['splitting']['seed']
        
        self.config['_internal_seeds'] = {
            'split': master_seed,
            'cv': master_seed + 1000,
            'model': master_seed + 2000,
            'bootstrap': master_seed + 3000,
            'stability_base': master_seed + 10000
        }
        logging.debug(f"Seeds propagated from master ({master_seed}): {self.config['_internal_seeds']}")
