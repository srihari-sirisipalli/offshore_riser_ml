import json
import os
import hashlib
import sys
import numpy as np
import jsonschema
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterGrid

from utils.exceptions import ConfigurationError

class ConfigurationManager:
    """
    Manages system configuration loading, validation, and access.
    Acts as the single source of truth for the pipeline.
    """
    
    def __init__(self, config_path: str = "config/config.json",
                 schema_path: str = "config/schema.json"):
        self.config_path = config_path
        self.schema_path = schema_path
        self.config: Dict[str, Any] = {}
        self.schema: Dict[str, Any] = {}
        self.run_id: Optional[str] = None
        
    def load_and_validate(self) -> Dict[str, Any]:
        """Load config, validate against schema/logic, apply defaults, propagate seeds."""
        self.config = self._load_json(self.config_path)
        self.schema = self._load_json(self.schema_path)
        
        self._validate_schema()
        self._validate_logic()
        self._validate_resources()
        self._propagate_seeds()
        
        return self.config

    def generate_run_id(self) -> str:
        """Generate unique run identifier based on timestamp."""
        if not self.run_id:
            timestamp = datetime.now()
            self.run_id = timestamp.strftime("%Y%m%d_%H%M%S")
        return self.run_id

    def save_artifacts(self, output_dir: str) -> None:
        """Save configuration artifacts (used config, hash, metadata)."""
        config_dir = Path(output_dir) / "00_CONFIG"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_dir / "config_used.json", 'w') as f:
            json.dump(self.config, f, indent=2)
            
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        with open(config_dir / "config_hash.txt", 'w') as f:
            f.write(config_hash)
            
        metadata = {
            'run_id': self.run_id,
            'start_time': datetime.now().isoformat(),
            'python_version': sys.version,
            'config_hash': config_hash
        }
        
        with open(config_dir / "run_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_json(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise ConfigurationError(f"File not found: {path}")
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {path}: {str(e)}")

    def _validate_schema(self) -> None:
        try:
            jsonschema.validate(instance=self.config, schema=self.schema)
        except jsonschema.ValidationError as e:
            raise ConfigurationError(f"Schema validation failed: {e.message}")

    def _validate_logic(self) -> None:
        """Comprehensive logical validation."""
        # Split sizes
        test_size = self.config['splitting']['test_size']
        val_size = self.config['splitting']['val_size']
        
        if not (0 < test_size < 1.0):
            raise ConfigurationError(f"test_size must be in (0, 1), got {test_size}")
        if not (0 < val_size < 1.0):
            raise ConfigurationError(f"val_size must be in (0, 1), got {val_size}")
        if test_size + val_size >= 1.0:
            raise ConfigurationError("test_size + val_size must be < 1.0")
        
        # Seeds must be non-negative
        if self.config['splitting']['seed'] < 0:
            raise ConfigurationError("seed must be non-negative")
        
        # HPO validation
        if self.config.get('hyperparameters', {}).get('enabled', False):
            grids = self.config['hyperparameters'].get('grids')
            if not grids:
                raise ConfigurationError("Hyperparameter grids cannot be empty")
            
            cv_folds = self.config['hyperparameters'].get('cv_folds', 5)
            if cv_folds < 2:
                raise ConfigurationError(f"cv_folds must be >= 2, got {cv_folds}")
        
        # Bootstrap validation
        if self.config.get('bootstrapping', {}).get('enabled', False):
            confidence = self.config['bootstrapping'].get('confidence_level', 0.95)
            if not (0 < confidence < 1):
                raise ConfigurationError(f"confidence_level must be in (0, 1)")
            
            sample_ratio = self.config['bootstrapping'].get('sample_ratio', 1.0)
            if sample_ratio <= 0:
                raise ConfigurationError(f"sample_ratio must be > 0")
        
        # HPO Analysis validation
        if self.config.get('hpo_analysis', {}):
            top_percent = self.config['hpo_analysis'].get('optimal_top_percent', 10)
            if not (0 < top_percent <= 100):
                raise ConfigurationError(f"optimal_top_percent must be in (0, 100]")

    def _validate_resources(self) -> None:
        """Validate against resource limits."""
        if self.config.get('hyperparameters', {}).get('enabled', False):
            hpo_configs = sum(
                len(list(ParameterGrid(grid)))
                for grid in self.config.get('hyperparameters', {}).get('grids', {}).values()
            )

            max_configs = self.config.get('resources', {}).get('max_hpo_configs', 1000)
            if hpo_configs > max_configs:
                raise ConfigurationError(
                    f"HPO grid would generate {hpo_configs} configs, "
                    f"exceeding limit of {max_configs}"
                )

    def _propagate_seeds(self) -> None:
        master_seed = self.config['splitting']['seed']
        # FIX #34: Use larger, non-overlapping offsets for seeds to improve isolation.
        self.config['_internal_seeds'] = {
            'split': master_seed,
            'cv': master_seed + 1000,
            'model': master_seed + 2000,
            'bootstrap': master_seed + 3000,
            'stability_base': master_seed + 10000
        }