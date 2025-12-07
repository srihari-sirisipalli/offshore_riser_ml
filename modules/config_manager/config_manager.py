import json
import os
import hashlib
import sys
import numpy as np
import jsonschema
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

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
        self._apply_defaults()
        self._validate_logic()
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

    def _apply_defaults(self) -> None:
        # Shallow pass for top-level defaults
        if 'precision' not in self.config['data']:
            self.config['data']['precision'] = 'float32'
        if 'hs_binning_method' not in self.config['splitting']:
            self.config['splitting']['hs_binning_method'] = 'quantile'
        if 'logging' not in self.config:
            self.config['logging'] = {'level': 'INFO', 'log_to_file': True, 'log_to_console': True}

    def _validate_logic(self) -> None:
        test_size = self.config['splitting']['test_size']
        val_size = self.config['splitting']['val_size']
        if test_size + val_size >= 1.0:
            raise ConfigurationError(f"test_size + val_size must be < 1.0")
            
        if self.config.get('hyperparameters', {}).get('enabled', False):
            if not self.config['hyperparameters'].get('grids'):
                raise ConfigurationError("Hyperparameter grids cannot be empty when HPO is enabled")

    def _propagate_seeds(self) -> None:
        master_seed = self.config['splitting']['seed']
        self.config['_internal_seeds'] = {
            'split': master_seed,
            'cv': master_seed + 1,
            'model': master_seed + 2,
            'bootstrap': master_seed + 3,
            'stability_base': master_seed + 100
        }
        np.random.seed(master_seed)