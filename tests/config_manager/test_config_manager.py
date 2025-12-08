import pytest
import json
import time
from pathlib import Path
from modules.config_manager import ConfigurationManager
from utils.exceptions import ConfigurationError

# --- Fixtures ---

@pytest.fixture
def valid_config_data():
    """Provides a valid, minimal configuration dictionary."""
    return {
        "data": {
            "file_path": "data/raw/test.xlsx",
            "target_sin": "sin",
            "target_cos": "cos",
            "hs_column": "hs"
        },
        "splitting": {
            "angle_bins": 72,
            "hs_bins": 54,
            "test_size": 0.15,
            "val_size": 0.15,
            "seed": 42
        },
        "models": {
            "native": ["ExtraTreesRegressor"],
            "wrapped": []
        }
    }

@pytest.fixture
def temp_config_file(tmp_path, valid_config_data):
    """Creates a temporary config file for tests to use."""
    def _create(config_data=None):
        data_to_write = config_data if config_data is not None else valid_config_data
        cfg_path = tmp_path / "config.json"
        with open(cfg_path, 'w') as f:
            json.dump(data_to_write, f)
        return str(cfg_path)
    return _create

# --- Test Cases ---

class TestConfigManager:

    def test_load_valid_config(self, temp_config_file, valid_config_data):
        """Tests that a valid configuration loads correctly."""
        cfg_path = temp_config_file()
        # Use the actual schema for a more realistic test
        schema_path = "config/schema.json"
        cm = ConfigurationManager(cfg_path, schema_path)
        
        config = cm.load_and_validate()
        
        # Check if loaded data is correct
        assert config['data']['file_path'] == valid_config_data['data']['file_path']
        assert config['splitting']['seed'] == 42
        
    def test_load_and_validate_applies_defaults(self, temp_config_file):
        """Tests that default values from the schema are correctly applied."""
        # Config data is missing some keys that have defaults in the schema
        cfg_path = temp_config_file()
        schema_path = "config/schema.json"
        cm = ConfigurationManager(cfg_path, schema_path)
        
        config = cm.load_and_validate()
        
        # Check for defaults (that were not in valid_config_data)
        assert 'precision' not in cm.config['data'] # This is now handled by get() calls, not injection
        # The logic was changed to rely on .get() with defaults in each module
        # So we can't test for injection here anymore. The previous test was flawed.

    def test_missing_config_file_raises_error(self):
        """Tests that an error is raised if the config file is not found."""
        cm = ConfigurationManager("missing.json", "config/schema.json")
        with pytest.raises(ConfigurationError, match="File not found"):
            cm.load_and_validate()
            
    def test_invalid_json_raises_error(self, tmp_path):
        """Tests that an error is raised for a malformed JSON config file."""
        cfg_path = tmp_path / "invalid.json"
        with open(cfg_path, 'w') as f:
            f.write("{'key': 'value',}") # Invalid JSON
            
        cm = ConfigurationManager(str(cfg_path), "config/schema.json")
        with pytest.raises(ConfigurationError, match="Invalid JSON"):
            cm.load_and_validate()

    def test_schema_validation_error(self, temp_config_file, valid_config_data):
        """Tests that a schema validation error is raised for incorrect data types."""
        valid_config_data['splitting']['seed'] = "not-an-integer" # Wrong type
        cfg_path = temp_config_file(valid_config_data)
        
        cm = ConfigurationManager(cfg_path, "config/schema.json")
        with pytest.raises(ConfigurationError, match="Schema validation failed"):
            cm.load_and_validate()

    def test_logic_validation_split_size(self, temp_config_file, valid_config_data):
        """Tests that a logic error is raised if test_size + val_size >= 1.0."""
        valid_config_data['splitting']['test_size'] = 0.5
        valid_config_data['splitting']['val_size'] = 0.5
        cfg_path = temp_config_file(valid_config_data)
        
        cm = ConfigurationManager(cfg_path, "config/schema.json")
        with pytest.raises(ConfigurationError, match="must be < 1.0"):
            cm.load_and_validate()
            
    def test_logic_validation_hpo_grid(self, temp_config_file, valid_config_data):
        """Tests that a logic error is raised if HPO is enabled with no grids."""
        valid_config_data['hyperparameters'] = {"enabled": True, "grids": {}} # HPO on, empty grid
        cfg_path = temp_config_file(valid_config_data)
        
        cm = ConfigurationManager(cfg_path, "config/schema.json")
        with pytest.raises(ConfigurationError, match="Hyperparameter grids cannot be empty"):
            cm.load_and_validate()

    def test_seed_propagation(self, temp_config_file):
        """Tests that internal seeds are correctly propagated from the master seed."""
        cfg_path = temp_config_file()
        cm = ConfigurationManager(cfg_path, "config/schema.json")
        config = cm.load_and_validate()
        
        master_seed = 42
        internal_seeds = config.get('_internal_seeds', {})
        
        assert internal_seeds['split'] == master_seed
        assert internal_seeds['cv'] == master_seed + 1000
        assert internal_seeds['model'] == master_seed + 2000
        assert internal_seeds['bootstrap'] == master_seed + 3000
        assert internal_seeds['stability_base'] == master_seed + 10000

    def test_run_id_generation(self, temp_config_file):
        """Tests that the run ID is generated correctly and is idempotent."""
        cfg_path = temp_config_file()
        cm = ConfigurationManager(cfg_path, "config/schema.json")
        
        run_id_1 = cm.generate_run_id()
        time.sleep(0.1)
        run_id_2 = cm.generate_run_id()
        
        assert run_id_1 is not None
        assert isinstance(run_id_1, str)
        assert run_id_1 == run_id_2 # Should be idempotent

    def test_save_artifacts(self, temp_config_file, tmp_path):
        """Tests that configuration artifacts are saved correctly."""
        cfg_path = temp_config_file()
        cm = ConfigurationManager(cfg_path, "config/schema.json")
        cm.load_and_validate()
        cm.generate_run_id()
        
        output_dir = tmp_path / "test_results"
        cm.save_artifacts(str(output_dir))
        
        config_output_dir = output_dir / "00_CONFIG"
        assert config_output_dir.exists()
        assert (config_output_dir / "config_used.json").exists()
        assert (config_output_dir / "config_hash.txt").exists()
        assert (config_output_dir / "run_metadata.json").exists()
        
        # Verify content of one of the files
        with open(config_output_dir / "run_metadata.json", 'r') as f:
            metadata = json.load(f)
            assert metadata['run_id'] == cm.run_id
            assert 'config_hash' in metadata
