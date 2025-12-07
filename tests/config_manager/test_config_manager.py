import pytest
import json
from pathlib import Path
from modules.config_manager import ConfigurationManager
from utils.exceptions import ConfigurationError

@pytest.fixture
def valid_config_data():
    return {
        "data": { "file_path": "test.xlsx", "target_sin": "sin", "target_cos": "cos", "hs_column": "hs" },
        "splitting": { "angle_bins": 72, "hs_bins": 54, "test_size": 0.15, "val_size": 0.15, "seed": 42 },
        "models": { "native": ["ExtraTreesRegressor"], "wrapped": [] }
    }

@pytest.fixture
def temp_files(tmp_path, valid_config_data):
    cfg_path = tmp_path / "config.json"
    schema_path = Path("config/schema.json")
    with open(cfg_path, 'w') as f:
        json.dump(valid_config_data, f)
    return str(cfg_path), str(schema_path)

def test_load_valid_config(temp_files):
    cfg_path, schema_path = temp_files
    cm = ConfigurationManager(cfg_path, schema_path)
    config = cm.load_and_validate()
    assert config['data']['file_path'] == "test.xlsx"
    assert config['data']['precision'] == "float32" # Default injection

def test_missing_file():
    cm = ConfigurationManager("missing.json", "config/schema.json")
    with pytest.raises(ConfigurationError, match="File not found"):
        cm.load_and_validate()

def test_logic_error(temp_files, valid_config_data):
    cfg_path, schema_path = temp_files
    
    # FIX: Use 0.5 (valid schema) so sum is 1.0 (invalid logic)
    valid_config_data['splitting']['test_size'] = 0.5 
    valid_config_data['splitting']['val_size'] = 0.5
    
    with open(cfg_path, 'w') as f:
        json.dump(valid_config_data, f)
        
    cm = ConfigurationManager(cfg_path, schema_path)
    with pytest.raises(ConfigurationError, match="must be < 1.0"):
        cm.load_and_validate()