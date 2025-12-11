import pytest
from unittest.mock import Mock, patch, call
import json
import os
import hashlib
import sys
import logging
import numpy as np
import jsonschema
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.model_selection import ParameterGrid

from modules.config_manager.config_manager import ConfigurationManager
from utils.exceptions import ConfigurationError

@pytest.fixture
def mock_logger():
    """Provides a mock logger instance."""
    return Mock()

@pytest.fixture
def config_manager(tmp_path):
    """Provides a ConfigurationManager instance with dummy config and schema paths."""
    config_path = tmp_path / "config.json"
    schema_path = tmp_path / "schema.json"
    return ConfigurationManager(str(config_path), str(schema_path))

@pytest.fixture
def mock_files(tmp_path):
    """Creates dummy config and schema files."""
    config_path = tmp_path / "config.json"
    schema_path = tmp_path / "schema.json"

    valid_config_content = {
        "data": {
            "file_path": "data/my_data.csv",
            "hs_column": "Hs_ft",
            "target_sin": "RiserAngle_Sin",
            "target_cos": "RiserAngle_Cos",
            "drop_columns": []
        },
        "splitting": {
            "test_size": 0.2,
            "val_size": 0.25,
            "seed": 42
        },
        "hyperparameters": {"enabled": False},
        "iterative": {"enabled": False},
        "bootstrapping": {"enabled": False},
        "outputs": {"base_results_dir": str(tmp_path / "results")},
        "execution": {"n_jobs": -1}
    }
    
    # A simplified schema for testing purposes
    valid_schema_content = {
        "type": "object",
        "properties": {
            "data": {"type": "object", "properties": {"file_path": {"type": "string"}}},
            "splitting": {"type": "object", "properties": {"test_size": {"type": "number"}}},
            "hyperparameters": {"type": "object", "properties": {"enabled": {"type": "boolean"}}},
            "iterative": {"type": "object", "properties": {"enabled": {"type": "boolean"}}},
            "bootstrapping": {"type": "object", "properties": {"enabled": {"type": "boolean"}}},
            "outputs": {"type": "object", "properties": {"base_results_dir": {"type": "string"}}},
            "execution": {"type": "object", "properties": {"n_jobs": {"type": "number"}}},
        },
        "required": ["data", "splitting", "hyperparameters", "iterative", "bootstrapping", "outputs", "execution"]
    }

    config_path.write_text(json.dumps(valid_config_content))
    schema_path.write_text(json.dumps(valid_schema_content))
    
    return config_path, schema_path, valid_config_content, valid_schema_content


@pytest.fixture(autouse=True)
def mock_external_deps():
    """Mocks external dependencies like jsonschema, psutil, datetime, hashlib."""
    with patch('jsonschema.validate') as mock_jsonschema_validate, \
         patch('psutil.virtual_memory') as mock_psutil_virtual_memory, \
         patch('modules.config_manager.config_manager.ParameterGrid') as mock_parameter_grid, \
         patch('datetime.datetime') as mock_datetime_datetime, \
         patch('hashlib.sha256') as mock_hashlib_sha256, \
         patch('os.path.exists', return_value=True) as mock_os_path_exists, \
         patch('os.getcwd', return_value="/mock/cwd") as mock_os_getcwd:
        
        # Configure psutil to return a reasonable memory value
        mock_virtual_memory = Mock()
        mock_virtual_memory.total = 8 * (1024 ** 3) # 8 GB
        mock_psutil_virtual_memory.return_value = mock_virtual_memory
        
        # Configure ParameterGrid to return a simple list of combinations
        mock_parameter_grid.return_value = [1, 2, 3, 4, 5] # 5 combinations

        # Configure datetime.now() for predictable run_id
        mock_datetime_datetime.now.return_value = datetime(2023, 10, 27, 10, 30, 0)
        
        # Configure hashlib for predictable hash
        mock_hash_instance = Mock()
        mock_hash_instance.hexdigest.return_value = "mocked_hash_value"
        mock_hashlib_sha256.return_value = mock_hash_instance

        yield mock_jsonschema_validate, mock_psutil_virtual_memory, mock_parameter_grid, \
              mock_datetime_datetime, mock_hashlib_sha256, mock_os_path_exists, mock_os_getcwd

@pytest.fixture(autouse=True)
def mock_open():
    """Mocks built-in open for JSON file operations."""
    m = mock_open(read_data='{}') # Default empty JSON content
    with patch('builtins.open', m) as mock_builtin_open:
        yield mock_builtin_open

# --- Test Cases ---

def test_load_and_validate_success(config_manager, mock_files, mock_external_deps, mock_open):
    """Tests successful loading and validation of a valid configuration."""
    _, _, valid_config_content, valid_schema_content = mock_files
    mock_jsonschema_validate, mock_psutil_virtual_memory, mock_parameter_grid, \
        mock_datetime_datetime, mock_hashlib_sha256, mock_os_path_exists, mock_os_getcwd = mock_external_deps
    
    # Configure mock_open to return specific content for config and schema paths
    mock_open.side_effect = [
        mock_open(read_data=json.dumps(valid_config_content)).return_value, # For config
        mock_open(read_data=json.dumps(valid_schema_content)).return_value  # For schema
    ]

    config = config_manager.load_and_validate()
    
    # Assertions for successful load and validation
    assert config['data']['hs_column'] == "Hs_ft"
    assert config['splitting']['seed'] == 42
    
    mock_jsonschema_validate.assert_called_once_with(instance=valid_config_content, schema=valid_schema_content)
    mock_psutil_virtual_memory.assert_called_once()
    
    # Assert seeds are propagated
    assert '_internal_seeds' in config
    assert config['_internal_seeds']['split'] == 42
    assert config['_internal_seeds']['model'] == 42 + 2000
    
    # Assert memory limit is set
    assert 'max_memory_mb' in config['resources']

def test_load_and_validate_config_not_found(config_manager, mock_external_deps):
    """Tests error handling when config file is not found."""
    _, _, _, _ = mock_external_deps
    mock_os_path_exists = mock_external_deps[5]
    mock_os_path_exists.side_effect = [False, True] # config not found, then schema found
    
    with pytest.raises(ConfigurationError, match="File not found: .*config.json"):
        config_manager.load_and_validate()

def test_load_and_validate_invalid_json(config_manager, mock_external_deps, mock_open):
    """Tests error handling for invalid JSON content."""
    _, _, _, _ = mock_external_deps
    mock_open.side_effect = [
        mock_open(read_data="this is not valid json").return_value, # For config
        mock_open(read_data='{}').return_value # For schema
    ]
    
    with pytest.raises(ConfigurationError, match="Invalid JSON in .*config.json"):
        config_manager.load_and_validate()

def test_load_and_validate_schema_failure(config_manager, mock_files, mock_external_deps, mock_open):
    """Tests error handling when schema validation fails."""
    _, _, valid_config_content, valid_schema_content = mock_files
    mock_jsonschema_validate = mock_external_deps[0]
    mock_jsonschema_validate.side_effect = jsonschema.ValidationError("Schema error message")
    
    mock_open.side_effect = [
        mock_open(read_data=json.dumps(valid_config_content)).return_value, # For config
        mock_open(read_data=json.dumps(valid_schema_content)).return_value  # For schema
    ]

    with pytest.raises(ConfigurationError, match="Schema validation failed: Schema error message"):
        config_manager.load_and_validate()

def test_validate_logic_missing_hs_column(config_manager, mock_files, mock_external_deps, mock_open):
    """Tests logical validation for missing hs_column."""
    _, _, valid_config_content, valid_schema_content = mock_files
    # Modify config to be invalid logically
    invalid_config = valid_config_content.copy()
    del invalid_config['data']['hs_column']

    mock_open.side_effect = [
        mock_open(read_data=json.dumps(invalid_config)).return_value, # For config
        mock_open(read_data=json.dumps(valid_schema_content)).return_value  # For schema
    ]
    
    with pytest.raises(ConfigurationError, match="Data 'hs_column' must be specified."):
        config_manager.load_and_validate()

def test_validate_resources_hpo_explosion(config_manager, mock_files, mock_external_deps, mock_open, mock_logger):
    """Tests resource validation for HPO grid explosion."""
    _, _, valid_config_content, valid_schema_content = mock_files
    mock_parameter_grid = mock_external_deps[2]

    # Modify config to enable HPO and trigger explosion
    hpo_config = valid_config_content.copy()
    hpo_config['hyperparameters'] = {"enabled": True, "grids": {"model1": {"param1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10}}} # 1000 combinations
    hpo_config['resources'] = {"max_hpo_configs": 500} # Max is 500, we'll have 1000
    
    mock_parameter_grid.return_value = [None] * 1000 # Simulate 1000 combinations
    
    mock_open.side_effect = [
        mock_open(read_data=json.dumps(hpo_config)).return_value, # For config
        mock_open(read_data=json.dumps(valid_schema_content)).return_value  # For schema
    ]

    with pytest.raises(ConfigurationError, match="HPO Grid Explosion Detected!"):
        config_manager.load_and_validate()
    
    mock_logger.info.assert_called_with("HPO Grid Size validated: 1000 combinations (Limit: 500)")

def test_generate_run_id(config_manager, mock_external_deps):
    """Tests that generate_run_id creates a unique timestamp-based ID."""
    mock_datetime_datetime = mock_external_deps[3]
    mock_datetime_datetime.now.return_value = datetime(2023, 10, 27, 10, 30, 0)
    
    run_id = config_manager.generate_run_id()
    assert run_id == "20231027_103000"
    assert config_manager.run_id == "20231027_103000" # Should be stored

def test_save_artifacts(config_manager, mock_files, mock_external_deps, mock_open, tmp_path):
    """Tests that save_artifacts saves all expected files."""
    _, _, valid_config_content, _ = mock_files
    
    mock_hashlib_sha256 = mock_external_deps[4]
    mock_os_getcwd = mock_external_deps[6]
    
    # We need a configured config_manager before saving artifacts
    config_manager.config = valid_config_content
    config_manager.run_id = config_manager.generate_run_id() # Generate a run_id
    
    output_dir = tmp_path / "my_results"
    
    # Set up mock_open for writing.
    # It will be called multiple times for config_used.json, config_hash.txt, run_metadata.json
    mock_file_handles = [Mock(), Mock(), Mock()] # For 3 files
    mock_open.side_effect = mock_file_handles
    
    # Mock Path.mkdir
    with patch('pathlib.Path.mkdir') as mock_path_mkdir:
        config_manager.save_artifacts(str(output_dir))
        
        mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # Assert config_used.json write
        mock_file_handles[0].__enter__().write.assert_called_once()
        json.loads(mock_file_handles[0].__enter__().write.call_args[0][0]) # Check it wrote valid JSON

        # Assert config_hash.txt write
        mock_file_handles[1].__enter__().write.assert_called_once_with("mocked_hash_value")
        mock_hashlib_sha256.assert_called_once()
        
        # Assert run_metadata.json write
        mock_file_handles[2].__enter__().write.assert_called_once()
        metadata = json.loads(mock_file_handles[2].__enter__().write.call_args[0][0])
        assert metadata['run_id'] == config_manager.run_id
        assert metadata['config_hash'] == "mocked_hash_value"
        assert metadata['working_directory'] == "/mock/cwd"
