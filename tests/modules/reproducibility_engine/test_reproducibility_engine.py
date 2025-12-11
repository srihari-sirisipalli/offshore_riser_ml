import pytest
from unittest.mock import Mock, patch, call
import pandas as pd
import numpy as np
import json
import shutil
import subprocess # For pip freeze
from pathlib import Path
from modules.reproducibility_engine.reproducibility_engine import ReproducibilityEngine

# Concrete implementation for testing ReproducibilityEngine
class ConcreteReproducibilityEngine(ReproducibilityEngine):
    def __init__(self, config, logger, output_dir_name="13_REPRODUCIBILITY_PACKAGE", standard_output_dir_name="96_REPRODUCIBILITY_PACKAGE"):
        self._test_output_dir_name = output_dir_name
        self._test_standard_output_dir_name = standard_output_dir_name
        super().__init__(config, logger)
        self.output_dir = Path(config['outputs']['base_results_dir']) / output_dir_name
        self.standard_output_dir = Path(config['outputs']['base_results_dir']) / standard_output_dir_name

    def _get_engine_directory_name(self) -> str:
        return self._test_output_dir_name

@pytest.fixture
def mock_logger():
    """Provides a mock logger instance."""
    return Mock()

@pytest.fixture
def base_config(tmp_path):
    """Provides a base configuration dictionary with a temporary results directory."""
    return {
        'outputs': {
            'base_results_dir': str(tmp_path),
            'archive_reproducibility_package': True # Enabled by default for tests
        },
        'data': { # Minimal data config
            'target_sin': 'target_sin',
            'target_cos': 'target_cos',
            'drop_columns': [],
            'hs_column': 'Hs_ft',
        },
        'logging': {'level': 'INFO'}, # Mock config sections to avoid warnings/errors
        'resource_limits': {}
    }

@pytest.fixture
def reproducibility_engine(base_config, mock_logger):
    """Provides a ReproducibilityEngine instance."""
    return ConcreteReproducibilityEngine(base_config, mock_logger)

@pytest.fixture(autouse=True)
def mock_shutil_functions():
    """Mocks shutil.rmtree, shutil.copy2, shutil.copytree."""
    with patch('shutil.rmtree') as mock_rmtree, \
         patch('shutil.copy2') as mock_copy2, \
         patch('shutil.copytree') as mock_copytree:
        yield mock_rmtree, mock_copy2, mock_copytree

@pytest.fixture(autouse=True)
def mock_subprocess_run():
    """Mocks subprocess.run for pip freeze."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "numpy==1.23.0\npandas==1.5.0"
    with patch('subprocess.run', return_value=mock_result) as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_json_dump():
    """Mocks json.dump."""
    with patch('json.dump') as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_path_write_text():
    """Mocks Path.write_text."""
    with patch('pathlib.Path.write_text') as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_save_dataframe():
    """Mocks save_dataframe from utils.file_io."""
    with patch('modules.reproducibility_engine.reproducibility_engine.save_dataframe') as mock:
        yield mock

# --- Test cases for ReproducibilityEngine ---

def test_execute_copies_to_standard_dir(reproducibility_engine, mock_shutil_functions, tmp_path, mock_logger):
    """
    Tests that execute copies the package to the standard directory when different from output_dir.
    """
    mock_rmtree, mock_copy2, mock_copytree = mock_shutil_functions
    
    # Ensure source_root exists for artifact copying simulation
    source_root = tmp_path
    (source_root / "00_CONFIG").mkdir()
    (source_root / "05_FINAL_MODEL").mkdir() # Required by _copy_artifacts
    (source_root / "06_PREDICTIONS").mkdir() # Required by _copy_artifacts
    (source_root / "07_EVALUATION").mkdir() # Required by _copy_artifacts
    (source_root / "08_EVALUATION").mkdir() # Required by _copy_artifacts
    (source_root / "10_REPORT").mkdir() # Required by _copy_artifacts

    # Create some dummy files so copy2 doesn't fail on non-existent sources
    (source_root / "00_CONFIG" / "config_used.json").touch()
    (source_root / "00_CONFIG" / "run_metadata.json").touch()

    run_id = "test_run_001"
    package_dir = tmp_path / "13_REPRODUCIBILITY_PACKAGE"
    standard_dir = tmp_path / "96_REPRODUCIBILITY_PACKAGE"

    # Ensure output_dir and standard_output_dir are correctly set on the engine
    reproducibility_engine.output_dir = package_dir
    reproducibility_engine.standard_output_dir = standard_dir
    
    # Ensure output_dir and standard_dir exist for rmtree to be called
    package_dir.mkdir(parents=True, exist_ok=True)
    standard_dir.mkdir(parents=True, exist_ok=True)
    
    returned_path = reproducibility_engine.execute(run_id)
    
    # Assert rmtree calls
    mock_rmtree.assert_has_calls([
        call(package_dir),
        call(standard_dir, ignore_errors=True),
    ], any_order=True)
    
    # Assert copytree to standard_dir
    mock_copytree.assert_called_once_with(package_dir, standard_dir)
    
    assert returned_path == str(package_dir)
    mock_logger.info.assert_any_call(f"Reproducibility Package successfully created at: {package_dir.absolute()}")

def test_execute_does_not_copy_if_same_dir(reproducibility_engine, mock_shutil_functions, tmp_path, mock_logger):
    """
    Tests that execute does not copy to standard_dir if it's the same as output_dir.
    """
    mock_rmtree, mock_copy2, mock_copytree = mock_shutil_functions
    
    source_root = tmp_path
    (source_root / "00_CONFIG").mkdir()
    (source_root / "05_FINAL_MODEL").mkdir()
    (source_root / "06_PREDICTIONS").mkdir()
    (source_root / "07_EVALUATION").mkdir()
    (source_root / "08_EVALUATION").mkdir()
    (source_root / "10_REPORT").mkdir()
    (source_root / "00_CONFIG" / "config_used.json").touch()
    (source_root / "00_CONFIG" / "run_metadata.json").touch()

    run_id = "test_run_002"
    package_dir = tmp_path / "13_REPRODUCibility_PACKAGE"
    
    # Set output_dir and standard_output_dir to be the same
    reproducibility_engine.output_dir = package_dir
    reproducibility_engine.standard_output_dir = package_dir
    
    returned_path = reproducibility_engine.execute(run_id)
    
    # Assert rmtree is called for package_dir
    mock_rmtree.assert_called_with(package_dir)
    assert package_dir.exists() # Due to package_dir.mkdir(parents=True, exist_ok=True) being real
    
    # Assert copytree was NOT called
    mock_copytree.assert_not_called()
    
    assert returned_path == str(package_dir)
    mock_logger.info.assert_any_call(f"Reproducibility Package successfully created at: {package_dir.absolute()}")

def test_execute_disabled(reproducibility_engine, mock_shutil_functions, mock_logger):
    """Tests that execute is skipped if disabled in config."""
    mock_rmtree, mock_copy2, mock_copytree = mock_shutil_functions
    
    reproducibility_engine.config['outputs']['archive_reproducibility_package'] = False
    reproducibility_engine.enabled = False # Directly set for test as it's read in __init__
    
    returned_path = reproducibility_engine.execute("disabled_run")
    
    mock_logger.info.assert_called_with("Reproducibility packaging disabled in config.")
    mock_rmtree.assert_not_called()
    mock_copytree.assert_not_called()
    assert returned_path == ""

