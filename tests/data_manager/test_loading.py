import pytest
import pandas as pd
import os
import logging
from unittest.mock import MagicMock
from modules.data_manager import DataManager
from utils.exceptions import DataValidationError

@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def sample_config(tmp_path):
    # Create a dummy excel file
    df = pd.DataFrame({
        'sin': [0, 1],
        'cos': [1, 0],
        'hs': [2.5, 3.0]
    })
    file_path = tmp_path / "test_data.xlsx"
    df.to_excel(file_path, index=False)
    
    return {
        'data': {
            'file_path': str(file_path),
            'target_sin': 'sin',
            'target_cos': 'cos',
            'hs_column': 'hs',
            'precision': 'float32'
        },
        'outputs': {'base_results_dir': str(tmp_path / "results")},
        'splitting': {'angle_bins': 10, 'hs_bins': 5}
    }

def test_load_data_success(sample_config, mock_logger):
    dm = DataManager(sample_config, mock_logger)
    df = dm.load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df['sin'].dtype == 'float32'

def test_load_data_missing_file(sample_config, mock_logger):
    sample_config['data']['file_path'] = "non_existent.xlsx"
    dm = DataManager(sample_config, mock_logger)
    with pytest.raises(DataValidationError, match="not found"):
        dm.load_data()

def test_missing_columns(sample_config, mock_logger):
    dm = DataManager(sample_config, mock_logger)
    dm.load_data()
    
    # Change config to expect a column that doesn't exist
    sample_config['data']['target_sin'] = 'missing_col'
    
    with pytest.raises(DataValidationError, match="Missing required columns"):
        dm.validate_columns()