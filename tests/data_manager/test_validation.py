import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock
from modules.data_manager import DataManager

@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)

def test_circle_validation_good(mock_logger):
    # Perfect data: sin=1, cos=0 -> magnitude 1
    df = pd.DataFrame({'sin': [1.0], 'cos': [0.0], 'hs': [2.0]})
    config = {
        'data': {'target_sin': 'sin', 'target_cos': 'cos', 'circle_tolerance': 0.01},
        'outputs': {}
    }
    
    dm = DataManager(config, mock_logger)
    dm.data = df
    
    val_df = dm.validate_circle_constraint()
    assert val_df.iloc[0]['status'] == 'GOOD'

def test_circle_validation_bad(mock_logger):
    # Bad data: sin=2, cos=2 -> magnitude >> 1
    df = pd.DataFrame({'sin': [2.0], 'cos': [2.0], 'hs': [2.0]})
    config = {
        'data': {'target_sin': 'sin', 'target_cos': 'cos', 'circle_tolerance': 0.01},
        'outputs': {}
    }
    
    dm = DataManager(config, mock_logger)
    dm.data = df
    
    val_df = dm.validate_circle_constraint()
    assert val_df.iloc[0]['status'] == 'BAD'
    # Check if error log was called (optional)
    mock_logger.error.assert_called()