import pytest
import pandas as pd
import logging
from unittest.mock import MagicMock
from modules.split_engine import SplitEngine

@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)

def test_fallback_strategy(mock_logger, tmp_path):
    """
    Test scenario where combined_bin is too rare (singleton),
    forcing fallback to hs_bin.
    """
    # 100 rows, unique combined_bin for everyone (100 bins)
    # But only 2 hs_bins
    df = pd.DataFrame({
        'combined_bin': range(100), # All singletons
        'hs_bin': [0]*50 + [1]*50,  # Stratifiable
        'angle_bin': [0]*100,
        'Hs': range(100)
    })
    
    config = {
        'splitting': {'test_size': 0.1, 'val_size': 0.1, 'seed': 42},
        'outputs': {'base_results_dir': str(tmp_path)}
    }
    
    splitter = SplitEngine(config, mock_logger)
    train, val, test = splitter.execute(df, "test_run")
    
    # Logger should warn/info about fallback
    # mock_logger.info.assert_any_call("Too many rare combined bins. Falling back to 'hs_bin' stratification.")
    
    # Check sizes
    assert len(test) == 10
    assert len(val) == 10
    assert len(train) == 80
    
    # Check if hs_bin was stratified (5 from bin 0, 5 from bin 1 in test)
    assert test['hs_bin'].value_counts()[0] == 5
    assert test['hs_bin'].value_counts()[1] == 5