import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock
from modules.data_manager import DataManager

@pytest.fixture
def dm_instance():
    logger = MagicMock(spec=logging.Logger)
    config = {
        'data': {
            'target_sin': 'sin',
            'target_cos': 'cos',
            'hs_column': 'hs'
        },
        'splitting': {
            'angle_bins': 4, # 90 degree bins
            'hs_bins': 2,
            'hs_binning_method': 'equal_width'
        }
    }
    return DataManager(config, logger)

def test_angle_computation(dm_instance):
    # 0 degrees: sin=0, cos=1
    # 90 degrees: sin=1, cos=0
    df = pd.DataFrame({
        'sin': [0.0, 1.0],
        'cos': [1.0, 0.0],
        'hs': [1.0, 1.0]
    })
    dm_instance.data = df
    dm_instance.compute_derived_columns()
    
    # Check angles
    angles = dm_instance.data['angle_deg']
    assert np.isclose(angles[0], 0.0)
    assert np.isclose(angles[1], 90.0)

def test_bin_computation(dm_instance):
    # Angle bins: 4 (0-90, 90-180, 180-270, 270-360)
    # angle=45 -> bin 0
    # angle=135 -> bin 1
    
    # Hs bins: 2 (equal width). 
    # hs=1, hs=10 -> bin 0, bin 1
    
    # Note: angles computed from sin/cos
    # 45 deg: sin=0.707, cos=0.707
    # 135 deg: sin=0.707, cos=-0.707
    
    df = pd.DataFrame({
        'sin': [0.7071, 0.7071], 
        'cos': [0.7071, -0.7071],
        'hs': [1.0, 10.0]
    })
    dm_instance.data = df
    dm_instance.compute_derived_columns()
    
    # Check Angle Bin
    assert dm_instance.data.loc[0, 'angle_bin'] == 0
    assert dm_instance.data.loc[1, 'angle_bin'] == 1
    
    # Check Hs Bin
    assert dm_instance.data.loc[0, 'hs_bin'] == 0
    assert dm_instance.data.loc[1, 'hs_bin'] == 1
    
    # Check Combined Bin (angle_bin * n_hs + hs_bin)
    # Row 0: 0 * 2 + 0 = 0
    # Row 1: 1 * 2 + 1 = 3
    assert dm_instance.data.loc[0, 'combined_bin'] == 0
    assert dm_instance.data.loc[1, 'combined_bin'] == 3