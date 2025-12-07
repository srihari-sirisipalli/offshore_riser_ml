import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import MagicMock
from modules.prediction_engine import PredictionEngine

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def pred_config(tmp_path):
    return {
        'data': {
            'target_sin': 'target_sin',
            'target_cos': 'target_cos',
            'hs_column': 'Hs',
            'drop_columns': []
        },
        'outputs': {
            'base_results_dir': str(tmp_path),
            'save_predictions': True
        }
    }

def test_prediction_output(pred_config, mock_logger):
    # Mock Model
    class MockModel:
        def predict(self, X):
            # Always return 45 degrees (sin=0.707, cos=0.707)
            return np.array([[0.7071, 0.7071]] * len(X))
            
    # Mock Data (True angle 45 degrees)
    data = pd.DataFrame({
        'feat1': [1, 2],
        'target_sin': [0.7071, 0.7071],
        'target_cos': [0.7071, 0.7071],
        'angle_deg': [45.0, 45.0],
        'Hs': [1.5, 2.0],
        'hs_bin': [0, 1]
    })
    
    engine = PredictionEngine(pred_config, mock_logger)
    df = engine.predict(MockModel(), data, "val", "test_run")
    
    # Check columns
    expected_cols = ['true_angle', 'pred_angle', 'error', 'abs_error', 'Hs', 'hs_bin']
    for col in expected_cols:
        assert col in df.columns
        
    # Check error calculation (should be approx 0)
    assert df['abs_error'].mean() < 0.1
    
    # Check file saving
    base = pred_config['outputs']['base_results_dir']
    assert os.path.exists(os.path.join(base, "06_PREDICTIONS", "predictions_val.xlsx"))