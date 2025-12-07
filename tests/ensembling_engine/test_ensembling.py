import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from modules.ensembling_engine import EnsemblingEngine

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def ens_config(tmp_path):
    return {
        'outputs': {'base_results_dir': str(tmp_path)},
        'ensembling': {
            'enabled': True,
            'strategy': 'simple',
            'weighting_scheme': 'inverse_error'
        }
    }

@pytest.fixture
def model_outputs():
    # Model 1: Predicts 0 degrees (sin=0, cos=1)
    df1 = pd.DataFrame({
        'true_angle': [45.0],
        'pred_angle': [0.0],
        'pred_sin': [0.0],
        'pred_cos': [1.0]
    })
    
    # Model 2: Predicts 90 degrees (sin=1, cos=0)
    df2 = pd.DataFrame({
        'true_angle': [45.0],
        'pred_angle': [90.0],
        'pred_sin': [1.0],
        'pred_cos': [0.0]
    })
    
    return [df1, df2]

def test_simple_average(ens_config, mock_logger, model_outputs):
    # Avg(0, 90) via vectors:
    # sin_avg = (0+1)/2 = 0.5
    # cos_avg = (1+0)/2 = 0.5
    # atan2(0.5, 0.5) = 45 degrees
    
    ens_config['ensembling']['strategy'] = 'simple'
    engine = EnsemblingEngine(ens_config, mock_logger)
    
    result = engine.ensemble(model_outputs, [{}, {}], "test_split", "run1")
    
    assert np.isclose(result['pred_angle'].iloc[0], 45.0)
    assert np.isclose(result['error'].iloc[0], 0.0) # True was 45, Pred is 45 -> Error 0

def test_weighted_average(ens_config, mock_logger, model_outputs):
    # Model 1 Metric: CMAE=10 (Weight ~ 1/10)
    # Model 2 Metric: CMAE=1  (Weight ~ 1/1  -> 10x stronger)
    
    metrics = [{'cmae': 10.0}, {'cmae': 1.0}]
    
    ens_config['ensembling']['strategy'] = 'weighted'
    engine = EnsemblingEngine(ens_config, mock_logger)
    
    result = engine.ensemble(model_outputs, metrics, "test_split", "run1")
    
    # Model 2 (90 deg) dominates. Result should be close to 90.
    # sin_avg ~ (0*0.09 + 1*0.91) = 0.91
    # cos_avg ~ (1*0.09 + 0*0.91) = 0.09
    # atan2(0.91, 0.09) ~ 84 degrees
    
    pred = result['pred_angle'].iloc[0]
    assert pred > 80.0 # Closer to 90 than 0

def test_single_model_fallback(ens_config, mock_logger, model_outputs):
    engine = EnsemblingEngine(ens_config, mock_logger)
    # Pass only 1 model
    result = engine.ensemble([model_outputs[0]], [{}], "test", "run1")
    
    # Should return original without change
    assert len(result) == 1
    assert result['pred_angle'].iloc[0] == 0.0