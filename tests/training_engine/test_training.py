import pytest
import pandas as pd
import numpy as np
import os
import joblib
from unittest.mock import MagicMock
from modules.training_engine import TrainingEngine

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [4, 5, 6],
        'target_sin': [0, 1, 0],
        'target_cos': [1, 0, -1],
        'angle_deg': [0, 90, 180]
    })

@pytest.fixture
def train_config(tmp_path):
    return {
        'data': {
            'target_sin': 'target_sin',
            'target_cos': 'target_cos',
            'drop_columns': []
        },
        'outputs': {
            'base_results_dir': str(tmp_path),
            'save_models': True
        }
    }

def test_training_execution(train_config, sample_data, mock_logger):
    engine = TrainingEngine(train_config, mock_logger)
    
    model_config = {
        'model': 'ExtraTreesRegressor',
        'params': {'n_estimators': 5}
    }
    
    model = engine.train(sample_data, model_config, "test_run")
    
    # Check if model object returned
    assert hasattr(model, 'predict')
    
    # Check if file saved
    base = train_config['outputs']['base_results_dir']
    assert os.path.exists(os.path.join(base, "05_FINAL_MODEL", "final_model.pkl"))
    assert os.path.exists(os.path.join(base, "05_FINAL_MODEL", "training_metadata.json"))