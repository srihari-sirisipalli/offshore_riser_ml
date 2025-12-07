import pytest
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from unittest.mock import MagicMock
from modules.hpo_search_engine import HPOSearchEngine

@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def sample_data():
    # Create dummy dataset suitable for HPO
    n = 20
    df = pd.DataFrame({
        'feature_1': np.random.rand(n),
        'feature_2': np.random.rand(n),
        'target_sin': np.sin(np.linspace(0, 3, n)),
        'target_cos': np.cos(np.linspace(0, 3, n)),
        'combined_bin': [0]*(n//2) + [1]*(n//2)
    })
    return df

@pytest.fixture
def hpo_config(tmp_path):
    return {
        'data': {
            'target_sin': 'target_sin',
            'target_cos': 'target_cos',
            'drop_columns': []
        },
        'outputs': {'base_results_dir': str(tmp_path)},
        'execution': {'n_jobs': 1}, # Sequential for test stability
        'hyperparameters': {
            'enabled': True,
            'cv_folds': 2,
            'grids': {
                'ExtraTreesRegressor': {
                    'n_estimators': [5, 10], # Small grid
                    'max_depth': [3]
                }
            }
        }
    }

def test_hpo_execution(hpo_config, sample_data, mock_logger):
    """Test full execution of HPO engine."""
    hpo = HPOSearchEngine(hpo_config, mock_logger)
    best_config = hpo.execute(sample_data, "test_run")
    
    assert best_config is not None
    assert 'model' in best_config
    assert 'params' in best_config
    assert best_config['model'] == 'ExtraTreesRegressor'
    
    # FIX: Use hpo.progress_file directly
    assert hpo.progress_file.exists()
    
    # Verify best config artifact
    output_dir = hpo.progress_file.parent
    assert (output_dir / "best_config.json").exists()

def test_resume_capability(hpo_config, sample_data, mock_logger):
    """Test that HPO skips already computed configs."""
    hpo = HPOSearchEngine(hpo_config, mock_logger)
    
    # 1. Run once (creates progress file)
    hpo.execute(sample_data, "test_run")
    
    # FIX: Use hpo.progress_file directly
    progress_file = hpo.progress_file
    assert progress_file.exists()
    
    with open(progress_file, 'r') as f:
        lines_initial = len(f.readlines())
        
    # 3. Run again (should do nothing new)
    hpo_2 = HPOSearchEngine(hpo_config, mock_logger)
    hpo_2.execute(sample_data, "test_run")
    
    # Check the file again
    # We must construct path again or assume same config -> same path
    # But hpo_2.progress_file will be set after execute()
    progress_file_2 = hpo_2.progress_file
    
    with open(progress_file_2, 'r') as f:
        lines_final = len(f.readlines())
        
    # Lines should be identical (Header + 2 configs)
    assert lines_initial == lines_final