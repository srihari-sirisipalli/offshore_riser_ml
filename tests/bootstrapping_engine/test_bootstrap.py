import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path
from modules.bootstrapping_engine import BootstrappingEngine

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def boot_config(tmp_path):
    return {
        'outputs': {'base_results_dir': str(tmp_path)},
        'bootstrapping': {
            'enabled': True,
            'num_samples': 50, # Small number for fast testing
            'confidence_level': 0.95,
            'sample_ratio': 1.0
        }
    }

@pytest.fixture
def sample_predictions():
    # 100 samples
    return pd.DataFrame({
        'true_angle': np.random.uniform(0, 360, 100),
        'pred_angle': np.random.uniform(0, 360, 100),
        'abs_error': np.random.uniform(0, 10, 100) # Random error 0-10
    })

def test_bootstrap_execution(boot_config, mock_logger, sample_predictions):
    engine = BootstrappingEngine(boot_config, mock_logger)
    ci_results = engine.bootstrap(sample_predictions, "test_split", "run1")
    
    # Check return structure
    assert 'cmae' in ci_results
    assert 'accuracy_5deg' in ci_results
    assert 'lower' in ci_results['cmae']
    assert 'upper' in ci_results['cmae']
    
    # Check logic: Lower CI should be <= Mean <= Upper CI
    cmae_stats = ci_results['cmae']
    assert cmae_stats['lower'] <= cmae_stats['mean']
    assert cmae_stats['mean'] <= cmae_stats['upper']

def test_bootstrap_artifacts(boot_config, mock_logger, sample_predictions):
    engine = BootstrappingEngine(boot_config, mock_logger)
    engine.bootstrap(sample_predictions, "test_split", "run1")
    
    base = Path(boot_config['outputs']['base_results_dir'])
    out_dir = base / "09_ADVANCED_ANALYTICS" / "bootstrapping" / "test_split"
    
    assert (out_dir / "bootstrap_ci.xlsx").exists()
    assert (out_dir / "bootstrap_samples.xlsx").exists()
    assert (out_dir / "bootstrap_dist_cmae.png").exists()

def test_disabled_bootstrap(boot_config, mock_logger, sample_predictions):
    boot_config['bootstrapping']['enabled'] = False
    engine = BootstrappingEngine(boot_config, mock_logger)
    res = engine.bootstrap(sample_predictions, "test_split", "run1")
    assert res == {}