import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil
from unittest.mock import MagicMock
from modules.diagnostics_engine import DiagnosticsEngine

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def sample_preds():
    n = 50
    return pd.DataFrame({
        'true_angle': np.random.uniform(0, 360, n),
        'pred_angle': np.random.uniform(0, 360, n),
        'error': np.random.normal(0, 5, n),
        'abs_error': np.abs(np.random.normal(0, 5, n)),
        'Hs': np.random.uniform(1, 5, n), # Hs column name match config
        'hs_bin': np.random.randint(0, 5, n)
    })

@pytest.fixture
def diag_config(tmp_path):
    return {
        'data': {'hs_column': 'Hs'},
        'outputs': {'base_results_dir': str(tmp_path)},
        'diagnostics': {
            'dpi': 50, # Low dpi for faster tests
            'save_format': 'png',
            'generate_index_plots': True,
            'generate_scatter_plots': True,
            'generate_qq_plots': False # Disable one to test flag
        }
    }

def test_directory_creation(diag_config, mock_logger, sample_preds):
    engine = DiagnosticsEngine(diag_config, mock_logger)
    engine.generate_all(sample_preds, "test_split", "run1")
    
    base = Path(diag_config['outputs']['base_results_dir'])
    diag_root = base / "08_DIAGNOSTICS"
    
    assert diag_root.exists()
    assert (diag_root / "scatter_plots").exists()
    assert (diag_root / "index_plots").exists()

def test_plot_generation(diag_config, mock_logger, sample_preds):
    engine = DiagnosticsEngine(diag_config, mock_logger)
    engine.generate_all(sample_preds, "test_split", "run1")
    
    base = Path(diag_config['outputs']['base_results_dir'])
    
    # Check if files exist
    scatter_file = base / "08_DIAGNOSTICS" / "scatter_plots" / "actual_vs_pred_test_split.png"
    assert scatter_file.exists()
    
    # Check that QQ plot was NOT generated (flag is False)
    qq_file = base / "08_DIAGNOSTICS" / "qq_plots" / "qq_plot_test_split.png"
    assert not qq_file.exists()

def test_missing_hs_column(diag_config, mock_logger, sample_preds):
    # Remove Hs column
    df_no_hs = sample_preds.drop(columns=['Hs'])
    
    engine = DiagnosticsEngine(diag_config, mock_logger)
    engine.generate_all(df_no_hs, "test_split", "run1")
    
    # Should warn but not crash
    mock_logger.warning.assert_called()