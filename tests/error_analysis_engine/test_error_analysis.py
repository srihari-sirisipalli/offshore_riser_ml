import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from unittest.mock import MagicMock
from modules.error_analysis_engine import ErrorAnalysisEngine

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def sample_data():
    # Predictions
    preds = pd.DataFrame({
        'row_index': [0, 1, 2, 3, 4],
        'true_angle': [10, 100, 200, 300, 50],
        'abs_error': [1, 2, 15, 25, 0.5], # 15>10, 25>20
        'error': [1, -2, 15, -25, 0.5]
    })
    
    # Features (Indices must match preds 'row_index' logic)
    # Here we map index 0..4
    features = pd.DataFrame({
        'feature_A': [10, 20, 30, 40, 50], # Perfectly correlated with index/error roughly
        'feature_B': [5, 5, 5, 5, 5]       # No correlation
    }, index=[0, 1, 2, 3, 4])
    
    return preds, features

@pytest.fixture
def ea_config(tmp_path):
    return {
        'outputs': {'base_results_dir': str(tmp_path)},
        'error_analysis': {
            'enabled': True,
            'error_thresholds': [10, 20],
            'outlier_detection': '3sigma',
            'correlation_analysis': True
        }
    }

def test_threshold_filtering(ea_config, mock_logger, sample_data):
    preds, feats = sample_data
    engine = ErrorAnalysisEngine(ea_config, mock_logger)
    
    engine.analyze(preds, feats, "test_split", "run1")
    
    base = Path(ea_config['outputs']['base_results_dir'])
    output_dir = base / "09_ERROR_ANALYSIS" / "test_split"
    
    # Check >10 deg file
    file_10 = output_dir / "samples_error_gt_10deg.xlsx"
    assert file_10.exists()
    df_10 = pd.read_excel(file_10)
    assert len(df_10) == 2 # 15 and 25
    
    # Check >20 deg file
    file_20 = output_dir / "samples_error_gt_20deg.xlsx"
    assert file_20.exists()
    df_20 = pd.read_excel(file_20)
    assert len(df_20) == 1 # 25

def test_correlation_analysis(ea_config, mock_logger, sample_data):
    preds, feats = sample_data
    # Modify data to have perfect correlation
    # Error: 1, 2, 15, 25, 0.5
    # Feat A: 1, 2, 15, 25, 0.5 (Correlation should be 1.0)
    feats['feature_A'] = preds['abs_error'].values
    
    engine = ErrorAnalysisEngine(ea_config, mock_logger)
    engine.analyze(preds, feats, "test_split", "run1")
    
    base = Path(ea_config['outputs']['base_results_dir'])
    corr_file = base / "09_ERROR_ANALYSIS" / "test_split" / "error_feature_correlations.xlsx"
    
    assert corr_file.exists()
    df_corr = pd.read_excel(corr_file)
    
    # Check if Feature A is top correlated
    row_a = df_corr[df_corr['Feature'] == 'feature_A']
    assert not row_a.empty
    assert row_a.iloc[0]['Correlation_with_AbsError'] > 0.99

def test_outlier_detection(ea_config, mock_logger):
    # Create data with one massive outlier
    # FIX: Added 'error' and 'true_angle' columns to prevent KeyError in _analyze_bias
    preds = pd.DataFrame({
        'row_index': range(100),
        'true_angle': [0] * 100, # Dummy values for bias analysis
        'abs_error': np.concatenate([np.random.normal(0, 1, 99), [100]]), # 100 is outlier
        'error': np.concatenate([np.random.normal(0, 1, 99), [100]])      # Signed error
    })
    feats = pd.DataFrame({'f1': range(100)})
    
    engine = ErrorAnalysisEngine(ea_config, mock_logger)
    engine.analyze(preds, feats, "test_split", "run1")
    
    base = Path(ea_config['outputs']['base_results_dir'])
    outlier_file = base / "09_ERROR_ANALYSIS" / "test_split" / "statistical_outliers_3sigma.xlsx"
    
    assert outlier_file.exists()
    df_out = pd.read_excel(outlier_file)
    assert len(df_out) >= 1
    assert df_out.iloc[0]['abs_error'] == 100