import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from modules.error_analysis_engine.error_analysis_engine import ErrorAnalysisEngine

# Concrete implementation for testing ErrorAnalysisEngine
class ConcreteErrorAnalysisEngine(ErrorAnalysisEngine):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def _get_engine_directory_name(self) -> str:
        return "09_ERROR_ANALYSIS"

@pytest.fixture
def mock_logger():
    """Provides a mock logger instance."""
    return Mock()

@pytest.fixture
def base_config(tmp_path):
    """Provides a base configuration dictionary with a temporary results directory."""
    return {
        'outputs': {
            'base_results_dir': str(tmp_path),
            'save_excel_copy': False
        },
        'error_analysis': {
            'enabled': True,
            'correlation_analysis': True, # Ensure correlation analysis is enabled
            'error_thresholds': [5, 10, 20],
            'outlier_detection': '3sigma'
        }
    }

@pytest.fixture
def error_analysis_engine(base_config, mock_logger):
    """Provides an ErrorAnalysisEngine instance."""
    return ConcreteErrorAnalysisEngine(base_config, mock_logger)

@pytest.fixture
def dummy_predictions_df():
    """Provides a dummy predictions DataFrame with required columns including row_index."""
    np.random.seed(42)
    num_samples = 100
    df = pd.DataFrame({
        'row_index': range(num_samples),
        'true_angle': np.random.uniform(0, 360, num_samples),
        'pred_angle': np.random.uniform(0, 360, num_samples),
        'abs_error': np.random.uniform(0, 25, num_samples),
        'error': np.random.uniform(-15, 15, num_samples),
        'true_sin': np.sin(np.radians(np.random.uniform(0, 360, num_samples))),
        'pred_sin': np.sin(np.radians(np.random.uniform(0, 360, num_samples))),
        'true_cos': np.cos(np.radians(np.random.uniform(0, 360, num_samples))),
        'pred_cos': np.cos(np.radians(np.random.uniform(0, 360, num_samples))),
    })
    return df # Return df with 'row_index' as a column, not the index

@pytest.fixture
def dummy_features_df():
    """Provides a dummy features DataFrame with required columns and matching index."""
    np.random.seed(43)
    num_samples = 100
    df = pd.DataFrame({
        'feature_A': np.random.uniform(10, 20, num_samples),
        'feature_B': np.random.normal(0, 1, num_samples),
        'feature_C': np.random.randint(0, 5, num_samples),
        'constant_feature': 100.0, # Will have std=0
    })
    df.index = range(num_samples) # Match predictions_df index
    return df

@pytest.fixture(autouse=True)
def mock_file_io():
    """Mocks save_dataframe."""
    with patch('modules.error_analysis_engine.error_analysis_engine.save_dataframe') as mock_save_df:
        yield mock_save_df

@pytest.fixture(autouse=True)
def mock_path_write_text():
    """Mocks Path.write_text to capture JSON output."""
    with patch('pathlib.Path.write_text') as mock_write_text:
        yield mock_write_text

# --- Test cases for prediction explanations ---

def test_generate_prediction_explanations_output(error_analysis_engine, dummy_predictions_df, dummy_features_df, mock_file_io, mock_path_write_text, tmp_path):
    """
    Tests that _generate_prediction_explanations generates prediction explanations
    parquet and summary JSON files with expected content.
    """
    mock_save_df = mock_file_io
    
    # Mock other internal methods called by execute to isolate _generate_prediction_explanations
    with patch.object(error_analysis_engine, '_analyze_thresholds'), \
         patch.object(error_analysis_engine, '_detect_outliers'), \
         patch.object(error_analysis_engine, '_analyze_correlations'), \
         patch.object(error_analysis_engine, '_analyze_bias'), \
         patch.object(error_analysis_engine, '_analyze_safety_thresholds'), \
         patch.object(error_analysis_engine, '_analyze_extremes'):

        # Call execute, which in turn calls _generate_prediction_explanations
        error_analysis_engine.execute(dummy_predictions_df, dummy_features_df, "test")

        # Construct expected output directory
        output_dir = tmp_path / "09_ERROR_ANALYSIS" / "test"
        
        # Assert save_dataframe was called for prediction_explanations.parquet
        # We need to find the specific call for this file
        
        # Check that save_dataframe was called at least once
        assert mock_save_df.called, "save_dataframe was not called."
        
        explanations_called_df = None
        for call_args in mock_save_df.call_args_list:
            if call_args[0][1] == (output_dir / "prediction_explanations.parquet"):
                explanations_called_df = call_args[0][0]
                break
        
        assert explanations_called_df is not None, "prediction_explanations.parquet was not saved."
        assert isinstance(explanations_called_df, pd.DataFrame)
        assert set(explanations_called_df.columns) == {'feature', 'corr_with_pred_angle', 'corr_with_abs_error'}
        assert 'constant_feature' not in explanations_called_df['feature'].values # Should be ignored (std=0)

        # Assert JSON summary was written
        mock_path_write_text.assert_called_once()
        json_content = json.loads(mock_path_write_text.call_args[0][0])
        assert "top_features_by_error" in json_content
        assert isinstance(json_content["top_features_by_error"], list)
        assert len(json_content["top_features_by_error"]) <= 5 # top K
        assert 'feature' in json_content["top_features_by_error"][0]


def test_generate_prediction_explanations_empty_features(error_analysis_engine, dummy_predictions_df, mock_logger):
    """
    Tests _generate_prediction_explanations is skipped when features are empty or None.
    """
    # Mock other internal methods called by execute
    with patch.object(error_analysis_engine, '_analyze_thresholds'), \
         patch.object(error_analysis_engine, '_detect_outliers'), \
         patch.object(error_analysis_engine, '_analyze_correlations'), \
         patch.object(error_analysis_engine, '_analyze_bias'), \
         patch.object(error_analysis_engine, '_analyze_safety_thresholds'), \
         patch.object(error_analysis_engine, '_analyze_extremes'), \
         patch.object(error_analysis_engine, '_generate_prediction_explanations') as mock_generate_explanations:
        
        error_analysis_engine.execute(dummy_predictions_df, pd.DataFrame(), "test") # Empty features
        
        mock_generate_explanations.assert_not_called()
        mock_logger.warning.assert_called_with("Features DataFrame missing or empty. Skipping correlation analysis.")

def test_generate_prediction_explanations_none_features(error_analysis_engine, dummy_predictions_df, mock_logger):
    """
    Tests _generate_prediction_explanations is skipped when features are None.
    """
    # Mock other internal methods called by execute
    with patch.object(error_analysis_engine, '_analyze_thresholds'), \
         patch.object(error_analysis_engine, '_detect_outliers'), \
         patch.object(error_analysis_engine, '_analyze_correlations'), \
         patch.object(error_analysis_engine, '_analyze_bias'), \
         patch.object(error_analysis_engine, '_analyze_safety_thresholds'), \
         patch.object(error_analysis_engine, '_analyze_extremes'), \
         patch.object(error_analysis_engine, '_generate_prediction_explanations') as mock_generate_explanations:
        
        error_analysis_engine.execute(dummy_predictions_df, None, "test") # None features
        
        mock_generate_explanations.assert_not_called()
        mock_logger.warning.assert_called_with("Features DataFrame missing or empty. Skipping correlation analysis.")
