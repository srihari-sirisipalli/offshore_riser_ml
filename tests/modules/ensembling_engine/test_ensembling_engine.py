import pytest
from unittest.mock import Mock, patch, call
import pandas as pd
import numpy as np
from pathlib import Path
from modules.ensembling_engine.ensembling_engine import EnsemblingEngine

# Concrete implementation for testing EnsemblingEngine
class ConcreteEnsemblingEngine(EnsemblingEngine):
    def __init__(self, config, logger, output_dir_name="09_ADVANCED_ANALYTICS", standard_output_dir_name="98_ENSEMBLING"):
        # Temporarily store names to be used by _get_engine_directory_name
        self._test_output_dir_name = output_dir_name
        self._test_standard_output_dir_name = standard_output_dir_name
        super().__init__(config, logger)
        # Override BaseEngine's paths for specific testing if needed
        self.output_dir = Path(config['outputs']['base_results_dir']) / output_dir_name
        self.standard_output_dir = Path(config['outputs']['base_results_dir']) / standard_output_dir_name

    def _get_engine_directory_name(self) -> str:
        return self._test_output_dir_name

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
        'ensembling': {
            'enabled': True,
            'strategy': 'weighted', # Default for most tests
            'weighting_scheme': 'inverse_error'
        },
        'data': { # Minimal data config
            'hs_column': 'Hs'
        }
    }

@pytest.fixture
def ensembling_engine(base_config, mock_logger):
    """Provides an EnsemblingEngine instance."""
    return ConcreteEnsemblingEngine(base_config, mock_logger)

@pytest.fixture
def mock_circular_metrics(dummy_predictions_list): # Add dummy_predictions_list to get num_samples
    """Mocks reconstruct_angle and wrap_angle."""
    num_samples = len(dummy_predictions_list[0]) # Get num_samples from dummy data
    with patch('modules.ensembling_engine.ensembling_engine.reconstruct_angle', return_value=np.array([10.0] * num_samples)) as mock_reconstruct, \
         patch('modules.ensembling_engine.ensembling_engine.wrap_angle', return_value=np.array([1.0] * num_samples)) as mock_wrap_angle:
        yield mock_reconstruct, mock_wrap_angle

@pytest.fixture
def dummy_predictions_list():
    """Provides a list of dummy predictions DataFrames."""
    np.random.seed(42)
    num_samples = 10
    
    # Model 1: Good performance
    df1 = pd.DataFrame({
        'true_angle': np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        'pred_sin': np.sin(np.radians([11, 21, 31, 41, 51, 61, 71, 81, 91, 101])),
        'pred_cos': np.cos(np.radians([11, 21, 31, 41, 51, 61, 71, 81, 91, 101])),
        'Hs': np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
        'Hs': np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]), # For Hs-sensitive
    })
    
    # Model 2: Average performance
    df2 = pd.DataFrame({
        'true_angle': np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        'pred_sin': np.sin(np.radians([13, 23, 33, 43, 53, 63, 73, 83, 93, 103])),
        'pred_cos': np.cos(np.radians([13, 23, 33, 43, 53, 63, 73, 83, 93, 103])),
        'Hs': np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
        'Hs_ft': np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
    })

    # Model 3: Bad performance
    df3 = pd.DataFrame({
        'true_angle': np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        'pred_sin': np.sin(np.radians([20, 30, 40, 50, 60, 70, 80, 90, 100, 110])),
        'pred_cos': np.cos(np.radians([20, 30, 40, 50, 60, 70, 80, 90, 100, 110])),
        'Hs': np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
        'Hs_ft': np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
    })

    return [df1, df2, df3]

@pytest.fixture
def dummy_metrics_list():
    """Provides a list of dummy metrics dictionaries."""
    return [
        {'cmae': 5.0, 'crmse': 7.0}, # Model 1 (better)
        {'cmae': 8.0, 'crmse': 10.0}, # Model 2 (average)
        {'cmae': 15.0, 'crmse': 20.0}, # Model 3 (worse)
    ]

@pytest.fixture(autouse=True)
def mock_save_dataframe():
    """Mocks save_dataframe from utils.file_io."""
    with patch('modules.ensembling_engine.ensembling_engine.save_dataframe') as mock:
        yield mock

# --- Test cases for ensembling ---

def test_ensembling_disabled(ensembling_engine, dummy_predictions_list, dummy_metrics_list, mock_logger, mock_save_dataframe):
    """Tests that ensembling is skipped if disabled in config."""
    ensembling_engine.enabled = False # Directly set the attribute for test
    
    result = ensembling_engine.execute(dummy_predictions_list, dummy_metrics_list, "val", "run_001")
    
    mock_logger.info.assert_called_with("Ensembling disabled in config.")
    mock_save_dataframe.assert_not_called()
    assert result.empty

def test_ensembling_less_than_two_models(ensembling_engine, dummy_predictions_list, dummy_metrics_list, mock_logger, mock_save_dataframe):
    """Tests that ensembling warns and returns first model if less than 2 predictions sets."""
    ensembling_engine.config['ensembling']['enabled'] = True
    
    # Test with 0 models
    result_0 = ensembling_engine.execute([], [], "val", "run_001")
    mock_logger.warning.assert_called_with("Ensembling requires at least 2 prediction sets. Skipping.")
    assert result_0.empty
    mock_logger.reset_mock() # Reset mock to check for next call
    
    # Test with 1 model
    result_1 = ensembling_engine.execute(dummy_predictions_list[:1], dummy_metrics_list[:1], "val", "run_001")
    mock_logger.warning.assert_called_with("Ensembling requires at least 2 prediction sets. Skipping.")
    assert result_1 is dummy_predictions_list[0]
    mock_save_dataframe.assert_not_called()

def test_simple_average_strategy(ensembling_engine, dummy_predictions_list, dummy_metrics_list, mock_save_dataframe, mock_circular_metrics, tmp_path):
    """Tests the simple average ensembling strategy."""
    mock_reconstruct, mock_wrap_angle = mock_circular_metrics
    ensembling_engine.config['ensembling']['strategy'] = 'simple'
    
    ensemble_df = ensembling_engine.execute(dummy_predictions_list, dummy_metrics_list, "test", "run_001")
    
    # Assert save_dataframe was called (for legacy and standard)
    assert mock_save_dataframe.call_count == 2
    
    # Check ensemble_df content
    expected_pred_sin = np.mean([df['pred_sin'] for df in dummy_predictions_list], axis=0)
    expected_pred_cos = np.mean([df['pred_cos'] for df in dummy_predictions_list], axis=0)
    
    pd.testing.assert_series_equal(ensemble_df['pred_sin'], pd.Series(expected_pred_sin), check_dtype=False, check_names=False)
    pd.testing.assert_series_equal(ensemble_df['pred_cos'], pd.Series(expected_pred_cos), check_dtype=False, check_names=False)
    
    args, kwargs = mock_reconstruct.call_args
    np.testing.assert_array_almost_equal(args[0], expected_pred_sin)
    np.testing.assert_array_almost_equal(args[1], expected_pred_cos)
    assert 'pred_angle' in ensemble_df.columns
    assert 'error' in ensemble_df.columns
    assert 'abs_error' in ensemble_df.columns
    assert 'true_angle' in ensemble_df.columns

    # Check that Hs column is transferred
    assert 'Hs' in ensemble_df.columns

    # Check save paths
    output_dir = tmp_path / "09_ADVANCED_ANALYTICS" / "ensembling"
    # The standard output also creates an "ensembling" subdirectory
    standard_output_dir = tmp_path / "98_ENSEMBLING" / "ensembling"
    mock_save_dataframe.assert_any_call(ensemble_df, output_dir / "ensemble_predictions_test.parquet", excel_copy=False, index=False)
    mock_save_dataframe.assert_any_call(ensemble_df, standard_output_dir / "ensemble_predictions_test.parquet", excel_copy=False, index=False)

def test_weighted_average_strategy_inverse_error(ensembling_engine, dummy_predictions_list, dummy_metrics_list, mock_save_dataframe, mock_circular_metrics, tmp_path):
    """Tests the weighted average ensembling strategy with inverse error weighting."""
    mock_reconstruct, mock_wrap_angle = mock_circular_metrics
    ensembling_engine.config['ensembling']['strategy'] = 'weighted'
    ensembling_engine.config['ensembling']['weighting_scheme'] = 'inverse_error'

    # Expected weights based on cmaes: 1/5, 1/8, 1/15, then normalized
    cmaes = np.array([5.0, 8.0, 15.0])
    weights = 1.0 / (cmaes + 0.001)
    weights = weights / np.sum(weights) # Normalized

    ensemble_df = ensembling_engine.execute(dummy_predictions_list, dummy_metrics_list, "val", "run_002")

    # Assert save_dataframe was called
    assert mock_save_dataframe.call_count == 2

    # Check ensemble_df content (pred_sin/cos are np.sum(sin_stack * weights[:, None], axis=0))
    sin_stack = np.stack([df['pred_sin'].values for df in dummy_predictions_list])
    cos_stack = np.stack([df['pred_cos'].values for df in dummy_predictions_list])

    expected_pred_sin = np.sum(sin_stack * weights[:, None], axis=0)
    expected_pred_cos = np.sum(cos_stack * weights[:, None], axis=0)

    pd.testing.assert_series_equal(ensemble_df['pred_sin'], pd.Series(expected_pred_sin), check_dtype=False, check_names=False)       
    pd.testing.assert_series_equal(ensemble_df['pred_cos'], pd.Series(expected_pred_cos), check_dtype=False, check_names=False)       
    args, kwargs = mock_reconstruct.call_args
    np.testing.assert_array_almost_equal(args[0], expected_pred_sin)
    np.testing.assert_array_almost_equal(args[1], expected_pred_cos)

def test_weighted_average_strategy_perfect_model(ensembling_engine, dummy_predictions_list, mock_save_dataframe, mock_circular_metrics, tmp_path):
    """Tests weighted average when a model has CMAE = 0."""
    mock_reconstruct, mock_wrap_angle = mock_circular_metrics
    ensembling_engine.config['ensembling']['strategy'] = 'weighted'
    ensembling_engine.config['ensembling']['weighting_scheme'] = 'inverse_error'

    metrics_with_perfect = [
        {'cmae': 0.0, 'crmse': 1.0}, # Model 1 (perfect)
        {'cmae': 8.0, 'crmse': 10.0}, # Model 2
        {'cmae': 0.0, 'crmse': 0.5}, # Model 3 (perfect)
    ]

    ensemble_df = ensembling_engine.execute(dummy_predictions_list, metrics_with_perfect, "val", "run_003")

    # Only perfect models get weights
    expected_weights = np.array([0.5, 0.0, 0.5]) # Normalized between the two perfect models
    
    sin_stack = np.stack([df['pred_sin'].values for df in dummy_predictions_list])
    cos_stack = np.stack([df['pred_cos'].values for df in dummy_predictions_list])

    expected_pred_sin = np.sum(sin_stack * expected_weights[:, None], axis=0)
    expected_pred_cos = np.sum(cos_stack * expected_weights[:, None], axis=0)
    
    pd.testing.assert_series_equal(ensemble_df['pred_sin'], pd.Series(expected_pred_sin), check_dtype=False, check_names=False)
    args, kwargs = mock_reconstruct.call_args
    np.testing.assert_array_almost_equal(args[0], expected_pred_sin)
    np.testing.assert_array_almost_equal(args[1], expected_pred_cos)

def test_hs_sensitive_ensemble_fallback(ensembling_engine, dummy_predictions_list, dummy_metrics_list, mock_logger, mock_save_dataframe, mock_circular_metrics):
    """Tests that Hs-sensitive ensemble falls back to weighted average and logs a warning."""
    mock_reconstruct, mock_wrap_angle = mock_circular_metrics
    ensembling_engine.config['ensembling']['strategy'] = 'hs_sensitive'
    
    # Execute should call _hs_sensitive_ensemble, which then calls _weighted_average
    with patch.object(ensembling_engine, '_weighted_average') as mock_weighted_average:
        # Set a valid return value for the mocked _weighted_average
        mock_weighted_average.return_value = pd.DataFrame({
            'pred_sin': np.array([0.5]*10),
            'pred_cos': np.array([0.5]*10),
            'pred_angle': np.array([45.0]*10), # Dummy pred_angle
            'true_angle': dummy_predictions_list[0]['true_angle'].values # Required for error calculation
        })
        ensemble_df = ensembling_engine.execute(dummy_predictions_list, dummy_metrics_list, "test", "run_004")
        
        mock_logger.warning.assert_called_with("Hs-Sensitive ensembling requires per-bin metrics. Falling back to Weighted Average.")
        mock_weighted_average.assert_called_once_with(dummy_predictions_list, dummy_metrics_list)
        
        # Check that the returned df is from the mocked weighted average
        assert ensemble_df is mock_weighted_average.return_value

def test_validate_alignment_different_lengths(ensembling_engine, dummy_predictions_list):
    """Tests _validate_alignment raises ValueError for different DataFrame lengths."""
    misaligned_list = dummy_predictions_list + [dummy_predictions_list[0].iloc[:5]] # Add a shorter df
    with pytest.raises(ValueError, match="DataFrames have different lengths."):
        ensembling_engine._validate_alignment(misaligned_list)

def test_validate_alignment_different_indices(ensembling_engine, dummy_predictions_list):
    """Tests _validate_alignment raises ValueError for different DataFrame indices."""
    df_diff_idx = dummy_predictions_list[0].copy()
    df_diff_idx.index = df_diff_idx.index + 100 # Shift index
    misaligned_list = [dummy_predictions_list[0], df_diff_idx]
    with pytest.raises(ValueError, match="indices do not match"):
        ensembling_engine._validate_alignment(misaligned_list)
