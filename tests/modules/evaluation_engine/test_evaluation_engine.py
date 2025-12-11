import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from pathlib import Path
from modules.evaluation_engine.evaluation_engine import EvaluationEngine
from modules.base.base_engine import BaseEngine # For inheritance

# Concrete implementation for testing EvaluationEngine
class ConcreteEvaluationEngine(EvaluationEngine):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def _get_engine_directory_name(self) -> str:
        return "08_EVALUATION"

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
        'evaluation': {
            'bootstrap_samples': 0 # Disable bootstrap for these tests
        }
    }

@pytest.fixture
def evaluation_engine(base_config, mock_logger):
    """Provides an EvaluationEngine instance."""
    return ConcreteEvaluationEngine(base_config, mock_logger)

@pytest.fixture
def dummy_predictions_df():
    """Provides a dummy predictions DataFrame."""
    np.random.seed(42)
    num_samples = 100
    return pd.DataFrame({
        'true_angle': np.random.uniform(0, 360, num_samples),
        'pred_angle': np.random.uniform(0, 360, num_samples),
        'abs_error': np.random.uniform(0, 20, num_samples),
        'error': np.random.uniform(-10, 10, num_samples),
        'true_sin': np.sin(np.radians(np.random.uniform(0, 360, num_samples))),
        'pred_sin': np.sin(np.radians(np.random.uniform(0, 360, num_samples))),
        'true_cos': np.cos(np.radians(np.random.uniform(0, 360, num_samples))),
        'pred_cos': np.cos(np.radians(np.random.uniform(0, 360, num_samples))),
    })

@pytest.fixture
def dummy_model_metrics():
    """Provides dummy model metrics."""
    return {"cmae": 5.0, "crmse": 7.0, "max_error": 15.0}

@pytest.fixture(autouse=True)
def mock_file_io():
    """Mocks save_dataframe and read_dataframe."""
    with patch('modules.evaluation_engine.evaluation_engine.save_dataframe') as mock_save_df, \
         patch('modules.evaluation_engine.evaluation_engine.read_dataframe') as mock_read_df:
        yield mock_save_df, mock_read_df

# --- Test cases for _compute_industry_baseline ---

def test_compute_industry_baseline_provided_metrics(evaluation_engine, dummy_predictions_df, dummy_model_metrics, mock_file_io, tmp_path):
    """
    Tests _compute_industry_baseline when metrics are provided directly in config.
    """
    mock_save_df, _ = mock_file_io
    
    # Configure with provided metrics
    evaluation_engine.config['evaluation']['industry_baseline'] = {
        "metrics": {"cmae": 6.0, "max_error": 18.0},
        "label": "CustomBaseline"
    }
    
    evaluation_engine._compute_industry_baseline(dummy_predictions_df, "test", dummy_model_metrics)
    
    # Assert save_dataframe was called
    mock_save_df.assert_called_once()
    
    # Verify the DataFrame content passed to save_dataframe
    called_df = mock_save_df.call_args[0][0]
    assert isinstance(called_df, pd.DataFrame)
    assert len(called_df) == 2
    
    # Check "CustomBaseline" row for cmae
    cmae_row = called_df[called_df['metric'] == 'cmae'].iloc[0]
    assert cmae_row['model'] == 'CustomBaseline'
    assert cmae_row['value'] == 6.0
    # delta_vs_model: baseline_value - model_value => 6.0 - 5.0 = 1.0
    assert cmae_row['delta_vs_model'] == pytest.approx(1.0) 
    assert cmae_row['source'] == 'provided_metrics'

    # Check output path
    expected_path = tmp_path / "08_EVALUATION" / "industry_baseline_comparison_test.parquet"
    assert mock_save_df.call_args[0][1] == expected_path


def test_compute_industry_baseline_predictions_path(evaluation_engine, dummy_predictions_df, dummy_model_metrics, mock_file_io, tmp_path):
    """
    Tests _compute_industry_baseline when a predictions_path is provided in config.
    """
    mock_save_df, mock_read_df = mock_file_io
    
    # Configure with predictions_path
    baseline_predictions_path = tmp_path / "baseline_preds.parquet"
    evaluation_engine.config['evaluation']['industry_baseline'] = {
        "predictions_path": str(baseline_predictions_path),
        "label": "PathBaseline"
    }
    
    # Mock read_dataframe to return a dummy baseline DataFrame
    mock_baseline_df = pd.DataFrame({
        'true_angle': np.array([10, 20]),
        'pred_angle': np.array([15, 25]),
        'abs_error': np.array([5, 5]),
        'error': np.array([5, 5]),
        'true_sin': np.sin(np.radians([10, 20])),
        'pred_sin': np.sin(np.radians([15, 25])),
        'true_cos': np.cos(np.radians([10, 20])),
        'pred_cos': np.cos(np.radians([15, 25])),
    })
    mock_read_df.return_value = mock_baseline_df
    
    # Mock _compute_metrics_flexible to return specific metrics for the baseline
    with patch.object(evaluation_engine, '_compute_metrics_flexible', return_value={"cmae": 4.0, "max_error": 12.0}) as mock_compute_metrics_flexible:
        evaluation_engine._compute_industry_baseline(dummy_predictions_df, "val", dummy_model_metrics)
    
        # Assert read_dataframe and _compute_metrics_flexible were called
        mock_read_df.assert_called_once_with(baseline_predictions_path)
        mock_compute_metrics_flexible.assert_called_once_with(mock_baseline_df)
        
        # Assert save_dataframe was called
        mock_save_df.assert_called_once()
        
        # Verify the DataFrame content passed to save_dataframe
        called_df = mock_save_df.call_args[0][0]
        assert isinstance(called_df, pd.DataFrame)
        assert len(called_df) == 2 # cmae and max_error
        
        cmae_row = called_df[called_df['metric'] == 'cmae'].iloc[0]
        assert cmae_row['model'] == 'PathBaseline'
        assert cmae_row['value'] == 4.0
        # delta_vs_model: baseline_value - model_value => 4.0 - 5.0 = -1.0
        assert cmae_row['delta_vs_model'] == pytest.approx(-1.0)
        assert cmae_row['source'] == 'predictions_path'

        # Check output path
        expected_path = tmp_path / "08_EVALUATION" / "industry_baseline_comparison_val.parquet"
        assert mock_save_df.call_args[0][1] == expected_path

def test_compute_industry_baseline_fallback_naive(evaluation_engine, dummy_predictions_df, dummy_model_metrics, mock_file_io, tmp_path):
    """
    Tests _compute_industry_baseline when falling back to naive baselines.
    """
    mock_save_df, _ = mock_file_io
    
    # Ensure no provided_metrics or predictions_path in config
    evaluation_engine.config['evaluation']['industry_baseline'] = {"label": "NaiveFallback"}
    
    # Mock _compute_naive_baselines to return specific metrics
    mock_naive_df = pd.DataFrame([
        {"predictor": "circular_mean", "pred_angle": 180, "cmae": 8.0, "median_abs_error": 7.0, "max_error": 25.0},
        {"predictor": "circular_median", "pred_angle": 190, "cmae": 7.5, "median_abs_error": 6.5, "max_error": 22.0},
    ])
    with patch.object(evaluation_engine, '_compute_naive_baselines', return_value=mock_naive_df) as mock_compute_naive_baselines:
        evaluation_engine._compute_industry_baseline(dummy_predictions_df, "test", dummy_model_metrics)
    
        # Assert _compute_naive_baselines was called
        mock_compute_naive_baselines.assert_called_once_with(dummy_predictions_df)
        
        # Assert save_dataframe was called
        mock_save_df.assert_called_once()
        
        # Verify the DataFrame content passed to save_dataframe
        called_df = mock_save_df.call_args[0][0]
        assert isinstance(called_df, pd.DataFrame)
        assert len(called_df) == 2 # two naive predictors
        
        cmae_row = called_df[called_df['metric'] == 'cmae'].iloc[0]
        assert cmae_row['model'] == 'circular_mean'
        assert cmae_row['value'] == 8.0
        # delta_vs_model: baseline_value - model_value => 8.0 - 5.0 = 3.0
        assert cmae_row['delta_vs_model'] == pytest.approx(3.0)
        assert cmae_row['source'] == 'naive_baseline'

        # Check output path
        expected_path = tmp_path / "08_EVALUATION" / "industry_baseline_comparison_test.parquet"
        assert mock_save_df.call_args[0][1] == expected_path

def test_compute_industry_baseline_no_config_skips(evaluation_engine, dummy_predictions_df, dummy_model_metrics, mock_file_io, mock_logger):
    """
    Tests _compute_industry_baseline when no config is provided for industry_baseline.
    Should skip and log info message.
    """
    mock_save_df, _ = mock_file_io
    
    # Ensure no industry_baseline config
    if 'industry_baseline' in evaluation_engine.config['evaluation']:
        del evaluation_engine.config['evaluation']['industry_baseline']
    
    evaluation_engine._compute_industry_baseline(dummy_predictions_df, "train", dummy_model_metrics)
    
    # Assert save_dataframe was NOT called
    mock_save_df.assert_not_called()
    
    # Assert info message was logged
    mock_logger.info.assert_called_with("Industry baseline comparison skipped (no config provided in 'evaluation.industry_baseline').")

def test_compute_industry_baseline_predictions_path_error(evaluation_engine, dummy_predictions_df, dummy_model_metrics, mock_file_io, mock_logger, tmp_path):
    """
    Tests _compute_industry_baseline gracefully handles errors during predictions_path loading.
    """
    mock_save_df, mock_read_df = mock_file_io
    
    baseline_predictions_path = tmp_path / "non_existent.parquet"
    evaluation_engine.config['evaluation']['industry_baseline'] = {
        "predictions_path": str(baseline_predictions_path),
        "label": "ErrorBaseline"
    }
    
    # Mock read_dataframe to raise an exception
    mock_read_df.side_effect = Exception("File not found simulation")
    
    evaluation_engine._compute_industry_baseline(dummy_predictions_df, "test", dummy_model_metrics)
    
    # Assert read_dataframe was called
    mock_read_df.assert_called_once_with(baseline_predictions_path)
    
    # Assert save_dataframe was NOT called
    mock_save_df.assert_not_called()
    
    # Assert warning was logged
    mock_logger.warning.assert_called_with(
        f"Industry baseline metrics could not be computed: File not found simulation"
    )
