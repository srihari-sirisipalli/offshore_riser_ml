import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from unittest.mock import MagicMock
from modules.evaluation_engine import EvaluationEngine

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    """Provides a mock logger for tests."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def base_config(tmp_path):
    """Provides a base configuration for the EvaluationEngine."""
    return {
        "outputs": {
            "base_results_dir": str(tmp_path)
        }
    }

@pytest.fixture
def sample_predictions_df():
    """Provides a sample predictions DataFrame for testing."""
    n = 20
    true_angle = np.linspace(0, 350, n)
    pred_angle = true_angle + np.random.normal(0, 5, n)
    pred_angle = pred_angle % 360
    
    error = (true_angle - pred_angle + 180) % 360 - 180
    abs_error = np.abs(error)
    
    return pd.DataFrame({
        'true_angle': true_angle,
        'pred_angle': pred_angle,
        'error': error,
        'abs_error': abs_error,
        'true_sin': np.sin(np.radians(true_angle)),
        'pred_sin': np.sin(np.radians(pred_angle)),
        'true_cos': np.cos(np.radians(true_angle)),
        'pred_cos': np.cos(np.radians(pred_angle)),
    })

# --- Test Cases ---

class TestEvaluationEngine:

    def test_compute_metrics_success(self, base_config, mock_logger, sample_predictions_df):
        """Tests that all metrics are computed correctly."""
        engine = EvaluationEngine(base_config, mock_logger)
        metrics = engine.compute_metrics(sample_predictions_df)
        
        assert isinstance(metrics, dict)
        assert 'cmae' in metrics
        assert 'crmse' in metrics
        assert 'max_error' in metrics
        assert 'accuracy_at_5deg' in metrics
        assert 'percentile_95' in metrics
        assert 'mae_sin' in metrics
        assert metrics['n_samples'] == len(sample_predictions_df)
        
        # Check a specific value
        assert metrics['accuracy_at_0deg'] == (np.sum(sample_predictions_df['abs_error'] == 0) / len(sample_predictions_df)) * 100

    def test_compute_metrics_empty_df(self, base_config, mock_logger):
        """Tests that an empty dictionary is returned for an empty DataFrame."""
        engine = EvaluationEngine(base_config, mock_logger)
        metrics = engine.compute_metrics(pd.DataFrame())
        
        assert metrics == {}
        mock_logger.warning.assert_called_with("Empty predictions dataframe provided for evaluation.")

    def test_identify_extremes(self, base_config, mock_logger):
        """Tests the identification of best and worst performing samples."""
        df = pd.DataFrame({'abs_error': [10, 1, 30, 5, 2]})
        engine = EvaluationEngine(base_config, mock_logger)
        
        best, worst = engine._identify_extremes(df, n=2)
        
        assert best['abs_error'].tolist() == [1, 2]
        assert worst['abs_error'].tolist() == [30, 10]

    def test_identify_extremes_empty_df(self, base_config, mock_logger):
        """Tests that empty dataframes are returned when identifying extremes on an empty dataframe."""
        engine = EvaluationEngine(base_config, mock_logger)
        best, worst = engine._identify_extremes(pd.DataFrame(), n=2)
        assert best.empty
        assert worst.empty

    def test_evaluate_orchestration_and_artifacts(self, base_config, mock_logger, sample_predictions_df, tmp_path):
        """Tests the full evaluate method orchestration and artifact creation."""
        engine = EvaluationEngine(base_config, mock_logger)
        
        metrics = engine.evaluate(sample_predictions_df, "test", "run1")
        
        assert isinstance(metrics, dict)
        assert 'cmae' in metrics
        
        # Check for created files
        output_dir = Path(tmp_path) / "08_EVALUATION"
        assert (output_dir / "metrics_test.xlsx").exists()
        assert (output_dir / "best_10_samples_test.xlsx").exists()
        assert (output_dir / "worst_10_samples_test.xlsx").exists()
        
        # Verify content of one of the files
        metrics_df = pd.read_excel(output_dir / "metrics_test.xlsx")
        assert 'cmae' in metrics_df.columns
        assert np.isclose(metrics_df.iloc[0]['cmae'], metrics['cmae'])
