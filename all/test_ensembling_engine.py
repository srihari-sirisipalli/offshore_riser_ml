import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from unittest.mock import MagicMock
from modules.ensembling_engine import EnsemblingEngine

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    """Provides a mock logger for tests."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def base_config(tmp_path):
    """Provides a base configuration for the EnsemblingEngine."""
    return {
        'outputs': {'base_results_dir': str(tmp_path)},
        'ensembling': {
            'enabled': True,
            'strategy': 'simple',
            'weighting_scheme': 'inverse_error'
        }
    }

@pytest.fixture
def sample_predictions():
    """Provides a list of two sample prediction DataFrames."""
    # Model 1: Predicts 0 degrees (sin=0, cos=1)
    df1 = pd.DataFrame({
        'true_angle': [45.0, 135.0],
        'pred_angle': [0.0, 0.0],
        'pred_sin': [0.0, 0.0],
        'pred_cos': [1.0, 1.0]
    }, index=[100, 101])
    
    # Model 2: Predicts 90 degrees (sin=1, cos=0)
    df2 = pd.DataFrame({
        'true_angle': [45.0, 135.0],
        'pred_angle': [90.0, 90.0],
        'pred_sin': [1.0, 1.0],
        'pred_cos': [0.0, 0.0]
    }, index=[100, 101])
    
    return [df1, df2]

# --- Test Cases ---

class TestEnsemblingEngine:

    def test_ensemble_disabled(self, base_config, mock_logger, sample_predictions):
        """Tests that the engine does nothing when disabled."""
        base_config['ensembling']['enabled'] = False
        engine = EnsemblingEngine(base_config, mock_logger)
        result = engine.ensemble(sample_predictions, [{}, {}], "test", "run1")
        assert result.empty
        mock_logger.info.assert_called_with("Ensembling disabled in config.")

    def test_single_model_input(self, base_config, mock_logger, sample_predictions):
        """Tests that the original DataFrame is returned if only one is provided."""
        engine = EnsemblingEngine(base_config, mock_logger)
        result = engine.ensemble([sample_predictions[0]], [{}], "test", "run1")
        pd.testing.assert_frame_equal(result, sample_predictions[0])
        mock_logger.warning.assert_called_with("Ensembling requires at least 2 prediction sets. Skipping.")

    def test_mismatched_length_error(self, base_config, mock_logger, sample_predictions):
        """Tests that a ValueError is raised for inputs with different lengths."""
        engine = EnsemblingEngine(base_config, mock_logger)
        mismatched_preds = [sample_predictions[0], sample_predictions[1].head(1)]
        with pytest.raises(ValueError, match="have different lengths"):
            engine.ensemble(mismatched_preds, [{}, {}], "test", "run1")

    def test_mismatched_index_error(self, base_config, mock_logger, sample_predictions):
        """Tests that a ValueError is raised for inputs with different indices."""
        engine = EnsemblingEngine(base_config, mock_logger)
        df2_bad_index = sample_predictions[1].copy()
        df2_bad_index.index = [200, 201]
        mismatched_preds = [sample_predictions[0], df2_bad_index]
        with pytest.raises(ValueError, match="indices do not match"):
            engine.ensemble(mismatched_preds, [{}, {}], "test", "run1")

    def test_simple_average_strategy(self, base_config, mock_logger, sample_predictions):
        """Tests the simple average strategy."""
        # For a true angle of 45, avg of 0 and 90 should be 45
        engine = EnsemblingEngine(base_config, mock_logger)
        result = engine.ensemble(sample_predictions, [{}, {}], "test", "run1")
        assert np.isclose(result['pred_angle'].iloc[0], 45.0)

    def test_weighted_average_strategy(self, base_config, mock_logger, sample_predictions):
        """Tests the weighted average strategy where one model is much better."""
        base_config['ensembling']['strategy'] = 'weighted'
        engine = EnsemblingEngine(base_config, mock_logger)
        
        # Model 1 has high error, Model 2 has low error
        metrics = [{'cmae': 45.0}, {'cmae': 5.0}]
        
        result = engine.ensemble(sample_predictions, metrics, "test", "run1")
        
        # The prediction should be heavily skewed towards Model 2 (90 degrees)
        assert result['pred_angle'].iloc[0] > 70.0 # Much closer to 90 than 45

    def test_weighted_average_with_perfect_model(self, base_config, mock_logger, sample_predictions):
        """Tests that a model with CMAE=0 gets all the weight."""
        base_config['ensembling']['strategy'] = 'weighted'
        engine = EnsemblingEngine(base_config, mock_logger)
        
        # Model 1 has error, Model 2 is perfect
        metrics = [{'cmae': 45.0}, {'cmae': 0.0}]
        
        result = engine.ensemble(sample_predictions, metrics, "test", "run1")
        
        # The prediction should be exactly Model 2's prediction (90 degrees)
        assert np.isclose(result['pred_angle'].iloc[0], 90.0)

    def test_unknown_strategy_fallback(self, base_config, mock_logger, sample_predictions):
        """Tests that an unknown strategy falls back to simple average."""
        base_config['ensembling']['strategy'] = 'unknown_strategy'
        engine = EnsemblingEngine(base_config, mock_logger)
        
        result = engine.ensemble(sample_predictions, [{}, {}], "test", "run1")
        
        mock_logger.warning.assert_called_with("Unknown strategy 'unknown_strategy', defaulting to simple average.")
        # Result should be the same as simple average
        assert np.isclose(result['pred_angle'].iloc[0], 45.0)

    def test_hs_sensitive_fallback(self, base_config, mock_logger, sample_predictions):
        """Tests that the 'hs_sensitive' strategy falls back as it's not implemented."""
        base_config['ensembling']['strategy'] = 'hs_sensitive'
        engine = EnsemblingEngine(base_config, mock_logger)
        
        result = engine.ensemble(sample_predictions, [{}, {}], "test", "run1")
        
        mock_logger.warning.assert_called_with("Hs-Sensitive ensembling requires per-bin metrics. Falling back to Weighted Average.")
        # Check that it falls back to weighted average (which with no metrics is a simple average)
        assert np.isclose(result['pred_angle'].iloc[0], 45.0)
        
    def test_ensemble_creates_file(self, base_config, mock_logger, sample_predictions, tmp_path):
        """Tests that the ensembling process creates an output file."""
        engine = EnsemblingEngine(base_config, mock_logger)
        engine.ensemble(sample_predictions, [{}, {}], "test", "run1")
        
        output_path = Path(tmp_path) / "09_ADVANCED_ANALYTICS" / "ensembling" / "ensemble_predictions_test.xlsx"
        assert output_path.exists()
