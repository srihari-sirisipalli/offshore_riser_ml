import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from unittest.mock import MagicMock
from modules.bootstrapping_engine import BootstrappingEngine

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    """Provides a mock logger for tests."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def base_config(tmp_path):
    """Provides a base configuration for the BootstrappingEngine."""
    return {
        'outputs': {'base_results_dir': str(tmp_path)},
        'bootstrapping': {
            'enabled': True,
            'num_samples': 50, # Small number for fast testing
            'confidence_level': 0.95,
            'sample_ratio': 1.0
        },
        'splitting': {'seed': 42}, # For fallback seed
        '_internal_seeds': {'bootstrap': 123} # For reproducible test
    }

@pytest.fixture
def sample_predictions_df():
    """Provides a sample predictions DataFrame."""
    n = 100
    return pd.DataFrame({
        'true_angle': np.linspace(0, 350, n),
        'pred_angle': (np.linspace(0, 350, n) + 5) % 360,
        'abs_error': np.full(n, 5.0),
        'error': np.full(n, 5.0)
    })

# --- Test Cases ---

class TestBootstrappingEngine:

    def test_bootstrap_execution_and_artifacts(self, base_config, mock_logger, sample_predictions_df, tmp_path):
        """Tests a full successful run, checking outputs and created files."""
        engine = BootstrappingEngine(base_config, mock_logger)
        ci_results = engine.bootstrap(sample_predictions_df, "test", "run1")
        
        # Check return structure
        assert 'cmae' in ci_results
        assert 'lower' in ci_results['cmae'] and 'upper' in ci_results['cmae']
        
        # Check logic: for constant error of 5, all stats should be 5
        assert np.isclose(ci_results['cmae']['mean'], 5.0)
        assert np.isclose(ci_results['cmae']['lower'], 5.0)
        assert np.isclose(ci_results['cmae']['upper'], 5.0)
        
        # Check that artifacts are created
        output_dir = Path(tmp_path) / "09_ADVANCED_ANALYTICS" / "bootstrapping" / "test"
        assert (output_dir / "bootstrap_ci.xlsx").exists()
        assert (output_dir / "bootstrap_samples.xlsx").exists()
        assert (output_dir / "bootstrap_dist_cmae.png").exists()

    def test_bootstrap_disabled(self, base_config, mock_logger, sample_predictions_df):
        """Tests that the engine does nothing when disabled."""
        base_config['bootstrapping']['enabled'] = False
        engine = BootstrappingEngine(base_config, mock_logger)
        result = engine.bootstrap(sample_predictions_df, "test", "run1")
        assert result == {}
        mock_logger.info.assert_called_with("Bootstrapping disabled in config.")

    def test_invalid_confidence_level_warning(self, base_config, mock_logger, sample_predictions_df):
        """Tests that an invalid confidence level is corrected and logs a warning."""
        base_config['bootstrapping']['confidence_level'] = 1.5 # Invalid
        engine = BootstrappingEngine(base_config, mock_logger)
        engine.bootstrap(sample_predictions_df, "test", "run1")
        
        mock_logger.warning.assert_called_with(
            "Invalid confidence level (1.5). Must be between 0 and 1 (exclusive). Defaulting to 0.95."
        )

    def test_invalid_sample_ratio_warning(self, base_config, mock_logger, sample_predictions_df):
        """Tests that an invalid sample ratio is corrected and logs a warning."""
        base_config['bootstrapping']['sample_ratio'] = -0.5 # Invalid
        engine = BootstrappingEngine(base_config, mock_logger)
        engine.bootstrap(sample_predictions_df, "test", "run1")
        
        mock_logger.warning.assert_called_with(
            "Sample ratio (-0.5) must be greater than 0. Defaulting to 1.0."
        )

    def test_small_sample_size_warning(self, base_config, mock_logger):
        """Tests that a warning is logged for a statistically small bootstrap sample size."""
        # A small dataframe where ratio * n_rows < 5
        small_df = pd.DataFrame({'true_angle': [1], 'pred_angle': [2], 'abs_error': [1]})
        base_config['bootstrapping']['sample_ratio'] = 1.0
        engine = BootstrappingEngine(base_config, mock_logger)
        
        engine.bootstrap(small_df, "test", "run1")
        
        mock_logger.warning.assert_called_with(
            "Calculated bootstrap sample size (1) is less than the recommended minimum (5). "
            "This may affect statistical significance. Consider increasing 'num_samples' or 'sample_ratio'."
        )

    def test_reproducibility_with_seed(self, base_config, mock_logger, sample_predictions_df):
        """Tests that two runs with the same seed produce identical results."""
        # Run 1
        engine1 = BootstrappingEngine(base_config, mock_logger)
        results1 = engine1.bootstrap(sample_predictions_df, "test", "run1")
        
        # Run 2 with a new engine but same config
        engine2 = BootstrappingEngine(base_config, mock_logger)
        results2 = engine2.bootstrap(sample_predictions_df, "test", "run2")
        
        # The results should be identical due to the fixed seed
        assert results1['cmae']['mean'] == results2['cmae']['mean']
        assert results1['cmae']['lower'] == results2['cmae']['lower']

    def test_compute_ci_logic(self, base_config, mock_logger):
        """Tests the confidence interval calculation logic with known data."""
        engine = BootstrappingEngine(base_config, mock_logger)
        
        # 100 samples from 0 to 99
        metrics_list = [{'cmae': i} for i in range(100)]
        ci_results = engine._compute_ci(metrics_list, 0.9) # 90% CI
        
        # For a 90% CI, alpha=0.1, lower_p=5, upper_p=95
        # With data 0..99, the 5th percentile is 4.95, and 95th is 94.05
        assert np.isclose(ci_results['cmae']['lower'], 4.95)
        assert np.isclose(ci_results['cmae']['upper'], 94.05)
        assert np.isclose(ci_results['cmae']['mean'], 49.5)
