import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from unittest.mock import MagicMock
from modules.diagnostics_engine import DiagnosticsEngine

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    """Provides a mock logger for tests."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def base_config(tmp_path):
    """Provides a base configuration for the DiagnosticsEngine."""
    return {
        'data': {'hs_column': 'hs'}, # Use 'hs' to match sample_df
        'outputs': {'base_results_dir': str(tmp_path)},
        'diagnostics': {
            'dpi': 50,
            'save_format': 'png',
            # Enable all plots by default for broad testing
            'generate_index_plots': True,
            'generate_scatter_plots': True,
            'generate_residual_plots': True,
            'generate_distribution_plots': True,
            'generate_qq_plots': True,
            'generate_per_hs_accuracy': True
        }
    }

@pytest.fixture
def sample_predictions_df():
    """Provides a sample predictions DataFrame."""
    n = 50
    return pd.DataFrame({
        'true_angle': np.random.uniform(0, 360, n),
        'pred_angle': np.random.uniform(0, 360, n),
        'error': np.random.normal(0, 5, n),
        'abs_error': np.abs(np.random.normal(0, 5, n)),
        'hs': np.random.uniform(1, 5, n), # Match config hs_column
        'hs_bin': np.random.randint(0, 5, n)
    })

# --- Test Cases ---

class TestDiagnosticsEngine:

    def test_generate_all_creates_directories_and_plots(self, base_config, mock_logger, sample_predictions_df, tmp_path):
        """Tests that a full run creates all expected directories and plot files."""
        engine = DiagnosticsEngine(base_config, mock_logger)
        engine.generate_all(sample_predictions_df, "test", "run1")
        
        root_dir = Path(tmp_path) / "08_DIAGNOSTICS"
        
        # Check all directories are created
        expected_dirs = ['index_plots', 'scatter_plots', 'residual_plots', 'distribution_plots', 'qq_plots', 'per_hs_plots']
        for d in expected_dirs:
            assert (root_dir / d).is_dir()
            
        # Check that a plot file exists in each directory
        assert len(list((root_dir / "index_plots").glob("*.png"))) > 0
        assert len(list((root_dir / "scatter_plots").glob("*.png"))) > 0
        assert len(list((root_dir / "residual_plots").glob("*.png"))) > 0
        assert len(list((root_dir / "distribution_plots").glob("*.png"))) > 0
        assert len(list((root_dir / "qq_plots").glob("*.png"))) > 0
        assert len(list((root_dir / "per_hs_plots").glob("*.png"))) > 0

    def test_plot_generation_flags(self, base_config, mock_logger, sample_predictions_df, tmp_path):
        """Tests that plot generation can be disabled via config flags."""
        # Disable two plot types
        base_config['diagnostics']['generate_scatter_plots'] = False
        base_config['diagnostics']['generate_qq_plots'] = False
        
        engine = DiagnosticsEngine(base_config, mock_logger)
        engine.generate_all(sample_predictions_df, "test", "run1")
        
        root_dir = Path(tmp_path) / "08_DIAGNOSTICS"
        
        # Check that scatter plot was NOT generated
        assert not (root_dir / "scatter_plots" / "actual_vs_pred_test.png").exists()
        
        # Check that QQ plot was NOT generated
        assert not (root_dir / "qq_plots" / "qq_plot_test.png").exists()
        
        # Check that another plot type WAS generated
        assert (root_dir / "residual_plots" / "residuals_vs_pred_test.png").exists()

    def test_missing_hs_column_skips_hs_plots(self, base_config, mock_logger, sample_predictions_df, tmp_path):
        """Tests that per-Hs plots are skipped if the Hs column is missing."""
        df_no_hs = sample_predictions_df.drop(columns=['hs'])
        
        engine = DiagnosticsEngine(base_config, mock_logger)
        engine.generate_all(df_no_hs, "test", "run1")
        
        # Should warn but not crash
        mock_logger.warning.assert_called_with("Hs column 'hs' not found in predictions. Skipping Per-Hs plots.")
        
        # Check that no per_hs plots were created
        root_dir = Path(tmp_path) / "08_DIAGNOSTICS"
        assert len(list((root_dir / "per_hs_plots").glob("*.png"))) == 0

    def test_missing_hs_bin_column_skips_hs_bin_plot(self, base_config, mock_logger, sample_predictions_df, tmp_path):
        """Tests that only the Hs bin plot is skipped if the 'hs_bin' column is missing."""
        df_no_hs_bin = sample_predictions_df.drop(columns=['hs_bin'])
        
        engine = DiagnosticsEngine(base_config, mock_logger)
        engine.generate_all(df_no_hs_bin, "test", "run1")
        
        root_dir = Path(tmp_path) / "08_DIAGNOSTICS"
        
        # The scatter plot should still exist
        assert (root_dir / "per_hs_plots" / "error_vs_hs_scatter_test.png").exists()
        
        # The boxplot by bin should NOT exist
        assert not (root_dir / "per_hs_plots" / "error_vs_hs_bin_test.png").exists()

    def test_empty_dataframe_handling(self, base_config, mock_logger, tmp_path):
        """Tests that the engine handles an empty DataFrame gracefully without errors."""
        engine = DiagnosticsEngine(base_config, mock_logger)
        
        # This should run without raising any exceptions
        try:
            engine.generate_all(pd.DataFrame(), "test", "run1")
        except Exception as e:
            pytest.fail(f"DiagnosticsEngine failed on empty DataFrame: {e}")
            
        # Check that no plot files were created
        root_dir = Path(tmp_path) / "08_DIAGNOSTICS"
        assert len(list(root_dir.rglob("*.png"))) == 0
