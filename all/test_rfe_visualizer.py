import pytest
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from modules.visualization.rfe_visualizer import RFEVisualizer

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def viz_config(tmp_path):
    return {
        'outputs': {'base_results_dir': str(tmp_path)}
    }

@pytest.fixture
def sample_lofo_results():
    return [
        {'feature': 'f1', 'delta_cmae': -0.5, 'val_cmae': 1.0, 'val_crmse': 2.0, 'val_accuracy_5deg': 90},
        {'feature': 'f2', 'delta_cmae': 0.2, 'val_cmae': 1.7, 'val_crmse': 2.5, 'val_accuracy_5deg': 85},
        {'feature': 'f3', 'delta_cmae': 0.0, 'val_cmae': 1.5, 'val_crmse': 2.3, 'val_accuracy_5deg': 88},
    ]

@pytest.fixture
def sample_preds():
    # 50 random samples
    true = np.random.uniform(0, 360, 50)
    pred = true + np.random.normal(0, 5, 50)
    error = pred - true
    
    return pd.DataFrame({
        'true_angle': true,
        'pred_angle': pred,
        'error': error,
        'abs_error': np.abs(error)
    })

# --- Tests ---

class TestRFEVisualizer:

    def test_visualize_lofo_impact(self, viz_config, mock_logger, sample_lofo_results, tmp_path):
        """Checks creation of LOFO bar charts and heatmaps."""
        viz = RFEVisualizer(viz_config, mock_logger)
        round_dir = Path(viz_config['outputs']['base_results_dir']) / "ROUND_000"
        
        viz.visualize_lofo_impact(round_dir, sample_lofo_results)
        
        plot_dir = round_dir / "04_FEATURE_EVALUATION" / "feature_evaluation_plots"
        assert plot_dir.exists()
        assert (plot_dir / "lofo_comparison_bar.png").exists()
        assert (plot_dir / "lofo_error_heatmap.png").exists()

    def test_visualize_comparison(self, viz_config, mock_logger, sample_preds, tmp_path):
        """Checks creation of comparison overlay plots."""
        viz = RFEVisualizer(viz_config, mock_logger)
        round_dir = Path(viz_config['outputs']['base_results_dir']) / "ROUND_000"
        
        # Create a slightly different df for 'dropped' to simulate difference
        dropped_preds = sample_preds.copy()
        dropped_preds['abs_error'] = dropped_preds['abs_error'] * 0.9 # Improved
        
        viz.visualize_comparison(round_dir, sample_preds, dropped_preds, "feat_X")
        
        plot_dir = round_dir / "06_COMPARISON" / "comparison_plots"
        assert plot_dir.exists()
        assert (plot_dir / "before_after_error_cdf.png").exists()
        assert (plot_dir / "angle_scatter_overlay.png").exists()
        assert (plot_dir / "residual_distribution_overlay.png").exists()

    def test_handle_empty_lofo(self, viz_config, mock_logger, tmp_path):
        """Ensures no crash on empty results list."""
        viz = RFEVisualizer(viz_config, mock_logger)
        round_dir = Path(viz_config['outputs']['base_results_dir']) / "ROUND_000"
        
        # Should execute safely and create nothing (or just the dir)
        viz.visualize_lofo_impact(round_dir, [])
        
        plot_dir = round_dir / "04_FEATURE_EVALUATION" / "feature_evaluation_plots"
        assert plot_dir.exists()
        assert not (plot_dir / "lofo_comparison_bar.png").exists()