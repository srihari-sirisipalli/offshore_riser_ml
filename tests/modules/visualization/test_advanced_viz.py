import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use the Agg backend for testing plots
import matplotlib.pyplot as plt # Import after setting backend

from modules.visualization.advanced_viz import AdvancedVisualizer

@pytest.fixture
def mock_logger():
    """Provides a mock logger instance."""
    return Mock()

@pytest.fixture
def advanced_visualizer(mock_logger):
    """Provides an AdvancedVisualizer instance with a mock config and logger."""
    config = {
        "visualization": {"parallel_plots": False}, # Disable parallel for simpler testing
        "execution": {"n_jobs": 1}
    }
    return AdvancedVisualizer(config=config, logger=mock_logger)

@pytest.fixture
def synthetic_dataframe():
    """Generates a synthetic DataFrame with common columns needed for plots."""
    np.random.seed(42)
    num_samples = 200

    data = {
        'Hs_ft': np.random.uniform(5, 50, num_samples),
        'true_angle': np.random.uniform(0, 360, num_samples),
        'pred_angle': np.random.uniform(0, 360, num_samples),
        'abs_error': np.random.uniform(0, 25, num_samples),
        'error': np.random.uniform(-20, 20, num_samples),
        'cluster_id': np.random.choice([0, 1, 2], num_samples),
        'round': np.random.randint(1, 5, num_samples),
        'pos_x': np.random.uniform(0, 100, num_samples),
        'pos_y': np.random.uniform(0, 100, num_samples),
    }
    df = pd.DataFrame(data)

    # Ensure some data near boundaries for boundary plots
    df.loc[:5, 'true_angle'] = np.random.uniform(350, 360, 6)
    df.loc[6:10, 'true_angle'] = np.random.uniform(0, 10, 5)

    # Add hs_bin and angle_bin for faceted_delta_grid
    df['hs_bin'] = pd.cut(df['Hs_ft'], bins=4, labels=[f'Hs_Bin_{i}' for i in range(4)])
    df['angle_bin'] = pd.cut(df['true_angle'], bins=4, labels=[f'Angle_Bin_{i}' for i in range(4)])
    df['model'] = np.random.choice(['Model_A', 'Model_B', 'Model_C'], num_samples)
    df['scenario'] = np.random.choice(['Scenario_X', 'Scenario_Y'], num_samples)

    # Add baseline/candidate errors for improvement heatmap
    df['abs_error_baseline'] = df['abs_error'] + np.random.uniform(-5, 5, num_samples)
    df['abs_error_candidate'] = df['abs_error'] + np.random.uniform(-5, 5, num_samples)


    return df

@pytest.fixture
def synthetic_metrics_history():
    """Generates synthetic metrics history for plot_round_progression."""
    data = {
        'round': np.tile(np.arange(1, 6), 4),
        'metric_name': ['val_cmae'] * 5 + ['test_cmae'] * 5 + ['val_accuracy_at_10deg'] * 5 + ['test_accuracy_at_10deg'] * 5,
        'value': np.random.uniform(1, 10, 20)
    }
    return pd.DataFrame(data)

@pytest.fixture
def synthetic_deltas():
    """Generates synthetic deltas for plot_improvement_waterfall."""
    data = {
        'step': [f'step_{i}' for i in range(5)],
        'delta': np.random.uniform(-5, 5, 5)
    }
    return pd.DataFrame(data)

@pytest.fixture
def synthetic_comparison_df(synthetic_dataframe):
    """Generates a synthetic comparison DataFrame for plot_faceted_delta_grid."""
    df_comp = synthetic_dataframe.copy()
    # Ensure there are at least two distinct 'model' values
    df_comp['model'] = np.random.choice(['Model_Base', 'Model_Candidate'], len(df_comp))
    # Add some variation to abs_error for delta calculation
    df_comp.loc[df_comp['model'] == 'Model_Candidate', 'abs_error'] += np.random.uniform(-2, 2, len(df_comp[df_comp['model'] == 'Model_Candidate']))
    return df_comp


# Helper to check if a plot file was created
def assert_plot_created(output_path: Path):
    assert output_path.is_file()
    assert output_path.stat().st_size > 0 # Check if file is not empty

# --- Individual Plot Function Tests (focus on new plots first) ---

def test_plot_high_error_zoom_facets(advanced_visualizer, synthetic_dataframe, tmp_path):
    """Tests plot_high_error_zoom_facets produces a file."""
    output_path = tmp_path / "high_error_zoom_facets.png"
    advanced_visualizer.plot_high_error_zoom_facets(synthetic_dataframe, output_path)
    assert_plot_created(output_path)

def test_plot_high_error_zoom_facets_missing_columns(advanced_visualizer, mock_logger, tmp_path):
    """Tests plot_high_error_zoom_facets handles missing columns gracefully."""
    df_missing = pd.DataFrame({'Hs_ft': [10], 'true_angle': [180]}) # Missing 'abs_error'
    output_path = tmp_path / "high_error_zoom_facets_missing.png"
    advanced_visualizer.plot_high_error_zoom_facets(df_missing, output_path)
    mock_logger.warning.assert_called_with("High-error zoom facets skipped: missing required columns.")
    assert not output_path.is_file()

def test_plot_boundary_gradient(advanced_visualizer, synthetic_dataframe, tmp_path):
    """Tests plot_boundary_gradient produces a file."""
    output_path = tmp_path / "boundary_gradient.png"
    advanced_visualizer.plot_boundary_gradient(synthetic_dataframe, output_path)
    assert_plot_created(output_path)

def test_plot_boundary_gradient_missing_columns(advanced_visualizer, mock_logger, tmp_path):
    """Tests plot_boundary_gradient handles missing columns gracefully."""
    df_missing = pd.DataFrame({'true_angle': [10]}) # Missing 'abs_error'
    output_path = tmp_path / "boundary_gradient_missing.png"
    advanced_visualizer.plot_boundary_gradient(df_missing, output_path)
    mock_logger.warning.assert_called_with("Boundary gradient skipped: missing required columns.")
    assert not output_path.is_file()

def test_plot_filtered_error_dashboard(advanced_visualizer, synthetic_dataframe, tmp_path):
    """Tests plot_filtered_error_dashboard produces a file."""
    output_path = tmp_path / "filtered_error_dashboard.png"
    advanced_visualizer.plot_filtered_error_dashboard(synthetic_dataframe, output_path)
    assert_plot_created(output_path)

def test_plot_filtered_error_dashboard_missing_columns(advanced_visualizer, mock_logger, tmp_path):
    """Tests plot_filtered_error_dashboard handles missing columns gracefully."""
    df_missing = pd.DataFrame({'Hs_ft': [10], 'abs_error': [5]}) # Missing 'true_angle'
    output_path = tmp_path / "filtered_error_dashboard_missing.png"
    advanced_visualizer.plot_filtered_error_dashboard(df_missing, output_path)
    mock_logger.warning.assert_called_with("Filtered error dashboard skipped: missing required columns.")
    assert not output_path.is_file()

def test_plot_operating_envelope_overlay(advanced_visualizer, synthetic_dataframe, tmp_path):
    """Tests plot_operating_envelope_overlay produces a file."""
    output_path = tmp_path / "operating_envelope_overlay.png"
    advanced_visualizer.plot_operating_envelope_overlay(synthetic_dataframe, output_path)
    assert_plot_created(output_path)

def test_plot_operating_envelope_overlay_missing_columns(advanced_visualizer, mock_logger, tmp_path):
    """Tests plot_operating_envelope_overlay handles missing columns gracefully."""
    df_missing = pd.DataFrame({'Hs_ft': [10], 'abs_error': [5]}) # Missing 'true_angle'
    output_path = tmp_path / "operating_envelope_overlay_missing.png"
    advanced_visualizer.plot_operating_envelope_overlay(df_missing, output_path)
    mock_logger.warning.assert_called_with("Operating envelope overlay skipped: missing required columns.")
    assert not output_path.is_file()

def test_plot_cluster_evolution_overlay(advanced_visualizer, synthetic_dataframe, tmp_path):
    """Tests plot_cluster_evolution_overlay produces a file."""
    output_path = tmp_path / "cluster_evolution_overlay.png"
    advanced_visualizer.plot_cluster_evolution_overlay(synthetic_dataframe, output_path)
    assert_plot_created(output_path)

def test_plot_cluster_evolution_overlay_empty_df(advanced_visualizer, mock_logger, tmp_path):
    """Tests plot_cluster_evolution_overlay handles empty DataFrame gracefully."""
    output_path = tmp_path / "cluster_evolution_overlay_empty.png"
    advanced_visualizer.plot_cluster_evolution_overlay(pd.DataFrame(), output_path)
    mock_logger.warning.assert_called_with("Cluster evolution overlay skipped: empty dataframe.")
    assert not output_path.is_file()

def test_plot_faceted_delta_grid(advanced_visualizer, synthetic_comparison_df, tmp_path):
    """Tests plot_faceted_delta_grid produces a file."""
    output_path = tmp_path / "faceted_delta_grid.png"
    advanced_visualizer.plot_faceted_delta_grid(synthetic_comparison_df, output_path)
    assert_plot_created(output_path)

def test_plot_faceted_delta_grid_missing_columns(advanced_visualizer, mock_logger, tmp_path):
    """Tests plot_faceted_delta_grid handles missing columns gracefully."""
    df_missing = pd.DataFrame({'hs_bin': ['A'], 'angle_bin': ['B'], 'abs_error': [1], 'model': ['M']}) # Missing 'scenario' or another 'model'
    output_path = tmp_path / "faceted_delta_grid_missing.png"
    advanced_visualizer.plot_faceted_delta_grid(df_missing, output_path)
    mock_logger.warning.assert_any_call("Faceted delta grid skipped: need at least two models.")
    assert not output_path.is_file()

# --- run_default_suite tests ---

def test_run_default_suite_full_data(advanced_visualizer, synthetic_dataframe, synthetic_metrics_history, synthetic_comparison_df, tmp_path):
    """Tests run_default_suite with full data, ensuring all expected plots are attempted."""
    output_dir = tmp_path / "default_suite_outputs"
    advanced_visualizer.run_default_suite(
        df=synthetic_dataframe,
        output_dir=output_dir,
        split_name="test",
        hs_col="Hs_ft",
        metrics_history=synthetic_metrics_history,
        comparison_df=synthetic_comparison_df
    )

    # Assert that a representative set of plots are created
    expected_plots = [
        "error_surface_3d_test.png",
        "optimal_zone_map_test.png",
        "error_vs_hs_response_test.png",
        "circular_error_vs_angle_test.png",
        "faceted_error_by_hs_bins_test.png",
        "faceted_error_by_angle_bins_test.png",
        "residual_diagnostics_test.png",
        "boundary_analysis_test.png",
        "error_distribution_by_hs_bins_test.png",
        "performance_contour_map_test.png",
        "high_error_zoom_test.png",
        "high_error_zoom_facets_test.png",
        "boundary_gradient_test.png",
        "filtered_error_dashboard_test.png",
        "operating_envelope_overlay_test.png",
        "faceted_delta_grid_test.png",
        "cluster_evolution_overlay_test.png",
        "round_progression_test.png",
    ]

    for plot_file in expected_plots:
        assert (output_dir / plot_file).is_file(), f"Plot {plot_file} was not created."
        assert (output_dir / plot_file).stat().st_size > 0, f"Plot {plot_file} is empty."

def test_run_default_suite_missing_df_columns_gracefully(advanced_visualizer, mock_logger, tmp_path):
    """
    Tests run_default_suite handles gracefully when the main DataFrame
    is missing columns required by sub-plots.
    """
    output_dir = tmp_path / "default_suite_missing_columns"
    df_minimal = pd.DataFrame({
        'Hs_ft': [10, 20],
        'true_angle': [90, 270],
        # Missing 'abs_error', 'pred_angle', 'error'
    })

    advanced_visualizer.run_default_suite(
        df=df_minimal,
        output_dir=output_dir,
        split_name="test",
        hs_col="Hs_ft"
    )

    # Check that warnings were logged for missing columns for various plots
    mock_logger.warning.assert_any_call("Missing required columns for 3D surface. Skipping.")
    mock_logger.warning.assert_any_call("Missing required columns for zone map. Skipping.")
    mock_logger.warning.assert_any_call("Missing required columns. Skipping.") # Generic for others

    # Assert that some files are NOT created due to missing data (e.g., 3D plot)
    assert not (output_dir / "error_surface_3d_test.png").is_file()
    # And potentially that some files ARE created if their requirements are met (e.g., if a plot only needs Hs_ft and true_angle, but that's not the case here for most)
    
    # Assert that output_dir is created, even if plots fail
    assert output_dir.is_dir()
