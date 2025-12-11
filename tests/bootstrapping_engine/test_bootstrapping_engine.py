import pytest
from unittest.mock import Mock, patch, call
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Import for mocking
from pathlib import Path
from modules.bootstrapping_engine.bootstrapping_engine import BootstrappingEngine

# Concrete implementation for testing BootstrappingEngine
class ConcreteBootstrappingEngine(BootstrappingEngine):
    def __init__(self, config, logger, output_dir_name="09_ADVANCED_ANALYTICS"):
        self._test_output_dir_name = output_dir_name
        super().__init__(config, logger)
        self.output_dir = Path(config['outputs']['base_results_dir']) / output_dir_name

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
        'bootstrapping': {
            'enabled': True, # Enabled by default for tests
            'n_samples': 10,
            'sample_ratio': 1.0,
            'confidence_level': 0.95
        },
        'splitting': { # For master_seed
            'seed': 42
        },
        '_internal_seeds': { # For bootstrap_seed
            'bootstrap': 123
        }
    }

@pytest.fixture
def bootstrapping_engine(base_config, mock_logger):
    """Provides a BootstrappingEngine instance."""
    return ConcreteBootstrappingEngine(base_config, mock_logger)

@pytest.fixture
def dummy_predictions_df():
    """Provides a dummy predictions DataFrame."""
    np.random.seed(42)
    num_samples = 100
    return pd.DataFrame({
        'true_angle': np.random.uniform(0, 360, num_samples),
        'pred_angle': np.random.uniform(0, 360, num_samples),
    })

@pytest.fixture(autouse=True)
def mock_save_dataframe():
    """Mocks save_dataframe from utils.file_io."""
    with patch('modules.bootstrapping_engine.bootstrapping_engine.save_dataframe') as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_compute_cmae():
    """Mocks compute_cmae from utils.circular_metrics."""
    with patch('modules.bootstrapping_engine.bootstrapping_engine.compute_cmae') as mock:
        mock.return_value = 5.0 # Constant CMAE for simplicity
        yield mock

@pytest.fixture(autouse=True)
def mock_plt():
    """Mocks matplotlib.pyplot functions."""
    with patch('modules.bootstrapping_engine.bootstrapping_engine.plt') as mock:
        yield mock

@pytest.fixture
def mock_numpy_random_state():
    """Mocks np.random.RandomState constructor and its choice method."""
    mock_rng_instance = Mock()
    mock_rng_instance.choice.return_value = np.arange(100) # num_samples for dummy_predictions_df
    with patch('numpy.random.RandomState', return_value=mock_rng_instance) as mock_RandomState_class:
        yield mock_rng_instance, mock_RandomState_class

# --- Test cases for BootstrappingEngine ---

def test_execute_disabled(bootstrapping_engine, dummy_predictions_df, mock_logger, mock_save_dataframe):
    """Tests that bootstrapping is skipped if disabled in config."""
    bootstrapping_engine.config['bootstrapping']['enabled'] = False
    bootstrapping_engine.enabled = False # Directly set for test as it's read in __init__
    
    result = bootstrapping_engine.execute(dummy_predictions_df, "val", "run_001")
    
    mock_logger.info.assert_called_with("Bootstrapping disabled in config.")
    mock_save_dataframe.assert_not_called()
    assert result == {}

def test_execute_empty_predictions(bootstrapping_engine, mock_logger, mock_save_dataframe):
    """Tests that bootstrapping warns and returns empty dict if predictions are empty."""
    bootstrapping_engine.config['bootstrapping']['enabled'] = True # Ensure enabled
    bootstrapping_engine.enabled = True
    
    empty_df = pd.DataFrame(columns=['true_angle', 'pred_angle'])
    result = bootstrapping_engine.execute(empty_df, "val", "run_001")
    
    mock_logger.warning.assert_called_with("No predictions available for bootstrapping.")
    mock_save_dataframe.assert_not_called()
    assert result == {}

def test_execute_generates_and_saves_ci_and_plot(bootstrapping_engine, dummy_predictions_df, mock_save_dataframe, mock_compute_cmae, mock_plt, mock_numpy_random_state, tmp_path):
    """
    Tests that execute runs bootstrapping, saves CI results and samples, and generates a plot.
    """
    mock_rng_instance, mock_RandomState_class = mock_numpy_random_state
    mock_compute_cmae.return_value = 5.0 # Ensure constant CMAE for easy CI prediction
    
    result = bootstrapping_engine.execute(dummy_predictions_df, "test", "run_001")
    
    # Assert compute_cmae was called n_samples times (10 in this config)
    assert mock_compute_cmae.call_count == 10
    
    # Assert RandomState.choice was called
    mock_rng_instance.choice.assert_called_with(len(dummy_predictions_df), size=len(dummy_predictions_df), replace=True)
    
    # Assert save_dataframe was called twice (for bootstrap_ci and bootstrap_samples)
    assert mock_save_dataframe.call_count == 2
        
    # Check the content of the CI DataFrame passed to save_dataframe
    ci_df_call_args = None
    samples_df_call_args = None
    for call_args, call_kwargs in mock_save_dataframe.call_args_list:
        if "bootstrap_ci.parquet" in str(call_args[1]):
            ci_df_call_args = call_args[0]
        elif "bootstrap_samples.parquet" in str(call_args[1]):
            samples_df_call_args = call_args[0]
    
    assert ci_df_call_args is not None
    assert isinstance(ci_df_call_args, pd.DataFrame)
    assert ci_df_call_args.iloc[0]['metric'] == 'CMAE'
    assert ci_df_call_args.iloc[0]['mean'] == 5.0
    assert ci_df_call_args.iloc[0]['ci_lower'] == 5.0
    assert ci_df_call_args.iloc[0]['ci_upper'] == 5.0
    assert ci_df_call_args.iloc[0]['n_bootstraps'] == 10
    
    assert samples_df_call_args is not None
    assert isinstance(samples_df_call_args, pd.DataFrame)
    assert samples_df_call_args.shape == (10, 1) # 10 samples, 1 metric (CMAE)
    
    # Define expected output directory for file path assertions
    expected_output_dir = tmp_path / "09_ADVANCED_ANALYTICS" / "bootstrapping" / "test"
    
    # Check that plt.savefig was called with the correct path
    mock_plt.savefig.assert_called_once_with(expected_output_dir / "bootstrap_dist_cmae.png")
    # Check return value
    assert result['cmae']['mean'] == 5.0

def test_compute_ci_calculates_correctly(bootstrapping_engine):
    """Tests _compute_ci calculates confidence intervals correctly."""
    metrics_list = [{'cmae': 1.0}, {'cmae': 2.0}, {'cmae': 3.0}, {'cmae': 4.0}, {'cmae': 5.0},
                    {'cmae': 6.0}, {'cmae': 7.0}, {'cmae': 8.0}, {'cmae': 9.0}, {'cmae': 10.0}]
    
    confidence = 0.80 # 80% CI means 10% on each side
    ci_results = bootstrapping_engine._compute_ci(metrics_list, confidence)
    
    assert ci_results['cmae']['mean'] == 5.5 # (1+10)/2
    assert ci_results['cmae']['lower'] == 1.9 # 10th percentile
    assert ci_results['cmae']['upper'] == 9.1 # 90th percentile

def test_plot_distribution_creates_plot(bootstrapping_engine, mock_plt, tmp_path):
    """Tests that _plot_distribution generates and saves a plot."""
    data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
    mean, lower, upper = 3.0, 1.5, 4.5
    output_dir = tmp_path / "plot_test"
    output_dir.mkdir()
    confidence = 0.95
    
    bootstrapping_engine._plot_distribution(data, mean, lower, upper, output_dir, confidence)
    
    mock_plt.hist.assert_called_once()
    mock_plt.savefig.assert_called_once_with(output_dir / "bootstrap_dist_cmae.png")
    mock_plt.close.assert_called_once()
