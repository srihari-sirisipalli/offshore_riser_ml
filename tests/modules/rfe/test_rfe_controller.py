import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import logging # Import logging
from pathlib import Path
from modules.rfe.rfe_controller import RFEController
from utils.results_layout import ResultsLayoutManager # To patch this for mirror_baseline_outputs
from modules.error_analysis_engine.safety_analysis import safety_threshold_summary # To patch this

# Concrete class for testing RFEController without initializing all sub-engines
class MockRFEController(RFEController):
    def __init__(self, config: dict, logger: logging.Logger, base_dir: Path):
        self.config = config
        self.logger = logger
        self.base_dir = base_dir
        self.run_id = base_dir.name
        self.excel_copy = config.get("outputs", {}).get("save_excel_copy", False)
        self.results_layout = Mock(spec=ResultsLayoutManager) # Mock ResultsLayoutManager
        self.results_layout.ensure_base_structure.return_value = None # Mock its methods
        self.results_layout.mirror_baseline_outputs.return_value = None
        self.current_round = 0
        self.safety_history = []
        # Mock engines used by _train_baseline_phase for mirror_baseline_outputs_called test
        self.evaluation_engine = Mock()
        self.error_analyzer = Mock()
        self.diagnostics_engine = Mock()
        # Initialize run_advanced_suite and run_dashboard attributes from config
        vis_cfg = config.get("visualization", {})
        self.run_advanced_suite = vis_cfg.get("run_advanced_suite", False)
        self.run_dashboard = vis_cfg.get("run_dashboard", False)
        # Mimic rfe_config with safety_gates
        self.rfe_config = config.get('iterative', {})

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
            'safety_gates': {
                'max_critical_pct': 10.0,
                'max_warning_pct': 20.0
            }
        },
        'visualization': {
            'run_advanced_suite': False, # Default to False for tests unless explicitly set
            'run_dashboard': False,      # Default to False for tests unless explicitly set
        },
        'data': { # Minimal data config for feature filtering if needed, though not directly in _summarize_safety
            'target_sin': 'target_sin',
            'target_cos': 'target_cos',
            'drop_columns': [],
            'hs_column': 'Hs_ft',
            'include_hs_in_features': True
        }
    }

@pytest.fixture
def rfe_controller(base_config, mock_logger, tmp_path):
    """Provides a MockRFEController instance."""
    return MockRFEController(base_config, mock_logger, tmp_path)

@pytest.fixture
def dummy_predictions_df():
    """Provides a dummy predictions DataFrame with abs_error, pred_angle, and row_index."""
    np.random.seed(42)
    num_samples = 100
    return pd.DataFrame({
        'row_index': range(num_samples),
        'abs_error': np.random.uniform(0, 30, num_samples),
        'true_angle': np.random.uniform(0, 360, num_samples),
        'pred_angle': np.random.uniform(0, 360, num_samples),
        'error': np.random.uniform(-10, 10, num_samples),
        'true_sin': np.sin(np.radians(np.random.uniform(0, 360, num_samples))),
        'pred_sin': np.sin(np.radians(np.random.uniform(0, 360, num_samples))),
        'true_cos': np.cos(np.radians(np.random.uniform(0, 360, num_samples))),
        'pred_cos': np.cos(np.radians(np.random.uniform(0, 360, num_samples))),
    })

@pytest.fixture
def dummy_features_only_df():
    """Provides a dummy features DataFrame (without predictions columns) with matching index."""
    np.random.seed(43)
    num_samples = 100
    df = pd.DataFrame({
        'row_index': range(num_samples),
        'feature_A': np.random.uniform(10, 20, num_samples),
        'feature_B': np.random.normal(0, 1, num_samples),
        'feature_C': np.random.randint(0, 5, num_samples),
    })
    return df

@pytest.fixture(autouse=True)
def mock_save_dataframe():
    """Mocks save_dataframe from utils.file_io."""
    with patch('modules.rfe.rfe_controller.save_dataframe') as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_safety_threshold_summary():
    """Mocks safety_threshold_summary from safety_analysis."""
    with patch('modules.rfe.rfe_controller.safety_threshold_summary') as mock:
        yield mock

# --- Test cases for _summarize_safety ---

def test_summarize_safety_gates_pass(rfe_controller, dummy_predictions_df, mock_save_dataframe, mock_safety_threshold_summary, tmp_path):
    """
    Tests _summarize_safety when safety gates should pass.
    """
    # Mock safety_threshold_summary to return data that passes gates
    mock_safety_threshold_summary.return_value = pd.DataFrame([
        {'tier': 'CRITICAL', 'percentage': 5.0},
        {'tier': 'WARNING', 'percentage': 5.0},
        {'tier': 'ACCEPTABLE', 'percentage': 90.0}
    ])
    
    round_dir = tmp_path / "ROUND_000"
    rfe_controller._summarize_safety(round_dir, dummy_predictions_df, dummy_predictions_df)
    
    # Assert safety_threshold_summary was called twice (for val and test)
    assert mock_safety_threshold_summary.call_count == 2
    
    # Assert save_dataframe was called for both safety_threshold_summary.parquet and safety_gate_status.parquet (for current round and all_rounds)
    assert mock_save_dataframe.call_count == 4 # 2 for safety_threshold_summary, 2 for safety_gate_status
    
    # Check safety_gate_status_all_rounds.parquet content for PASS
    gate_status_df = None
    for call_args in mock_save_dataframe.call_args_list:
        if "safety_gate_status_all_rounds.parquet" in str(call_args[0][1]):
            gate_status_df = call_args[0][0]
            break
    
    assert gate_status_df is not None
    assert gate_status_df['status'].iloc[0] == 'PASS'
    assert gate_status_df['status'].iloc[1] == 'PASS' # For both val and test splits

def test_summarize_safety_gates_warn(rfe_controller, dummy_predictions_df, mock_save_dataframe, mock_safety_threshold_summary, tmp_path):
    """
    Tests _summarize_safety when safety gates should warn.
    """
    rfe_controller.config['error_analysis']['safety_gates'] = {
        'max_critical_pct': 10.0, # max critical is 10
        'max_warning_pct': 20.0  # max warning is 20
    }
    # Mock safety_threshold_summary to return data that warns (critical > 10, but not > 20)
    mock_safety_threshold_summary.return_value = pd.DataFrame([
        {'tier': 'CRITICAL', 'percentage': 5.0},  # <= max_critical_pct (10.0)
        {'tier': 'WARNING', 'percentage': 25.0}, # > max_warning_pct (20.0) -> WARN
        {'tier': 'ACCEPTABLE', 'percentage': 70.0}
    ])
    
    round_dir = tmp_path / "ROUND_000"
    rfe_controller._summarize_safety(round_dir, dummy_predictions_df, dummy_predictions_df)
    
    gate_status_df = None
    for call_args in mock_save_dataframe.call_args_list:
        if "safety_gate_status_all_rounds.parquet" in str(call_args[0][1]):
            gate_status_df = call_args[0][0]
            break
    
            assert gate_status_df is not None
            assert gate_status_df['status'].iloc[0] == 'WARN'
            assert gate_status_df['status'].iloc[1] == 'WARN' # For both val and test splits
def test_summarize_safety_gates_fail(rfe_controller, dummy_predictions_df, mock_save_dataframe, mock_safety_threshold_summary, tmp_path):
    """
    Tests _summarize_safety when safety gates should fail.
    """
    rfe_controller.config['error_analysis']['safety_gates'] = {
        'max_critical_pct': 10.0, # max critical is 10
        'max_warning_pct': 20.0  # max warning is 20
    }
    # Mock safety_threshold_summary to return data that fails (critical > 20)
    mock_safety_threshold_summary.return_value = pd.DataFrame([
        {'tier': 'CRITICAL', 'percentage': 25.0}, # Over max_warning_pct (20.0) -> fail
        {'tier': 'WARNING', 'percentage': 5.0},
        {'tier': 'ACCEPTABLE', 'percentage': 70.0}
    ])
    
    round_dir = tmp_path / "ROUND_000"
    rfe_controller._summarize_safety(round_dir, dummy_predictions_df, dummy_predictions_df)
    
    gate_status_df = None
    for call_args in mock_save_dataframe.call_args_list:
        if "safety_gate_status_all_rounds.parquet" in str(call_args[0][1]):
            gate_status_df = call_args[0][0]
            break
    
            assert gate_status_df is not None
    
            assert gate_status_df['status'].iloc[0] == 'FAIL'
    
            assert gate_status_df['status'].iloc[1] == 'FAIL'

def test_summarize_safety_no_safety_gates_config(rfe_controller, dummy_predictions_df, mock_save_dataframe, mock_safety_threshold_summary, tmp_path):
    """
    Tests _summarize_safety when safety_gates config is not present.
    Should still save summary but not gate status.
    """
    # Remove safety_gates config
    del rfe_controller.config['error_analysis']['safety_gates']
    
    mock_safety_threshold_summary.return_value = pd.DataFrame([
        {'tier': 'CRITICAL', 'percentage': 5.0},
        {'tier': 'WARNING', 'percentage': 5.0},
        {'tier': 'ACCEPTABLE', 'percentage': 90.0}
    ])
    
    round_dir = tmp_path / "ROUND_000"
    rfe_controller._summarize_safety(round_dir, dummy_predictions_df, dummy_predictions_df)
    
    # Assert safety_threshold_summary was called twice
    assert mock_safety_threshold_summary.call_count == 2
    
    # Assert save_dataframe was called only for safety_threshold_summary
    save_calls = [str(c[0][1]) for c in mock_save_dataframe.call_args_list]
    assert any("safety_threshold_summary.parquet" in s for s in save_calls)
    assert any("safety_threshold_summary_all_rounds.parquet" in s for s in save_calls)
    assert not any("safety_gate_status.parquet" in s for s in save_calls) # Should not be saved
    assert not any("safety_gate_status_all_rounds.parquet" in s for s in save_calls) # Should not be saved
    assert mock_save_dataframe.call_count == 2 # Only for safety_threshold_summary and safety_threshold_summary_all_rounds

def test_mirror_baseline_outputs_called(rfe_controller, dummy_predictions_df, dummy_features_only_df, mock_save_dataframe, mock_safety_threshold_summary, tmp_path):
    """
    Tests that ResultsLayoutManager.mirror_baseline_outputs is called during _train_baseline_phase.
    This requires more extensive mocking of RFEController's internal methods.
    """
    # Mock all internal methods called by _train_baseline_phase
    with patch.object(rfe_controller, '_filter_to_active_features', return_value=dummy_features_only_df), \
         patch.object(rfe_controller, '_train_model_internal', return_value=Mock()), \
         patch.object(rfe_controller, '_make_predictions', return_value=dummy_predictions_df.copy()), \
         patch.object(rfe_controller.evaluation_engine, 'compute_metrics', return_value={'cmae': 5.0}), \
         patch.object(rfe_controller.error_analyzer, 'execute', return_value={}), \
         patch.object(rfe_controller.diagnostics_engine, 'execute', return_value={}):

            round_dir = tmp_path / "ROUND_000"
            round_dir.mkdir() # Ensure round_dir exists for mirror_baseline_outputs
            (round_dir / "03_BASE_MODEL_RESULTS").mkdir() # Ensure this dir also exists

            # Call _train_baseline_phase, which should then call _summarize_safety and mirror_baseline_outputs
            rfe_controller._train_baseline_phase(round_dir, Mock(), Mock(), Mock(), Mock())
            
            # Assert that mirror_baseline_outputs was called with the correct round_dir
            rfe_controller.results_layout.mirror_baseline_outputs.assert_called_once_with(round_dir)