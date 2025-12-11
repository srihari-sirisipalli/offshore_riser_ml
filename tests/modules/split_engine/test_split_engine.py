import pytest
from unittest.mock import Mock, patch, call
import pandas as pd
import numpy as np
from pathlib import Path
from modules.split_engine.split_engine import SplitEngine
from modules.base.base_engine import BaseEngine # For inheritance

# Concrete implementation for testing SplitEngine
class ConcreteSplitEngine(SplitEngine):
    def __init__(self, config, logger, output_dir_name="02_SMART_SPLIT", standard_output_dir_name="02_MASTER_SPLITS"):
        # Store the desired names first
        self._test_output_dir_name = output_dir_name
        self._test_standard_output_dir_name = standard_output_dir_name

        # Call BaseEngine.__init__ first
        super().__init__(config, logger)
        
        # Override the paths set by BaseEngine for testing purposes
        self.output_dir = Path(config['outputs']['base_results_dir']) / output_dir_name
        self.standard_output_dir = Path(config['outputs']['base_results_dir']) / standard_output_dir_name

    def _get_engine_directory_name(self) -> str:
        # Return the name that BaseEngine expects from the test setup
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
        'splitting': {
            'test_size': 0.2,
            'val_size': 0.25, # 0.25 of remaining 0.8 = 0.2 overall
            'seed': 42,
            'drop_incomplete_bins': False
        },
        'data': {
            'hs_column': 'Hs_ft'
        },
        '_internal_seeds': {
            'split': 42 # For _perform_split
        }
    }

@pytest.fixture
def dummy_dataframe():
    """Provides a dummy DataFrame with columns needed for splitting."""
    np.random.seed(0)
    num_samples = 100
    df = pd.DataFrame({
        'feature_A': np.random.rand(num_samples),
        'feature_B': np.random.rand(num_samples),
        'Hs_ft': np.random.randint(5, 50, num_samples),
        'angle_deg': np.random.randint(0, 360, num_samples),
        'combined_bin': [f'{h}_{a}' for h, a in zip(np.random.randint(5, 50, num_samples), np.random.randint(0, 360, num_samples))],
        'hs_bin': np.random.randint(0, 5, num_samples),
        'angle_bin': np.random.randint(0, 8, num_samples),
    })
    return df

@pytest.fixture
def dummy_splits(dummy_dataframe):
    """Provides dummy train, val, test DataFrames."""
    train = dummy_dataframe.iloc[:60].copy()
    val = dummy_dataframe.iloc[60:80].copy()
    test = dummy_dataframe.iloc[80:].copy()
    return train, val, test

@pytest.fixture(autouse=True)
def mock_save_dataframe():
    """Mocks save_dataframe from utils.file_io."""
    with patch('modules.split_engine.split_engine.save_dataframe') as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_train_test_split():
    """Mocks sklearn.model_selection.train_test_split."""
    with patch('modules.split_engine.split_engine.train_test_split') as mock:
        # Mock a two-stage split
        mock.side_effect = [
            (Mock(spec=pd.DataFrame), Mock(spec=pd.DataFrame)), # For train_val_main, test
            (Mock(spec=pd.DataFrame), Mock(spec=pd.DataFrame))  # For train, val
        ]
        yield mock

@pytest.fixture(autouse=True)
def mock_plt():
    """Mocks matplotlib.pyplot functions."""
    with patch('modules.split_engine.split_engine.plt') as mock:
        yield mock

# --- Test cases for _save_splits ---

def test_save_splits_to_both_dirs(base_config, mock_logger, mock_save_dataframe, tmp_path, dummy_splits):
    """
    Tests that _save_splits saves to both legacy and standard directories when they are different.
    """
    engine = ConcreteSplitEngine(base_config, mock_logger, 
                                 output_dir_name="02_SMART_SPLIT", 
                                 standard_output_dir_name="02_MASTER_SPLITS")
    
    train, val, test = dummy_splits
    engine._save_splits(train, val, test)
    
    # Expected calls (3 for legacy, 3 for standard)
    assert mock_save_dataframe.call_count == 6
    
    # Extract paths from actual calls
    actual_paths = [call_args[0][1] for call_args in mock_save_dataframe.call_args_list]

    # Check legacy paths
    legacy_output_dir = tmp_path / "02_SMART_SPLIT"
    expected_legacy_paths = [
        legacy_output_dir / "train.parquet",
        legacy_output_dir / "val.parquet",
        legacy_output_dir / "test.parquet",
    ]
    for p in expected_legacy_paths:
        assert p in actual_paths, f"Expected legacy path {p} not found in actual calls."

    # Check standard paths
    standard_output_dir = tmp_path / "02_MASTER_SPLITS"
    expected_standard_paths = [
        standard_output_dir / "train.parquet",
        standard_output_dir / "val.parquet",
        standard_output_dir / "test.parquet",
    ]
    for p in expected_standard_paths:
        assert p in actual_paths, f"Expected standard path {p} not found in actual calls."

def test_save_splits_to_single_dir(base_config, mock_logger, mock_save_dataframe, tmp_path, dummy_splits):
    """
    Tests that _save_splits saves only to one directory when legacy and standard are the same.
    """
    engine = ConcreteSplitEngine(base_config, mock_logger, 
                                 output_dir_name="02_MASTER_SPLITS", # Make them the same
                                 standard_output_dir_name="02_MASTER_SPLITS")
    
    train, val, test = dummy_splits
    engine._save_splits(train, val, test)
    
    # Expected calls (3 for one directory)
    assert mock_save_dataframe.call_count == 3
    
    # Extract paths from actual calls
    actual_paths = [call_args[0][1] for call_args in mock_save_dataframe.call_args_list]

    # Check paths
    output_dir = tmp_path / "02_MASTER_SPLITS"
    expected_paths = [
        output_dir / "train.parquet",
        output_dir / "val.parquet",
        output_dir / "test.parquet",
    ]
    for p in expected_paths:
        assert p in actual_paths, f"Expected path {p} not found in actual calls."

# --- Test cases for execute workflow ---

def test_execute_workflow(base_config, mock_logger, mock_save_dataframe, mock_train_test_split, mock_plt, dummy_dataframe, dummy_splits):
    """
    Tests the overall execute workflow, ensuring internal methods are called and splits are saved.
    """
    train_df, val_df, test_df = dummy_splits # Unpack for direct use in patching where actual dfs might be needed

    # Patch internal methods called by execute to control their behavior
    with (\
        patch.object(SplitEngine, '_determine_stratification_strategy', return_value='combined_bin') as mock_determine_strat, \
        patch.object(SplitEngine, '_perform_split', return_value=dummy_splits) as mock_perform_split, \
        patch.object(SplitEngine, '_generate_balance_report') as mock_generate_balance_report, \
        patch.object(SplitEngine, '_generate_split_plots') as mock_generate_split_plots, \
        patch.object(SplitEngine, '_save_splits') as mock_save_splits, \
        patch.object(SplitEngine, '_compute_signature', return_value=None), \
        patch.object(SplitEngine, '_try_load_cached_split', return_value=None), \
        patch.object(SplitEngine, '_store_cached_split') \
    ):

        engine = ConcreteSplitEngine(base_config, mock_logger)
        
        # Ensure output directories exist for report/plot generation (mocking mkdir behavior)
        engine.output_dir.mkdir(parents=True, exist_ok=True)
        engine.standard_output_dir.mkdir(parents=True, exist_ok=True)


        result_train, result_val, result_test = engine.execute(dummy_dataframe, "test_run_id")
        
        # Assert initial steps
        mock_determine_strat.assert_called_once_with(dummy_dataframe)
        mock_perform_split.assert_called_once_with(dummy_dataframe, 'combined_bin')
        
        # Assert reports and plots generated for both legacy and standard
        assert mock_generate_balance_report.call_count == 2
        mock_generate_balance_report.assert_has_calls([
            call(result_train, result_val, result_test, 'combined_bin', engine.output_dir),
            call(result_train, result_val, result_test, 'combined_bin', engine.standard_output_dir),
        ], any_order=True)

        assert mock_generate_split_plots.call_count == 2
        mock_generate_split_plots.assert_has_calls([
            call(result_train, result_val, result_test, engine.output_dir),
            call(result_train, result_val, result_test, engine.standard_output_dir),
        ], any_order=True)
        
        # Assert splits are saved (this mock will be called with the return values from _perform_split)
        mock_save_splits.assert_called_once_with(result_train, result_val, result_test)

        # Assert correct return values
        assert result_train is dummy_splits[0]
        assert result_val is dummy_splits[1]
        assert result_test is dummy_splits[2]

        mock_logger.info.assert_any_call("Starting Split Engine execution...")
        mock_logger.info.assert_any_call("Splits saved: Train=60, Val=20, Test=20") # Based on dummy_splits sizes
