import pytest
from unittest.mock import Mock, patch, call
import pandas as pd
import numpy as np
import json
from pathlib import Path
import compileall # To check Python code validity

from modules.reporting_engine.reconstruction_mapper import ReconstructionMapper

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
        'data': { # Minimal data config for snippet generation
            'target_sin': 'sin_angle',
            'target_cos': 'cos_angle',
        }
    }

@pytest.fixture
def reconstruction_mapper(base_config, mock_logger):
    """Provides a ReconstructionMapper instance."""
    return ReconstructionMapper(base_config, mock_logger)

@pytest.fixture
def dummy_rounds_history():
    """Provides a list of dummy rounds history dictionaries."""
    return [
        {
            "round": 0,
            "n_features": 10,
            "dropped_feature": "None",
            "metrics": {"cmae": 5.0, "val_cmae": 5.0, "accuracy_at_5deg": 80.0},
            "hyperparameters": {"model_name": "ExtraTreesRegressor", "n_estimators": 100},
            "stopping_reason": "Continue",
            "active_features_list": ["feat_A", "feat_B", "feat_C"]
        },
        {
            "round": 1,
            "n_features": 9,
            "dropped_feature": "feat_C",
            "metrics": {"cmae": 4.5, "val_cmae": 4.5, "accuracy_at_5deg": 85.0},
            "hyperparameters": {"model_name": "RandomForestRegressor", "n_estimators": 150},
            "stopping_reason": "Min Feature Count Reached",
            "active_features_list": ["feat_A", "feat_B"]
        }
    ]

@pytest.fixture
def dummy_data_files_info(tmp_path):
    """Provides dummy data files info."""
    return {
        "train_path": str(tmp_path / "data" / "train.parquet"),
        "val_path": str(tmp_path / "data" / "val.parquet"),
        "input_hash": "abcdef12345",
    }

@pytest.fixture(autouse=True)
def mock_file_io():
    """Mocks save_dataframe and read_dataframe."""
    with patch('modules.reporting_engine.reconstruction_mapper.save_dataframe') as mock_save_df, \
         patch('modules.reporting_engine.reconstruction_mapper.read_dataframe') as mock_read_df:
        # Mock read_dataframe to return a dummy DataFrame to prevent errors during copy
        mock_read_df.return_value = pd.DataFrame({'col': [1]})
        yield mock_save_df, mock_read_df

# --- Test cases ---

def test_generate_mapping_saves_and_copies_artifacts(reconstruction_mapper, dummy_rounds_history, dummy_data_files_info, mock_file_io, tmp_path):
    """
    Tests that generate_mapping saves all artifacts to output_dir and copies them
    to the standardized 97_RECONSTRUCTION_MAPPING directory.
    """
    mock_save_df, mock_read_df = mock_file_io
    output_dir = tmp_path / "run_output_dir"
    output_dir.mkdir() # Ensure output_dir exists
    
    reconstruction_mapper.generate_mapping(dummy_rounds_history, dummy_data_files_info, output_dir)
    
    # Expected files (5 dataframes + 1 index)
    expected_filenames = [
        "model_reconstruction_summary.parquet",
        "model_reconstruction_hyperparameters.parquet",
        "model_reconstruction_features.parquet",
        "model_reconstruction_data_files.parquet",
        "model_reconstruction_code.parquet",
        "model_reconstruction_mapping.parquet",
    ]
    
    # Assert save_dataframe was called for each file in output_dir AND std_dir
    # (6 files * 2 locations = 12 calls)
    assert mock_save_df.call_count == 12
    
    # Assert read_dataframe was called for each file once when copying from output_dir to std_dir
    assert mock_read_df.call_count == len(expected_filenames) # 6 calls
    
    # Check that standard_dir was created
    std_dir = tmp_path / "97_RECONSTRUCTION_MAPPING"
    assert std_dir.is_dir()

    # Get the DataFrame that _build_summary_sheet would produce for comparison
    df_summary_expected = reconstruction_mapper._build_summary_sheet(dummy_rounds_history)

    # Verify call arguments (checking one example path for each location)
    output_summary_call_found = False
    std_summary_call_found = False

    for args, kwargs in mock_save_df.call_args_list:
        if args[1] == (output_dir / "model_reconstruction_summary.parquet"):
            pd.testing.assert_frame_equal(args[0], df_summary_expected) # Correct comparison
            output_summary_call_found = True
        elif args[1] == (std_dir / "model_reconstruction_summary.parquet"):
            # When copying to std_dir, it reads the content that was saved to output_dir
            # and then saves it. The content will be the mock_read_df's return_value.
            pd.testing.assert_frame_equal(args[0], mock_read_df.return_value)
            std_summary_call_found = True    


def test_build_summary_sheet(reconstruction_mapper, dummy_rounds_history):
    """Tests that _build_summary_sheet creates a DataFrame with correct structure and values."""
    df_summary = reconstruction_mapper._build_summary_sheet(dummy_rounds_history)
    
    assert isinstance(df_summary, pd.DataFrame)
    assert df_summary.shape == (2, 6) # 2 rounds, 6 columns
    assert list(df_summary.columns) == ["Round", "N_Features", "Dropped_Feature", "Val_CMAE", "Val_Accuracy_5deg", "Stop_Reason"]
    
    assert df_summary.iloc[0]["Round"] == 0
    assert df_summary.iloc[0]["N_Features"] == 10
    assert df_summary.iloc[0]["Val_CMAE"] == 5.0
    
    assert df_summary.iloc[1]["Round"] == 1
    assert df_summary.iloc[1]["Dropped_Feature"] == "feat_C"
    assert df_summary.iloc[1]["Val_Accuracy_5deg"] == 85.0

def test_generate_python_snippet_valid_code(reconstruction_mapper, dummy_rounds_history, dummy_data_files_info):
    """Tests that _generate_python_snippet generates valid Python code."""
    round_data = dummy_rounds_history[0]
    snippet = reconstruction_mapper._generate_python_snippet(round_data, dummy_data_files_info)
    
    assert isinstance(snippet, str)
    assert "import pandas as pd" in snippet
    assert "model = ExtraTreesRegressor(" in snippet
    assert "features = ['feat_A', 'feat_B', 'feat_C']" in snippet # From active_features_list    
    # Attempt to compile the code to check for basic syntax validity
    try:
        compile(snippet, '<string>', 'exec')
    except SyntaxError as e:
        pytest.fail(f"Generated Python snippet has a SyntaxError: {e}")
    except Exception as e:
        pytest.fail(f"Failed to compile snippet: {e}")
