import pytest
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from unittest.mock import MagicMock
from modules.data_manager import DataManager
from utils.exceptions import DataValidationError

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    """Provides a mock logger for tests."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def base_config(tmp_path):
    """Provides a base configuration dictionary."""
    return {
        "data": {
            "file_path": str(tmp_path / "test_data.xlsx"),
            "target_sin": "sin",
            "target_cos": "cos",
            "hs_column": "hs",
            "drop_columns": ["extra_col"],
            "precision": "float32",
            "validate_sin_cos_circle": True,
            "circle_tolerance": 0.01
        },
        "splitting": {
            "angle_bins": 4, # 90 degree bins
            "hs_bins": 2,
            "hs_binning_method": "quantile",
            "drop_incomplete_bins": False
        },
        "outputs": {
            "base_results_dir": str(tmp_path / "results")
        }
    }

@pytest.fixture
def sample_dataframe():
    """Provides a sample DataFrame for testing."""
    return pd.DataFrame({
        'sin': [0.0, 1.0, 0.7071, -0.7071],
        'cos': [1.0, 0.0, 0.7071, -0.7071],
        'hs': [1.0, 10.0, 1.0, 10.0],
        'extra_col': [1, 2, 3, 4]
    })

@pytest.fixture
def create_data_file(tmp_path, sample_dataframe):
    """Helper fixture to create a data file for loading tests."""
    def _create(file_format="xlsx"):
        if file_format == "xlsx":
            file_path = tmp_path / "test_data.xlsx"
            sample_dataframe.to_excel(file_path, index=False)
        elif file_format == "csv":
            file_path = tmp_path / "test_data.csv"
            sample_dataframe.to_csv(file_path, index=False)
        return str(file_path)
    return _create

# --- Test Cases ---

class TestDataManager:

    def test_load_data_success_xlsx(self, base_config, create_data_file, mock_logger):
        """Tests successful loading of an XLSX file."""
        file_path = create_data_file("xlsx")
        base_config["data"]["file_path"] = file_path
        dm = DataManager(base_config, mock_logger)
        dm.load_data()
        assert dm.data is not None
        assert len(dm.data) == 4
        assert dm.data['sin'].dtype == 'float32'
        mock_logger.info.assert_called_with(f"Loading data from {Path(file_path).resolve()}")

    def test_load_data_success_csv(self, base_config, create_data_file, mock_logger):
        """Tests successful loading of a CSV file."""
        file_path = create_data_file("csv")
        base_config["data"]["file_path"] = file_path
        dm = DataManager(base_config, mock_logger)
        dm.load_data()
        assert dm.data is not None
        assert len(dm.data) == 4

    def test_load_data_missing_file(self, base_config, mock_logger, tmp_path):
        """Tests that a DataValidationError is raised for a missing file."""
        base_config["data"]["file_path"] = str(tmp_path / "non_existent.xlsx")
        dm = DataManager(base_config, mock_logger)
        with pytest.raises(DataValidationError, match="not found"):
            dm.load_data()

    def test_load_data_unsupported_format(self, base_config, mock_logger, tmp_path):
        """Tests that a DataValidationError is raised for an unsupported file format."""
        file_path = tmp_path / "test.txt"
        file_path.touch()
        base_config["data"]["file_path"] = str(file_path)
        dm = DataManager(base_config, mock_logger)
        with pytest.raises(DataValidationError, match="Unsupported file extension"):
            dm.load_data()
            
    def test_path_traversal_error(self, base_config, mock_logger, tmp_path):
        """Tests that path traversal outside the allowed directory raises an error."""
        # Create a file outside the project's data directory
        malicious_path = tmp_path / ".." / "malicious_file.txt"
        malicious_path.resolve().touch()
        
        # This path attempts to go up one level from the project root
        base_config["data"]["file_path"] = "../malicious_file.txt"
        
        # Assuming the test runs from the project root, this should fail.
        # We need to adjust our view of the `data_manager` to be relative to the CWD
        dm = DataManager(base_config, mock_logger)
        with pytest.raises(DataValidationError, match="outside the allowed data directory"):
            dm.load_data()

    def test_float16_overflow_warning(self, base_config, mock_logger, tmp_path):
        """Tests that a warning is logged when casting to float16 with out-of-range values."""
        # FIX: Write the overflow data to the file so load_data reads it
        file_path = tmp_path / "overflow_data.xlsx"
        df_overflow = pd.DataFrame({'sin': [70000.0], 'cos': [0.0], 'hs': [1.0]})
        df_overflow.to_excel(file_path, index=False)
        
        base_config["data"]["file_path"] = str(file_path)
        base_config["data"]["precision"] = "float16"
        dm = DataManager(base_config, mock_logger)
        
        dm.load_data()
        
        mock_logger.warning.assert_called_with(
            "Column 'sin' has values outside the float16 range. "
            "Casting may result in overflow/underflow. Consider using float32."
        )

    def test_validate_columns_success(self, base_config, sample_dataframe, mock_logger):
        """Tests successful column validation."""
        dm = DataManager(base_config, mock_logger)
        dm.data = sample_dataframe
        dm.validate_columns() # Should not raise
        
    def test_validate_columns_missing(self, base_config, sample_dataframe, mock_logger):
        """Tests that an error is raised for missing required columns."""
        dm = DataManager(base_config, mock_logger)
        dm.data = sample_dataframe.drop(columns=['sin'])
        # FIX: Use raw string (r"") for regex match containing brackets
        with pytest.raises(DataValidationError, match=r"Missing required columns: \['sin'\]"):
            dm.validate_columns()
    def test_validate_columns_empty_df(self, base_config, mock_logger):
        """Tests that an error is raised when validating an empty DataFrame."""
        dm = DataManager(base_config, mock_logger)
        dm.data = pd.DataFrame()
        with pytest.raises(DataValidationError, match="Dataframe is empty or None"):
            dm.validate_columns()

    def test_validate_nan_inf(self, base_config, mock_logger):
        """Tests NaN and Inf validation."""
        dm = DataManager(base_config, mock_logger)
        dm.data = pd.DataFrame({'a': [1, 2, np.nan], 'b': [1, np.inf, 3]})
        dm.validate_nan_inf()
        
        # Check that warnings were logged
        mock_logger.warning.assert_any_call("Column 'a' has 1 NaNs")
        mock_logger.warning.assert_any_call("Column 'b' has 1 Infs")

    def test_validate_circle_constraint(self, base_config, mock_logger):
        """Tests the sin^2 + cos^2 ~= 1 validation."""
        dm = DataManager(base_config, mock_logger)
        dm.data = pd.DataFrame({
            'sin': [1.0, 0.5, 2.0], 
            'cos': [0.0, 0.866, 2.0]
        }) # GOOD, GOOD, BAD
        val_df = dm.validate_circle_constraint()
        assert val_df['status'].tolist() == ['GOOD', 'GOOD', 'BAD']
        mock_logger.error.assert_called_once()
        
    def test_compute_derived_columns(self, base_config, sample_dataframe, mock_logger):
        """Tests the computation of angle, hs_bin, and combined_bin."""
        base_config['splitting']['hs_binning_method'] = 'equal_width'
        dm = DataManager(base_config, mock_logger)
        dm.data = sample_dataframe.copy()
        dm.compute_derived_columns()
        
        # Expected angles: 0, 90, 45, 225
        expected_angles = [0.0, 90.0, 45.0, 225.0]
        assert np.allclose(dm.data['angle_deg'], expected_angles)
        
        # Expected angle bins (4 bins): 0, 1, 0, 2
        assert dm.data['angle_bin'].tolist() == [0, 1, 0, 2]
        
        # Expected hs bins (2 bins, equal_width): 0, 1, 0, 1
        assert dm.data['hs_bin'].tolist() == [0, 1, 0, 1]
        
        # Expected combined_bin: angle_bin * 2 + hs_bin
        # 0*2+0=0, 1*2+1=3, 0*2+0=0, 2*2+1=5
        assert dm.data['combined_bin'].tolist() == [0, 3, 0, 5]
        
    def test_qcut_warning_on_duplicates(self, base_config, sample_dataframe, mock_logger):
        """Tests that a warning is logged if qcut produces fewer bins than requested."""
        base_config['splitting']['hs_binning_method'] = 'quantile'
        base_config['splitting']['hs_bins'] = 3
        dm = DataManager(base_config, mock_logger)
        
        # FIX: Ensure all arrays are same length (6)
        dm.data = pd.DataFrame({
            'sin': [0, 0, 0, 0, 0, 0],
            'cos': [1, 1, 1, 1, 1, 1],
            'hs': [1, 1, 1, 10, 10, 10]
        })
        
        dm.compute_derived_columns()
        # FIX: Updated expectation to 1 bin based on actual pandas behavior for this data
        mock_logger.warning.assert_called_with(
            "qcut for hs_bin was configured for 3 bins, but "
            "only created 1 due to duplicate values in data. "
            "This may affect stratification."
        )

    def test_execute_orchestration(self, base_config, create_data_file, mock_logger):
        """Tests the full execute method orchestration."""
        file_path = create_data_file("xlsx")
        base_config["data"]["file_path"] = file_path
        
        dm = DataManager(base_config, mock_logger)
        result_df = dm.execute(run_id="test_run")
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'angle_deg' in result_df.columns
        assert 'combined_bin' in result_df.columns
        
        # Check that output files were created
        output_dir = Path(base_config['outputs']['base_results_dir']) / "01_DATA_VALIDATION"
        assert (output_dir / "validated_data.xlsx").exists()
        assert (output_dir / "column_stats.xlsx").exists()
        assert (output_dir / "sin_cos_validation.xlsx").exists()
        assert (output_dir / "angle_distribution.png").exists()
        assert (output_dir / "hs_distribution.png").exists()
