import pytest
import pandas as pd
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from modules.reporting_engine import ReconstructionMapper

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def report_config(tmp_path):
    return {
        'data': {
            'target_sin': 'target_sin',
            'target_cos': 'target_cos'
        }
    }

@pytest.fixture
def mock_history():
    return [
        {
            'round': 0,
            'n_features': 10,
            'dropped_feature': 'feat_X',
            'metrics': {'cmae': 1.5, 'accuracy_at_5deg': 80.0},
            'hyperparameters': {'n_estimators': 100, 'max_depth': 10, 'model_name': 'ExtraTreesRegressor'},
            'active_features_list': ['f1', 'f2', 'f3']
        },
        {
            'round': 1,
            'n_features': 9,
            'dropped_feature': 'feat_Y',
            'metrics': {'cmae': 1.4, 'accuracy_at_5deg': 82.0},
            'hyperparameters': {'n_estimators': 150, 'max_depth': 12, 'model_name': 'ExtraTreesRegressor'},
            'active_features_list': ['f1', 'f3']
        }
    ]

@pytest.fixture
def mock_data_info():
    return {
        'train_path': 'data/train_v1.xlsx',
        'val_path': 'data/val_v1.xlsx',
        'input_hash': 'abc123hash'
    }

# --- Tests ---

class TestReconstructionMapper:

    def test_excel_generation(self, report_config, mock_logger, mock_history, mock_data_info, tmp_path):
        """Verifies the Excel file is created with all sheets."""
        mapper = ReconstructionMapper(report_config, mock_logger)
        
        mapper.generate_mapping(mock_history, mock_data_info, tmp_path)
        
        output_file = tmp_path / "model_reconstruction_mapping.xlsx"
        assert output_file.exists()
        
        # Verify Sheet Names
        xl = pd.ExcelFile(output_file)
        assert 'Rounds_Summary' in xl.sheet_names
        assert 'Hyperparameters' in xl.sheet_names
        assert 'Recreation_Code' in xl.sheet_names

    def test_code_generation_content(self, report_config, mock_logger, mock_history, mock_data_info):
        """Verifies the Python code snippet contains the right variables."""
        mapper = ReconstructionMapper(report_config, mock_logger)
        
        round_0 = mock_history[0]
        code = mapper._generate_python_snippet(round_0, mock_data_info)
        
        # Check for key components
        assert "import pandas" in code
        assert "ExtraTreesRegressor" in code
        assert "n_estimators=100" in code
        assert "max_depth=10" in code
        assert "random_state=456" in code
        assert "target_sin" in code # From config fixture
        assert "['f1', 'f2', 'f3']" in code

    def test_summary_data_correctness(self, report_config, mock_logger, mock_history, mock_data_info, tmp_path):
        """Verifies the data inside the summary sheet matches input history."""
        mapper = ReconstructionMapper(report_config, mock_logger)
        mapper.generate_mapping(mock_history, mock_data_info, tmp_path)
        
        df_summary = pd.read_excel(tmp_path / "model_reconstruction_mapping.xlsx", sheet_name='Rounds_Summary')
        
        assert len(df_summary) == 2
        row0 = df_summary.iloc[0]
        assert row0['Round'] == 0
        assert row0['Dropped_Feature'] == 'feat_X'
        assert row0['Val_CMAE'] == 1.5