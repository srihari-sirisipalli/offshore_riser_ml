import pytest
import pandas as pd
import shutil
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.rfe.rfe_controller import RFEController

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def rfe_config(tmp_path):
    return {
        'outputs': {'base_results_dir': str(tmp_path)},
        'iterative': {
            'enabled': True,
            'min_features': 2,
            'max_rounds': 5
        },
        'data': {
            'target_sin': 'sin',
            'target_cos': 'cos',
            'hs_column': 'hs',
            'drop_columns': ['ignore_me']
        }
    }

@pytest.fixture
def sample_data():
    # 5 features, 10 rows
    df = pd.DataFrame({
        'f1': range(10), 'f2': range(10), 'f3': range(10), 'f4': range(10), 'f5': range(10),
        'sin': [0]*10, 'cos': [1]*10, 'hs': [1.5]*10, 'ignore_me': [0]*10,
        'row_index': range(10)
    })
    return df

# --- Tests ---

class TestRFEController:

    def test_initialization(self, rfe_config, mock_logger):
        controller = RFEController(rfe_config, mock_logger)
        assert controller.min_features == 2
        assert controller.current_round == 0
        assert controller.active_features == []

    def test_feature_initialization(self, rfe_config, mock_logger, sample_data):
        controller = RFEController(rfe_config, mock_logger)
        feats = controller._initialize_features(sample_data)
        
        # Expected: f1, f2, f3, f4, f5 (5 features)
        # Excluded: sin, cos, hs, ignore_me, row_index
        assert len(feats) == 5
        assert 'f1' in feats
        assert 'sin' not in feats
        assert 'ignore_me' not in feats

    @patch('modules.rfe.rfe_controller.TrainingEngine')
    @patch('modules.rfe.rfe_controller.HPOSearchEngine')
    def test_full_loop_execution_logic(self, MockHPO, MockTrain, rfe_config, mock_logger, sample_data):
        """
        Verifies the loop runs, creates folders, and stops when min_features is reached.
        """
        controller = RFEController(rfe_config, mock_logger)

        # Mock sub-methods to avoid complex logic dependencies
        controller._execute_grid_search_phase = MagicMock(return_value={'model': 'LinearRegression', 'params': {}})
        # _train_baseline_phase returns (metrics_dict, predictions_df)
        # Create complete mock predictions with all required columns
        mock_predictions = pd.DataFrame({
            'row_index': [0, 1, 2],
            'true_sin': [0.1, 0.2, 0.3],
            'true_cos': [0.9, 0.8, 0.7],
            'pred_sin': [0.11, 0.21, 0.31],
            'pred_cos': [0.89, 0.79, 0.69],
            'true_angle': [6.3, 14.0, 23.2],
            'pred_angle': [7.1, 14.7, 24.0],
            'error': [0.8, 0.7, 0.8],
            'abs_error': [0.8, 0.7, 0.8]
        })
        controller._train_baseline_phase = MagicMock(return_value=({'cmae': 1.0}, mock_predictions))
        # Mock internal training to avoid ModelFactory calls
        mock_model = MagicMock()
        controller._train_model_internal = MagicMock(return_value=mock_model)
        controller._make_predictions = MagicMock(return_value=mock_predictions)
        
        # Scenario: Start with 5 features. Min is 2.
        # R0 (5 feats) -> drops f1. Rem: 4. Stop? No.
        # R1 (4 feats) -> drops f2. Rem: 3. Stop? No.
        # R2 (3 feats) -> drops f3. Rem: 2. Stop? YES (because 3-1 = 2 <= min).
        
        controller._execute_lofo_phase = MagicMock(side_effect=[
            [{'feature': 'f1', 'val_cmae': 0.1, 'delta_cmae': 0.05, 'metrics': {}, 'val_predictions': mock_predictions},
             {'feature': 'f2', 'val_cmae': 0.2, 'delta_cmae': 0.10, 'metrics': {}, 'val_predictions': mock_predictions}], # Round 0
            [{'feature': 'f2', 'val_cmae': 0.1, 'delta_cmae': 0.05, 'metrics': {}, 'val_predictions': mock_predictions},
             {'feature': 'f3', 'val_cmae': 0.2, 'delta_cmae': 0.10, 'metrics': {}, 'val_predictions': mock_predictions}], # Round 1
            [{'feature': 'f3', 'val_cmae': 0.1, 'delta_cmae': 0.05, 'metrics': {}, 'val_predictions': mock_predictions}], # Round 2
        ])
        
        # Execute
        controller.run(sample_data, sample_data, sample_data)
        
        # Checks
        base_dir = Path(rfe_config['outputs']['base_results_dir'])
        
        # 1. Directory Creation
        assert (base_dir / "ROUND_000").exists()
        assert (base_dir / "ROUND_001").exists()
        assert (base_dir / "ROUND_002").exists()
        # Round 3 should NOT start
        assert not (base_dir / "ROUND_003").exists()
        
        # 2. Flag Creation
        assert (base_dir / "ROUND_000" / "_ROUND_COMPLETE.flag").exists()
        
        # 3. State Check
        # Should finish loop after Round 2
        assert controller.current_round == 2
        
        # Check logs for stop message
        mock_logger.info.assert_any_call("Stopping Criteria Met: Minimum Feature Count Reached")

    def test_resume_logic(self, rfe_config, mock_logger, sample_data):
        """Tests that the controller skips completed rounds."""
        base_dir = Path(rfe_config['outputs']['base_results_dir'])
        
        # Manually create Round 0 artifact
        round0 = base_dir / "ROUND_000"
        (round0 / "00_DATASETS").mkdir(parents=True)
        
        # Create completion flag
        with open(round0 / "_ROUND_COMPLETE.flag", 'w') as f:
            json.dump({'round': 0, 'dropped_feature': 'f5'}, f)
            
        # Create feature list for Round 0 (start state)
        with open(round0 / "00_DATASETS" / "feature_list.json", 'w') as f:
            json.dump(['f1', 'f2', 'f3', 'f4', 'f5'], f)
            
        # Run Controller
        controller = RFEController(rfe_config, mock_logger)
        
        # Mock internal methods to prevent real processing AND IO errors
        controller._setup_round_directory = MagicMock()
        controller._save_round_datasets = MagicMock()
        controller._execute_grid_search_phase = MagicMock(return_value={'model': 'LinearRegression', 'params': {}})
        # _train_baseline_phase returns (metrics_dict, predictions_df)
        # Create complete mock predictions with all required columns
        mock_predictions = pd.DataFrame({
            'row_index': [0, 1, 2],
            'true_sin': [0.1, 0.2, 0.3],
            'true_cos': [0.9, 0.8, 0.7],
            'pred_sin': [0.11, 0.21, 0.31],
            'pred_cos': [0.89, 0.79, 0.69],
            'true_angle': [6.3, 14.0, 23.2],
            'pred_angle': [7.1, 14.7, 24.0],
            'error': [0.8, 0.7, 0.8],
            'abs_error': [0.8, 0.7, 0.8]
        })
        controller._train_baseline_phase = MagicMock(return_value=({'val_cmae': 1.0}, mock_predictions))
        controller._execute_lofo_phase = MagicMock(return_value=[
            {'feature':'f1', 'val_cmae': 0.9, 'delta_cmae': -0.1, 'metrics': {'val_cmae': 0.9}, 'val_predictions': mock_predictions}
        ])
        # Mock training and prediction methods
        mock_model = MagicMock()
        controller._train_model_internal = MagicMock(return_value=mock_model)
        controller._make_predictions = MagicMock(return_value=mock_predictions)
        
        # FIX: Mock _finalize_round so it doesn't try to write to non-existent folders 
        # (since _setup_round_directory is mocked out)
        controller._finalize_round = MagicMock()
        
        controller.run(sample_data, sample_data, sample_data)
        
        # Controller should have detected Round 0 is done
        # active_features should have 'f5' removed
        assert 'f5' not in controller.active_features
        # Current round should be > 0
        assert controller.current_round > 0