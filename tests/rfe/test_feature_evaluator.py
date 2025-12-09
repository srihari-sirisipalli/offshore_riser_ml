import pytest
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.rfe.feature_evaluator import FeatureEvaluator

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def base_config(tmp_path):
    return {
        'outputs': {'base_results_dir': str(tmp_path)},
        'data': {
            'target_sin': 'sin',
            'target_cos': 'cos',
            'hs_column': 'hs',
            'drop_columns': []
        },
        'models': {'native': ['ExtraTreesRegressor']}
    }

@pytest.fixture
def sample_data():
    # 3 features, 5 rows
    df = pd.DataFrame({
        'f1': np.random.rand(5),
        'f2': np.random.rand(5),
        'f3': np.random.rand(5),
        'sin': [0, 1, 0, -1, 0],
        'cos': [1, 0, -1, 0, 1]
    })
    return df

@pytest.fixture
def mock_hyperparams():
    return {'n_estimators': 10}

# --- Tests ---

class TestFeatureEvaluator:

    def test_subset_logic(self, base_config, mock_logger, sample_data):
        """Tests that the dataframe is correctly subsetted to specific features + targets."""
        evaluator = FeatureEvaluator(base_config, mock_logger)
        
        features = ['f1', 'f3']
        subset = evaluator._subset_df(sample_data, features)
        
        assert 'f1' in subset.columns
        assert 'f3' in subset.columns
        assert 'f2' not in subset.columns
        assert 'sin' in subset.columns # Targets must remain
        assert 'cos' in subset.columns

    @patch('modules.rfe.feature_evaluator.TrainingEngine')
    @patch('modules.rfe.feature_evaluator.EvaluationEngine')
    def test_lofo_loop_execution(self, MockEval, MockTrain, base_config, mock_logger, sample_data, mock_hyperparams):
        """
        Verifies that the loop runs for every feature and calculates deltas.
        Mocks the actual training/eval to return predictable numbers.
        """
        evaluator = FeatureEvaluator(base_config, mock_logger)
        
        # Setup Mocks
        mock_train_instance = MockTrain.return_value
        mock_model = MagicMock()
        # Predict dummy (n_samples, 2)
        mock_model.predict.return_value = np.zeros((len(sample_data), 2)) 
        mock_train_instance.train.return_value = mock_model
        
        mock_eval_instance = MockEval.return_value
        
        # Side effects for compute_metrics:
        # Call 1: Baseline (Baseline CMAE = 1.0)
        # Call 2: Drop f1 (CMAE = 1.2 -> Worsened, Delta +0.2)
        # Call 3: Drop f2 (CMAE = 0.8 -> Improved, Delta -0.2)
        # Call 4: Drop f3 (CMAE = 1.0 -> Neutral, Delta 0.0)
        mock_eval_instance.compute_metrics.side_effect = [
            {'cmae': 1.0, 'crmse': 1.0, 'accuracy_at_5deg': 90}, # Baseline
            {'cmae': 1.2, 'crmse': 1.2, 'accuracy_at_5deg': 80}, # Drop f1
            {'cmae': 0.8, 'crmse': 0.8, 'accuracy_at_5deg': 95}, # Drop f2
            {'cmae': 1.0, 'crmse': 1.0, 'accuracy_at_5deg': 90}, # Drop f3
        ]
        
        # Prepare output dir
        round_dir = Path(base_config['outputs']['base_results_dir']) / "ROUND_000"
        
        # EXECUTE
        active_features = ['f1', 'f2', 'f3']
        results = evaluator.evaluate_features(round_dir, active_features, mock_hyperparams, sample_data, sample_data)
        
        # VERIFY
        assert len(results) == 3
        
        # Check f1 (worsened performance)
        res_f1 = next(r for r in results if r['feature'] == 'f1')
        assert np.isclose(res_f1['delta_cmae'], 0.2)
        
        # Check f2 (improved performance -> best candidate to drop)
        res_f2 = next(r for r in results if r['feature'] == 'f2')
        assert np.isclose(res_f2['delta_cmae'], -0.2)
        
        # Check file output
        excel_path = round_dir / "04_FEATURE_EVALUATION" / "feature_impact_all_features.xlsx"
        assert excel_path.exists()
        
        df_results = pd.read_excel(excel_path)
        # Rank 1 should be f2 (lowest CMAE)
        assert df_results.iloc[0]['feature'] == 'f2'
        assert df_results.iloc[0]['rank'] == 1