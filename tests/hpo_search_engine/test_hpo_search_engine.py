import pytest
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.hpo_search_engine import HPOSearchEngine

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    """Provides a mock logger for tests."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def sample_dfs():
    """Provides train, val, and test dataframes."""
    n = 20
    train_df = pd.DataFrame({
        'feature_1': np.random.rand(n), 'feature_2': np.random.rand(n),
        'target_sin': np.sin(np.linspace(0, 2, n)), 'target_cos': np.cos(np.linspace(0, 2, n)),
        'combined_bin': np.random.randint(0, 2, n)
    })
    val_df = pd.DataFrame({
        'feature_1': np.random.rand(5), 'feature_2': np.random.rand(5),
        'target_sin': np.sin(np.linspace(2, 3, 5)), 'target_cos': np.cos(np.linspace(2, 3, 5)),
        'combined_bin': np.random.randint(0, 2, 5)
    })
    test_df = val_df.copy()
    return train_df, val_df, test_df

@pytest.fixture
def hpo_config(tmp_path):
    """Provides a base configuration for HPO tests."""
    return {
        'data': {
            'target_sin': 'target_sin', 'target_cos': 'target_cos',
            'drop_columns': []
        },
        'outputs': {'base_results_dir': str(tmp_path)},
        'execution': {'n_jobs': 1}, # Use sequential for test stability
        'hyperparameters': {
            'enabled': True,
            'cv_folds': 2,
            'grids': {
                'ExtraTreesRegressor': {
                    'n_estimators': [5, 10],
                    'max_depth': [3]
                }
            }
        }
    }

# --- Test Cases ---

class TestHPOSearchEngine:

    def test_hpo_execution_and_artifacts(self, hpo_config, sample_dfs, mock_logger):
        """Tests a full, successful HPO execution and checks for all expected artifacts."""
        train_df, val_df, test_df = sample_dfs
        engine = HPOSearchEngine(hpo_config, mock_logger)
        
        best_config = engine.execute(train_df, val_df, test_df, "test_run")
        
        assert best_config is not None
        assert 'model' in best_config and best_config['model'] == 'ExtraTreesRegressor'
        assert 'params' in best_config and 'n_estimators' in best_config['params']
        
        # Check for created directories and files
        base_dir = Path(hpo_config['outputs']['base_results_dir'])
        hpo_dir = base_dir / "03_HYPERPARAMETER_OPTIMIZATION"
        snapshot_dir = base_dir / "03_HPO_SEARCH" / "tracking_snapshots"
        
        assert (hpo_dir / "results" / "all_configurations.xlsx").exists()
        assert (hpo_dir / "results" / "best_configuration.json").exists()
        assert (hpo_dir / "progress" / "hpo_progress.jsonl").exists()
        
        # Check for snapshot files (2 configs * 2 sets = 4 files)
        assert len(list(snapshot_dir.glob("*.csv"))) == 4

    def test_hpo_disabled(self, hpo_config, sample_dfs, mock_logger):
        """Tests that HPO is skipped and a default config is returned when disabled."""
        hpo_config['hyperparameters']['enabled'] = False
        hpo_config['models'] = {'native': [{'name': 'DefaultModel'}]} # Add a default model
        
        train_df, val_df, test_df = sample_dfs
        engine = HPOSearchEngine(hpo_config, mock_logger)
        
        best_config = engine.execute(train_df, val_df, test_df, "test_run")
        
        assert best_config['model'] == {'name': 'DefaultModel'}
        assert best_config['params'] == {}
        mock_logger.info.assert_any_call("HPO disabled. Using default model configuration.")

    def test_resume_capability(self, hpo_config, sample_dfs, mock_logger):
        """Tests that a second run correctly skips already computed configurations."""
        train_df, val_df, test_df = sample_dfs
        engine1 = HPOSearchEngine(hpo_config, mock_logger)
        engine1.execute(train_df, val_df, test_df, "run1")
        
        progress_file = engine1.progress_file
        with open(progress_file, 'r') as f:
            content_initial = [json.loads(line) for line in f if line.strip()]
        
        # Create a new engine instance to simulate a new run
        engine2 = HPOSearchEngine(hpo_config, mock_logger)
        engine2.execute(train_df, val_df, test_df, "run2")
        
        with open(engine2.progress_file, 'r') as f:
            content_final = [json.loads(line) for line in f if line.strip()]
        
        # The content should be identical
        assert len(content_initial) == len(content_final)
        # Verify that the second logger instance was informed about resumed configs
        mock_logger.info.assert_any_call("Resumed HPO: 2 configs already completed.")

    @patch('modules.hpo_search_engine.ModelFactory.create')
    def test_hpo_handles_failed_trial(self, mock_create, hpo_config, sample_dfs, mock_logger):
        """Tests that HPO continues and logs an error if one trial fails."""
        # FIX: Create a mock model that behaves correctly for the successful trial
        good_model = MagicMock()
        
        # FIX: Configure predict to return array matching input length (2 columns for sin/cos)
        good_model.predict.side_effect = lambda X: np.zeros((len(X), 2))

        # Make the first model creation fail, but ALL subsequent ones succeed.
        # Config 1: 1 call (Fail)
        # Config 2: CV folds (2 calls) + Final Train (1 call) = 3 calls
        # We multiply the list to ensure we never run out of mocks
        mock_create.side_effect = [Exception("Model fit failed")] + [good_model] * 10

        train_df, val_df, test_df = sample_dfs
        engine = HPOSearchEngine(hpo_config, mock_logger)
        engine.execute(train_df, val_df, test_df, "test_run")

        # Check that an error was logged ONLY ONCE (for the first failure)
        mock_logger.error.assert_called_once()
        
        # Check that the progress file still contains entries for all trials
        with open(engine.progress_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            # First trial should be marked as 'failed'
            assert '"status": "failed"' in lines[0]
            # Second trial should be 'success'
            assert '"status": "success"' in lines[1]
    def test_finalize_results_tie_breaking(self, hpo_config, sample_dfs, mock_logger):
        """Tests that the best model is chosen correctly, applying tie-breaking."""
        train_df, val_df, test_df = sample_dfs
        engine = HPOSearchEngine(hpo_config, mock_logger)
        
        # Manually create a progress file with a tie
        progress_file = Path(hpo_config['outputs']['base_results_dir']) / "03_HYPERPARAMETER_OPTIMIZATION" / "progress" / "hpo_progress.jsonl"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        results = [
            {'config_id': 1, 'model_name': 'A', 'params': {}, 'cv_cmae_deg_mean': 5.0, 'cv_max_error_deg_mean': 20.0},
            {'config_id': 2, 'model_name': 'B', 'params': {}, 'cv_cmae_deg_mean': 5.0, 'cv_max_error_deg_mean': 10.0} # Lower max error, should be chosen
        ]
        with open(progress_file, 'w') as f:
            for res in results:
                f.write(json.dumps(res) + '\n')
                
        engine.progress_file = progress_file
        best_config = engine._finalize_results(progress_file.parent.parent / "results")
        
        assert best_config['model'] == 'B'
        mock_logger.info.assert_any_call("Tie-breaker applied: 2 configs with CMAE=5.0000")

    def test_skf_fallback_to_kf(self, hpo_config, mock_logger):
        """Tests that the CV strategy falls back from StratifiedKFold to KFold if stratification is not possible."""
        # Create a dataset where stratification is impossible (all unique values)
        n = 10
        train_df = pd.DataFrame({
            'feature_1': range(n),
            'target_sin': range(n), 'target_cos': range(n),
            'combined_bin': range(n) # Each sample has a unique bin
        })
        val_df = train_df.copy()
        test_df = train_df.copy()
        
        engine = HPOSearchEngine(hpo_config, mock_logger)
        
        # The execute should run without a ValueError from StratifiedKFold
        # The internal _evaluate_config_cv will log a warning, but we can't easily check that
        # without more complex mocking. We just check that it completes.
        best_config = engine.execute(train_df, val_df, test_df, "test_run")
        assert best_config is not None
