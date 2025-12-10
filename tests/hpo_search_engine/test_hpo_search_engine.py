import pytest
import pandas as pd
import numpy as np
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.hpo_search_engine import HPOSearchEngine
import logging

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def sample_dfs():
    n = 20
    df = pd.DataFrame({
        'feat1': np.random.rand(n),
        'feat2': np.random.rand(n),
        'target_sin': np.sin(range(n)),
        'target_cos': np.cos(range(n)),
        'combined_bin': [0, 1] * 10
    })
    return df, df.copy(), df.copy()

@pytest.fixture
def hpo_config(tmp_path):
    return {
        'data': {
            'target_sin': 'target_sin', 'target_cos': 'target_cos',
            'drop_columns': []
        },
        'outputs': {'base_results_dir': str(tmp_path)},
        'execution': {'n_jobs': 1},
        'hyperparameters': {
            'enabled': True,
            'cv_folds': 2,
            'grids': {
                'ExtraTreesRegressor': {'n_estimators': [5]}
            }
        },
        'resources': {'max_hpo_configs': 10}
    }

# --- Tests ---

class TestHPOSearchEngine:

    def test_snapshot_parquet_generation(self, hpo_config, sample_dfs, mock_logger):
        """Test that snapshots are saved as Parquet."""
        train, val, test = sample_dfs
        engine = HPOSearchEngine(hpo_config, mock_logger)
        
        # Run execution
        engine.execute(train, val, test, "test_run")

        # Check output
        base = Path(hpo_config['outputs']['base_results_dir'])
        snap_dir = base / "03_HYPERPARAMETER_OPTIMIZATION" / "tracking_snapshots"

        files = list(snap_dir.glob("*.parquet"))
        assert len(files) > 0, "No Parquet snapshots created"

        # Verify content
        df = pd.read_parquet(files[0])
        assert 'pred_angle' in df.columns
        assert 'abs_error' in df.columns

    def test_memory_cleanup_called(self, hpo_config, sample_dfs, mock_logger):
        """Verify garbage collection is triggered."""
        with patch('gc.collect') as mock_gc:
            engine = HPOSearchEngine(hpo_config, mock_logger)
            engine.execute(*sample_dfs, "test_run")
            assert mock_gc.call_count >= 1

    def test_resume_logic_reads_partial(self, hpo_config, mock_logger, tmp_path):
        """Test that resume reads existing progress."""
        progress_path = tmp_path / "03_HYPERPARAMETER_OPTIMIZATION" / "progress" / "hpo_progress.jsonl"
        progress_path.parent.mkdir(parents=True)
        
        # Write fake entry
        fake_entry = {'config_hash': 'abc123hash', 'status': 'success'}
        with open(progress_path, 'w') as f:
            f.write(json.dumps(fake_entry) + '\n')
            
        engine = HPOSearchEngine(hpo_config, mock_logger)
        # _load_progress is called in execute, so we call it manually to test init logic
        # Ideally, we mock execute to just check init state
        engine.progress_file = progress_path
        engine._load_progress()
        
        assert 'abc123hash' in engine.completed_hashes