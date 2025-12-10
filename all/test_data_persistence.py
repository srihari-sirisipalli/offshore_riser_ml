import pytest
import pandas as pd
import numpy as np
import shutil
import json
from pathlib import Path
from unittest.mock import MagicMock
from modules.data_manager import DataPersistence

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def persistence_config(tmp_path):
    return {
        'outputs': {'base_results_dir': str(tmp_path)}
    }

@pytest.fixture
def sample_df():
    # 10 rows, 3 features, 2 targets
    data = {
        'f1': np.random.rand(10).astype(np.float32),
        'f2': np.random.rand(10).astype(np.float32),
        'f3': np.random.rand(10).astype(np.float32),
        'sin': np.zeros(10, dtype=np.float32),
        'cos': np.ones(10, dtype=np.float32)
    }
    return pd.DataFrame(data)

# --- Tests ---

class TestDataPersistence:

    def test_persist_and_load_integrity(self, persistence_config, mock_logger, sample_df):
        """Verify data saved matches data loaded."""
        dp = DataPersistence(persistence_config, mock_logger)
        
        features = ['f1', 'f2', 'f3']
        targets = ['sin', 'cos']
        
        # 1. Persist
        meta = dp.persist_as_memmap(sample_df, "train", features, targets)
        
        assert Path(meta['x_path']).exists()
        assert Path(meta['y_path']).exists()
        assert meta['x_shape'] == [10, 3] # List because JSON conversion
        assert meta['y_shape'] == [10, 2]
        
        # 2. Load
        X_mm, y_mm = dp.load_memmap(meta, mode='r')
        
        # 3. Check Properties
        assert isinstance(X_mm, np.memmap)
        assert X_mm.shape == (10, 3)
        assert X_mm.dtype == 'float32'
        
        # 4. Check Values
        original_X = sample_df[features].values.astype(np.float32)
        np.testing.assert_array_almost_equal(X_mm, original_X)
        
        # 5. Check Read-Only Enforcement
        # Try to modify read-only memmap -> should raise error
        with pytest.raises(ValueError):
            X_mm[0, 0] = 999.0

    def test_cleanup(self, persistence_config, mock_logger):
        """Verify cleanup deletes the cache folder."""
        dp = DataPersistence(persistence_config, mock_logger)
        
        # Create a dummy file to simulate usage
        dummy_file = dp.storage_dir / "test.dat"
        dummy_file.touch()
        
        assert dp.storage_dir.exists()
        
        dp.cleanup()
        
        assert not dp.storage_dir.exists()

    def test_missing_file_error(self, persistence_config, mock_logger):
        """Verify error raised if metadata points to non-existent file."""
        dp = DataPersistence(persistence_config, mock_logger)
        
        fake_meta = {
            'split_name': 'test',
            'x_path': 'non_existent_X.dat',
            'y_path': 'non_existent_y.dat',
            'x_shape': (10, 1),
            'y_shape': (10, 1),
            'dtype': 'float32'
        }
        
        with pytest.raises(FileNotFoundError):
            dp.load_memmap(fake_meta)