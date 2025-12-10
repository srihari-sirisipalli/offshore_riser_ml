import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from unittest.mock import MagicMock
from modules.split_engine import SplitEngine

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    """Provides a mock logger for tests."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def base_config(tmp_path):
    """Provides a base configuration for the SplitEngine."""
    return {
        "data": {
            "hs_column": "hs"
        },
        "splitting": {
            "test_size": 0.2,
            "val_size": 0.2,
            "seed": 42,
            "drop_incomplete_bins": False
        },
        "outputs": {
            "base_results_dir": str(tmp_path)
        },
        "_internal_seeds": {
            "split": 42
        }
    }

@pytest.fixture
def sample_dataframe():
    """Provides a sample DataFrame with various bin distributions."""
    n = 100
    data = {
        'hs': np.random.rand(n) * 10,
        'angle_deg': np.random.rand(n) * 360,
        # 'combined_bin' with high cardinality
        'combined_bin': np.random.randint(0, 50, n),
        # 'hs_bin' with good distribution
        'hs_bin': np.random.randint(0, 4, n),
        # 'angle_bin' with good distribution
        'angle_bin': np.random.randint(0, 8, n),
        'other_feature': np.random.randn(n)
    }
    return pd.DataFrame(data)

# --- Test Cases ---

class TestSplitEngine:

    def test_stratification_strategy_combined(self, base_config, mock_logger, sample_dataframe):
        """Tests that 'combined_bin' is chosen when it's viable."""
        # Make 'combined_bin' viable by reducing its cardinality
        sample_dataframe['combined_bin'] = np.random.randint(0, 5, len(sample_dataframe))
        engine = SplitEngine(base_config, mock_logger)
        strategy = engine._determine_stratification_strategy(sample_dataframe)
        assert strategy == 'combined_bin'
        mock_logger.info.assert_any_call("Using 'combined_bin' for stratification.")

    def test_stratification_strategy_fallback_to_hs(self, base_config, mock_logger, sample_dataframe):
        """Tests fallback to 'hs_bin' when 'combined_bin' has too many singletons."""
        # 'combined_bin' has high cardinality by default in the fixture
        engine = SplitEngine(base_config, mock_logger)
        strategy = engine._determine_stratification_strategy(sample_dataframe)
        assert strategy == 'hs_bin'
        mock_logger.warning.assert_any_call("Too many rare combined bins. Falling back to 'hs_bin' stratification.")

    def test_stratification_strategy_fallback_to_angle(self, base_config, mock_logger, sample_dataframe):
        """Tests fallback to 'angle_bin' when both 'combined_bin' and 'hs_bin' are not viable."""
        # Make hs_bin not viable
        sample_dataframe['hs_bin'] = np.arange(len(sample_dataframe))
        engine = SplitEngine(base_config, mock_logger)
        strategy = engine._determine_stratification_strategy(sample_dataframe)
        assert strategy == 'angle_bin'
        mock_logger.warning.assert_any_call("Falling back to 'angle_bin' stratification.")
    
    def test_stratification_strategy_fallback_to_none(self, base_config, mock_logger, sample_dataframe):
        """Tests fallback to random split when no stratification key is viable."""
        # Make all potential strat columns non-viable
        sample_dataframe['combined_bin'] = np.arange(len(sample_dataframe))
        sample_dataframe['hs_bin'] = np.arange(len(sample_dataframe))
        sample_dataframe['angle_bin'] = np.arange(len(sample_dataframe))
        
        engine = SplitEngine(base_config, mock_logger)
        strategy = engine._determine_stratification_strategy(sample_dataframe)
        assert strategy is None
        mock_logger.warning.assert_any_call("Cannot stratify safely. Using random splitting.")

    def test_perform_split_sizes_and_no_overlap(self, base_config, mock_logger, sample_dataframe):
        """Tests that the split produces correctly sized, non-overlapping dataframes."""
        engine = SplitEngine(base_config, mock_logger)
        n = len(sample_dataframe)
        test_size = int(n * base_config['splitting']['test_size'])
        val_size = int(n * base_config['splitting']['val_size'])
        train_size = n - test_size - val_size
        
        train, val, test = engine._perform_split(sample_dataframe, strat_col='hs_bin')
        
        assert len(test) == test_size
        assert len(val) == val_size
        assert len(train) == train_size
        
        # Check for no overlap
        assert pd.concat([train, val, test]).index.nunique() == n
        
    def test_perform_split_with_rare_bins(self, base_config, mock_logger):
        """Tests that rare bins (count < 3) are handled and moved to the training set."""
        # Create a dataframe with a rare bin (class 'c')
        df = pd.DataFrame({
            'strat_col': ['a']*10 + ['b']*10 + ['c']*2,
            'feature': range(22)
        })
        base_config['splitting']['strat_col'] = 'strat_col'
        
        engine = SplitEngine(base_config, mock_logger)
        train, val, test = engine._perform_split(df, strat_col='strat_col')
        
        # The 2 samples from class 'c' should be in the training set
        assert len(train[train['strat_col'] == 'c']) == 2
        assert len(val[val['strat_col'] == 'c']) == 0
        assert len(test[test['strat_col'] == 'c']) == 0
        mock_logger.warning.assert_called_with("2 samples belong to bins with < 3 members. They will be put into Train set automatically.")

    def test_perform_split_with_drop_incomplete_bins(self, base_config, mock_logger):
        """Tests that rare bins are dropped when the config flag is True."""
        df = pd.DataFrame({
            'strat_col': ['a']*10 + ['b']*10 + ['c']*2,
            'feature': range(22)
        })
        base_config['splitting']['strat_col'] = 'strat_col'
        base_config['splitting']['drop_incomplete_bins'] = True # Enable dropping
        
        engine = SplitEngine(base_config, mock_logger)
        train, val, test = engine._perform_split(df, strat_col='strat_col')
        
        # The total number of samples should be 20 (2 dropped)
        assert len(train) + len(val) + len(test) == 20
        mock_logger.warning.assert_called_with("2 samples belong to incomplete bins and will be DROPPED.")
        
    def test_execute_creates_files(self, base_config, mock_logger, sample_dataframe, tmp_path):
        """Tests the full execute method to ensure it creates the expected output files."""
        engine = SplitEngine(base_config, mock_logger)
        engine.execute(sample_dataframe, run_id="test_run")
        
        output_dir = tmp_path / "02_SMART_SPLIT"
        assert (output_dir / "train.xlsx").exists()
        assert (output_dir / "val.xlsx").exists()
        assert (output_dir / "test.xlsx").exists()
        assert (output_dir / "split_balance_report.xlsx").exists()
        assert (output_dir / "split_hs_dist.png").exists()
