import pytest
import pandas as pd
import shutil
import os
from pathlib import Path
from unittest.mock import MagicMock
from modules.global_error_tracking.evolution_tracker import EvolutionTracker

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def track_config(tmp_path):
    return {
        'outputs': {'base_results_dir': str(tmp_path)}
    }

# --- Tests ---

class TestEvolutionTracker:

    def test_directory_initialization(self, track_config, mock_logger, tmp_path):
        """Verifies that the 01_GLOBAL_TRACKING structure is created."""
        tracker = EvolutionTracker(track_config, mock_logger)
        
        base = Path(track_config['outputs']['base_results_dir']) / "01_GLOBAL_TRACKING"
        assert (base / "01_metrics").exists()
        assert (base / "02_features").exists()
        assert (base / "04_evolution_plots").exists()

    def test_metrics_update_flow(self, track_config, mock_logger):
        """Verifies adding rounds creates and updates the Excel file."""
        tracker = EvolutionTracker(track_config, mock_logger)
        
        # Round 0
        r0 = {
            'round': 0, 'n_features': 100, 'dropped_feature': 'f1',
            'metrics': {'val_cmae': 2.0, 'val_accuracy_5deg': 80}
        }
        tracker.update_tracker(r0)
        
        # Round 1
        r1 = {
            'round': 1, 'n_features': 99, 'dropped_feature': 'f2',
            'metrics': {'val_cmae': 1.8, 'val_accuracy_5deg': 82}
        }
        tracker.update_tracker(r1)
        
        # Check File
        file_path = tracker.tracking_dir / "01_metrics" / "metrics_all_rounds.xlsx"
        assert file_path.exists()
        
        df = pd.read_excel(file_path)
        assert len(df) == 2
        assert df.iloc[0]['cmae'] == 2.0
        assert df.iloc[1]['cmae'] == 1.8
        assert df.iloc[1]['feature_removed'] == 'f2'

    def test_resume_handling_no_duplicates(self, track_config, mock_logger):
        """Verifies that re-running the same round update doesn't duplicate rows."""
        tracker = EvolutionTracker(track_config, mock_logger)
        
        r0 = {
            'round': 0, 'n_features': 100, 'dropped_feature': 'f1',
            'metrics': {'val_cmae': 2.0}
        }
        
        tracker.update_tracker(r0)
        tracker.update_tracker(r0) # Duplicate call
        
        file_path = tracker.tracking_dir / "01_metrics" / "metrics_all_rounds.xlsx"
        df = pd.read_excel(file_path)
        
        assert len(df) == 1 # Still 1 row

    def test_plot_generation(self, track_config, mock_logger):
        """Verifies that plots are generated after 2+ rounds."""
        tracker = EvolutionTracker(track_config, mock_logger)
        
        # FIX: Include all metrics required for plotting (cmae AND accuracy_at_5deg)
        r0 = {
            'round': 0, 'n_features': 10, 
            'metrics': {'cmae': 1.0, 'accuracy_at_5deg': 80.0}, 
            'hyperparameters': {}
        }
        r1 = {
            'round': 1, 'n_features': 9, 
            'metrics': {'cmae': 0.9, 'accuracy_at_5deg': 85.0}, 
            'hyperparameters': {}
        }
        
        tracker.update_tracker(r0)
        tracker.update_tracker(r1)
        
        plot_dir = tracker.tracking_dir / "04_evolution_plots"
        
        assert (plot_dir / "cmae_evolution.png").exists()
        # This will now exist because we provided the data
        assert (plot_dir / "accuracy5_evolution.png").exists() 
        assert (plot_dir / "feature_count_vs_error.png").exists()