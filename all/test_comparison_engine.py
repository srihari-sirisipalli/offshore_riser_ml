import pytest
import pandas as pd
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from modules.rfe.comparison_engine import ComparisonEngine

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def comp_config(tmp_path):
    return {
        'outputs': {'base_results_dir': str(tmp_path)}
    }

@pytest.fixture
def sample_metrics():
    # Baseline: Average Error 1.0, Acc 90%
    base = {
        'cmae': 1.0,
        'crmse': 2.0,
        'accuracy_at_5deg': 90.0,
        'max_error': 10.0
    }
    return base

# --- Tests ---

class TestComparisonEngine:

    def test_improvement_logic(self, comp_config, mock_logger, sample_metrics):
        """
        Scenario: Dropping the feature IMPROVES the model.
        Errors should go DOWN (Negative Delta).
        Accuracy should go UP (Positive Delta).
        """
        engine = ComparisonEngine(comp_config, mock_logger)
        round_dir = Path(comp_config['outputs']['base_results_dir']) / "ROUND_000"
        
        # New model is BETTER
        dropped_metrics = {
            'cmae': 0.8,  # -0.2 (Improved)
            'crmse': 1.8, # -0.2 (Improved)
            'accuracy_at_5deg': 92.0, # +2.0 (Improved)
            'max_error': 9.0
        }
        
        engine.compare(round_dir, sample_metrics, dropped_metrics, "feat_bad")
        
        # Check output file
        df = pd.read_excel(round_dir / "06_COMPARISON" / "delta_metrics.xlsx")
        
        # Check CMAE
        row_cmae = df[df['metric'] == 'cmae'].iloc[0]
        assert row_cmae['delta'] == -0.2
        assert row_cmae['status'] == "IMPROVEMENT"
        
        # Check Accuracy
        row_acc = df[df['metric'] == 'accuracy_at_5deg'].iloc[0]
        assert row_acc['delta'] == 2.0
        assert row_acc['status'] == "IMPROVEMENT"

    def test_degradation_logic(self, comp_config, mock_logger, sample_metrics):
        """
        Scenario: Dropping the feature HURTS the model.
        Errors go UP (Positive Delta).
        Accuracy goes DOWN (Negative Delta).
        """
        engine = ComparisonEngine(comp_config, mock_logger)
        round_dir = Path(comp_config['outputs']['base_results_dir']) / "ROUND_001"
        
        # New model is WORSE
        dropped_metrics = {
            'cmae': 1.5,  # +0.5 (Degraded)
            'crmse': 2.5, 
            'accuracy_at_5deg': 85.0, # -5.0 (Degraded)
            'max_error': 15.0
        }
        
        engine.compare(round_dir, sample_metrics, dropped_metrics, "feat_good")
        
        df = pd.read_excel(round_dir / "06_COMPARISON" / "delta_metrics.xlsx")
        
        # Check CMAE
        row_cmae = df[df['metric'] == 'cmae'].iloc[0]
        assert row_cmae['delta'] == 0.5
        assert row_cmae['status'] == "DEGRADATION"
        
        # Check Accuracy
        row_acc = df[df['metric'] == 'accuracy_at_5deg'].iloc[0]
        assert row_acc['delta'] == -5.0
        assert row_acc['status'] == "DEGRADATION"

    def test_text_summary_creation(self, comp_config, mock_logger, sample_metrics):
        """Verifies the text report is generated."""
        engine = ComparisonEngine(comp_config, mock_logger)
        round_dir = Path(comp_config['outputs']['base_results_dir']) / "ROUND_002"
        
        engine.compare(round_dir, sample_metrics, sample_metrics, "feat_neutral")
        
        summary_path = round_dir / "06_COMPARISON" / "improvement_summary.txt"
        assert summary_path.exists()
        
        content = summary_path.read_text()
        assert "Verdict:" in content
        assert "feat_neutral" in content