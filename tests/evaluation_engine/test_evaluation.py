import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from unittest.mock import MagicMock
from modules.evaluation_engine import EvaluationEngine


@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def sample_predictions():
    # Construct a dataframe with known errors
    # Row 1: Perfect match (Error 0)
    # Row 2: 5 deg error
    # Row 3: 10 deg error
    # Row 4: 20 deg error
    # Row 5: 180 deg error (Max)
    
    data = {
        'true_angle': [10, 100, 200, 300, 0],
        'pred_angle': [10, 105, 210, 320, 180],
        'error':      [0,  -5, -10, -20, -180], # Signed wrapped error
        'abs_error':  [0,   5,  10,  20,  180],
        
        # Component dummy data (just for existence check)
        'true_sin': [0]*5, 'pred_sin': [0]*5,
        'true_cos': [1]*5, 'pred_cos': [1]*5,
        
        'Hs': [1.0]*5
    }
    return pd.DataFrame(data)

@pytest.fixture
def eval_config(tmp_path):
    return {
        'outputs': {'base_results_dir': str(tmp_path)}
    }

def test_metrics_calculation(eval_config, mock_logger, sample_predictions):
    engine = EvaluationEngine(eval_config, mock_logger)
    metrics = engine.compute_metrics(sample_predictions)
    
    # Check CMAE
    # Errors: 0, 5, 10, 20, 180
    # Mean: (0+5+10+20+180)/5 = 215/5 = 43.0
    assert metrics['cmae'] == 43.0
    
    # Check Max Error
    assert metrics['max_error'] == 180.0
    
    # Check Accuracy Bands
    # <= 5: 2 samples (0, 5) -> 40%
    assert metrics['accuracy_at_5deg'] == 40.0
    
    # <= 10: 3 samples (0, 5, 10) -> 60%
    assert metrics['accuracy_at_10deg'] == 60.0
    
    # <= 20: 4 samples (0, 5, 10, 20) -> 80%
    assert metrics['accuracy_at_20deg'] == 80.0

def test_extremes_identification(eval_config, mock_logger, sample_predictions):
    engine = EvaluationEngine(eval_config, mock_logger)
    best, worst = engine._identify_extremes(sample_predictions, n=2)
    
    # Best should be error 0 and 5
    assert best.iloc[0]['abs_error'] == 0
    assert best.iloc[1]['abs_error'] == 5
    
    # Worst should be error 180 and 20
    assert worst.iloc[0]['abs_error'] == 180
    assert worst.iloc[1]['abs_error'] == 20

def test_evaluation_workflow(eval_config, mock_logger, sample_predictions):
    """Test the full execute method including file saving."""
    engine = EvaluationEngine(eval_config, mock_logger)
    metrics = engine.evaluate(sample_predictions, "test_split", "run_id_123")
    
    assert 'cmae' in metrics
    
    # Check artifacts
    base = Path(eval_config['outputs']['base_results_dir'])
    # FIX: Corrected directory name to match implementation
    eval_dir = base / "08_EVALUATION"
    
    assert (eval_dir / "metrics_test_split.xlsx").exists()
    assert (eval_dir / "best_10_samples_test_split.xlsx").exists()
    assert (eval_dir / "worst_10_samples_test_split.xlsx").exists()