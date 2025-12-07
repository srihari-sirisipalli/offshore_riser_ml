import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path
from modules.global_error_tracking import GlobalErrorTrackingEngine

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def tracking_config(tmp_path):
    return {
        'outputs': {'base_results_dir': str(tmp_path)},
        'global_tracking': {
            'enabled': True,
            'failure_threshold': 5.0
        }
    }

@pytest.fixture
def round_history():
    # Simulate 3 rounds of predictions for 5 samples
    # Sample 0: Always good (Error 1.0)
    # Sample 1: Always bad (Error 10.0) -> Persistent
    # Sample 2: Good then Breaks at Round 2
    
    r1 = pd.DataFrame({'row_index': [0, 1, 2], 'abs_error': [1.0, 10.0, 2.0]})
    r2 = pd.DataFrame({'row_index': [0, 1, 2], 'abs_error': [1.2, 11.0, 15.0]}) # Index 2 breaks
    r3 = pd.DataFrame({'row_index': [0, 1, 2], 'abs_error': [0.9, 9.5, 14.0]})
    
    return [r1, r2, r3]

@pytest.fixture
def feature_history():
    return [
        {'round': 1, 'dropped': ['feat_A']}, # Transition R1 -> R2
        {'round': 2, 'dropped': ['feat_B']}  # Transition R2 -> R3
    ]

def test_evolution_matrix_construction(tracking_config, mock_logger, round_history, feature_history):
    engine = GlobalErrorTrackingEngine(tracking_config, mock_logger)
    
    # Run
    result = engine.track(round_history, feature_history, "test_run")
    output_dir = Path(result['output_dir'])
    
    # Check Matrix
    matrix_file = output_dir / "error_evolution_matrix.xlsx"
    assert matrix_file.exists()
    
    df = pd.read_excel(matrix_file)
    assert len(df) == 3
    assert 'round_1_error' in df.columns
    assert 'round_2_error' in df.columns
    assert 'round_3_error' in df.columns
    
    # Check values for Sample 2 (Row 2 in file, index 2)
    row2 = df[df['row_index'] == 2].iloc[0]
    assert row2['round_1_error'] == 2.0
    assert row2['round_2_error'] == 15.0

def test_persistent_failures(tracking_config, mock_logger, round_history, feature_history):
    engine = GlobalErrorTrackingEngine(tracking_config, mock_logger)
    result = engine.track(round_history, feature_history, "test_run")
    output_dir = Path(result['output_dir'])
    
    # Check persistent file
    persist_file = output_dir / "persistent_failures.xlsx"
    assert persist_file.exists()
    
    df = pd.read_excel(persist_file)
    # Sample 1 was always > 5.0
    assert 1 in df['row_index'].values
    # Sample 0 never failed
    assert 0 not in df['row_index'].values

def test_breakpoint_detection(tracking_config, mock_logger, round_history, feature_history):
    engine = GlobalErrorTrackingEngine(tracking_config, mock_logger)
    result = engine.track(round_history, feature_history, "test_run")
    output_dir = Path(result['output_dir'])
    
    # Check breakpoints
    bp_file = output_dir / "breakpoint_analysis.xlsx"
    assert bp_file.exists()
    
    df = pd.read_excel(bp_file)
    
    # Sample 2 broke at Round 2 (prev 2.0 -> curr 15.0)
    # Feature dropped prior was 'feat_A' (index 0 in history)
    row = df[df['row_index'] == 2].iloc[0]
    assert row['breakpoint_round'] == 2
    assert row['prev_error'] == 2.0
    assert row['new_error'] == 15.0
    assert "feat_A" in row['features_dropped_prior']