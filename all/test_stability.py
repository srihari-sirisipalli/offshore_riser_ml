import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from modules.stability_engine import StabilityEngine

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def stability_config(tmp_path):
    return {
        'outputs': {'base_results_dir': str(tmp_path)},
        'splitting': {'seed': 42},
        'hyperparameters': {'enabled': False}, # Disable HPO for fast test
        'models': {'native': ['ExtraTreesRegressor']},
        'data': {
            'target_sin': 'sin', 'target_cos': 'cos', 
            'hs_column': 'hs', 'drop_columns': []
        },
        'stability': {
            'enabled': True,
            'num_runs': 2
        }
    }

@pytest.fixture
def raw_data():
    return pd.DataFrame({
        'feat1': np.random.rand(20),
        'sin': np.random.rand(20),
        'cos': np.random.rand(20),
        'hs': np.random.rand(20),
        'angle_deg': np.random.rand(20)*360,
        # Required for split engine mock or real execution
        'combined_bin': [0, 1] * 10,
        'angle_bin': [0]*20,
        'hs_bin': [0]*20
    })

def test_jaccard_calculation(stability_config, mock_logger):
    engine = StabilityEngine(stability_config, mock_logger)
    
    # Identical sets -> 1.0
    sets = [{'a', 'b'}, {'a', 'b'}]
    assert engine._compute_feature_stability(sets) == 1.0
    
    # Disjoint sets -> 0.0
    sets = [{'a'}, {'b'}]
    assert engine._compute_feature_stability(sets) == 0.0
    
    # Partial (A={a,b}, B={b,c}) -> I={b}(1), U={a,b,c}(3) -> 0.33
    sets = [{'a', 'b'}, {'b', 'c'}]
    assert abs(engine._compute_feature_stability(sets) - 0.333) < 0.01

@patch('modules.stability_engine.stability_engine.SplitEngine')
@patch('modules.stability_engine.stability_engine.TrainingEngine')
@patch('modules.stability_engine.stability_engine.PredictionEngine')
@patch('modules.stability_engine.stability_engine.EvaluationEngine')
def test_stability_flow(MockEval, MockPred, MockTrain, MockSplit, stability_config, mock_logger, raw_data):
    """
    Test the orchestration logic by mocking the heavy engines.
    """
    engine = StabilityEngine(stability_config, mock_logger)
    
    # Setup Mocks
    # Split returns dummy dfs
    MockSplit.return_value.execute.return_value = (raw_data, raw_data, raw_data)
    
    # Eval returns dummy metrics
    MockEval.return_value.compute_metrics.side_effect = [
        {'cmae': 5.0, 'crmse': 6.0, 'accuracy_at_5deg': 50}, # Run 1
        {'cmae': 5.5, 'crmse': 6.5, 'accuracy_at_5deg': 45}  # Run 2
    ]
    
    result = engine.run_stability_analysis(raw_data, "test_run")
    
    assert result['num_runs'] == 2
    assert result['cmae_mean'] == 5.25 # (5.0 + 5.5) / 2
    assert result['feature_stability_jaccard'] == 1.0 # Feature set didn't change (all cols)