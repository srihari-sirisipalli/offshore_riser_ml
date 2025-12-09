import pytest
import logging
from unittest.mock import MagicMock
from modules.rfe.stopping_criteria import StoppingCriteria

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def base_config():
    return {
        'iterative': {
            'min_features': 5,
            'max_rounds': 10,
            'degradation_tolerance': 0.1, # 10% tolerance
            'stopping_criteria': 'combined'
        },
        'hyperparameters': {
            'primary_metric': 'cmae'
        }
    }

# --- Tests ---

class TestStoppingCriteria:

    def test_min_features_stop(self, base_config, mock_logger):
        """Test stopping when feature count limit is reached."""
        stopper = StoppingCriteria(base_config, mock_logger)
        
        # History where we start with 6 features.
        # After dropping 1, we have 5 left.
        # Since min_features=5, checking if remaining <= min -> True.
        history = [
            {'round': 0, 'n_features': 6, 'metrics': {'cmae': 1.0}}
        ]
        
        stop, reason = stopper.should_stop(history)
        assert stop is True
        assert "Minimum feature count" in reason

    def test_max_rounds_stop(self, base_config, mock_logger):
        """Test stopping when max rounds limit is reached."""
        stopper = StoppingCriteria(base_config, mock_logger)
        base_config['iterative']['max_rounds'] = 2
        stopper = StoppingCriteria(base_config, mock_logger) # Re-init with new config
        
        history = [
            {'round': 0, 'n_features': 20, 'metrics': {'cmae': 1.0}},
            {'round': 1, 'n_features': 19, 'metrics': {'cmae': 1.0}},
            {'round': 2, 'n_features': 18, 'metrics': {'cmae': 1.0}}
        ]
        
        stop, reason = stopper.should_stop(history)
        assert stop is True
        assert "Maximum rounds" in reason

    def test_performance_degradation_stop(self, base_config, mock_logger):
        """Test stopping when error increases beyond tolerance."""
        stopper = StoppingCriteria(base_config, mock_logger)
        
        # Round 0: CMAE 1.0
        # Round 1: CMAE 1.2 (20% increase)
        # Tolerance is 10% (1.1 threshold).
        # Should Stop.
        history = [
            {'round': 0, 'n_features': 20, 'metrics': {'cmae': 1.0}},
            {'round': 1, 'n_features': 19, 'metrics': {'cmae': 1.2}}
        ]
        
        stop, reason = stopper.should_stop(history)
        assert stop is True
        assert "Performance degraded" in reason

    def test_performance_within_tolerance(self, base_config, mock_logger):
        """Test continuing when error increase is within tolerance."""
        stopper = StoppingCriteria(base_config, mock_logger)
        
        # Round 0: CMAE 1.0
        # Round 1: CMAE 1.05 (5% increase)
        # Tolerance is 10%.
        # Should NOT Stop.
        history = [
            {'round': 0, 'n_features': 20, 'metrics': {'cmae': 1.0}},
            {'round': 1, 'n_features': 19, 'metrics': {'cmae': 1.05}}
        ]
        
        stop, reason = stopper.should_stop(history)
        assert stop is False

    def test_accuracy_metric_logic(self, base_config, mock_logger):
        """Test stopping logic when using an Accuracy metric (Higher is Better)."""
        base_config['hyperparameters']['primary_metric'] = 'accuracy_at_5deg'
        stopper = StoppingCriteria(base_config, mock_logger)
        
        # Round 0: Acc 90%
        # Round 1: Acc 70% (Significant drop)
        # Tolerance 10% -> Threshold 81%
        # 70 < 81 -> Stop.
        history = [
            {'round': 0, 'n_features': 20, 'metrics': {'accuracy_at_5deg': 90.0}},
            {'round': 1, 'n_features': 19, 'metrics': {'accuracy_at_5deg': 70.0}}
        ]
        
        stop, reason = stopper.should_stop(history)
        assert stop is True
        assert "degraded" in reason