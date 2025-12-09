import logging
from typing import List, Dict, Tuple

class StoppingCriteria:
    """
    Evaluates whether the RFE loop should terminate.
    
    Strategies:
    - min_features: Stop when n_features <= threshold.
    - degradation: Stop when performance worsens beyond a tolerance relative to previous round.
    - stability: Stop when performance improvement is negligible for N rounds (plateau).
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        self.rfe_config = config.get('iterative', {})
        self.min_features = self.rfe_config.get('min_features', 10)
        self.max_rounds = self.rfe_config.get('max_rounds', 50)
        
        # Degradation Thresholds (e.g., 0.05 means 5% worsening is allowed)
        self.degradation_tolerance = self.rfe_config.get('degradation_tolerance', 0.0) 
        
        # Strategy selection
        self.strategy = self.rfe_config.get('stopping_criteria', 'combined')

    def should_stop(self, rounds_history: List[Dict]) -> Tuple[bool, str]:
        """
        Determines if RFE should stop based on history.
        
        Args:
            rounds_history: List of round summary dicts.
            
        Returns:
            (bool, reason_string)
        """
        if not rounds_history:
            return False, ""

        current_round = rounds_history[-1]
        
        # 1. Hard Limit: Min Features
        # Note: 'n_features' in history is the count BEFORE the drop in that round.
        # So if round starts with 11 and drops 1, we have 10 left.
        # We need to check if the count AFTER drop hits the limit.
        features_remaining = current_round['n_features'] - 1
        if features_remaining <= self.min_features:
            return True, f"Minimum feature count reached ({self.min_features})"

        # 2. Hard Limit: Max Rounds
        if current_round['round'] >= self.max_rounds:
            return True, f"Maximum rounds reached ({self.max_rounds})"

        # 3. Performance Checks (Degradation / Stability)
        if len(rounds_history) >= 2:
            return self._check_performance_trends(rounds_history)

        return False, ""

    def _check_performance_trends(self, history: List[Dict]) -> Tuple[bool, str]:
        """Analyzes metric trends."""
        # Compare current round (after drop) vs previous round
        # We use 'metrics' which usually represents the baseline of that round.
        # BUT, to see the effect of the DROP, we should ideally compare:
        # Round N Baseline vs Round N+1 Baseline.
        # Since Round N+1 Baseline isn't computed yet, we approximate using
        # the LOFO prediction of the *dropped* feature in Round N.
        
        # However, a cleaner architectural approach (used in RFEController)
        # is to check if the BEST metric achieved in this round (by dropping a feature)
        # is worse than the baseline of the PREVIOUS round.
        
        # History[-1] is current round. History[-2] is previous round.
        
        curr_metrics = history[-1]['metrics'] # Baseline of current round
        prev_metrics = history[-2]['metrics'] # Baseline of previous round
        
        # Metric to track (Primary Optimization Metric)
        # Default to CMAE if not specified
        metric_name = self.config.get('hyperparameters', {}).get('primary_metric', 'cmae')
        
        curr_val = curr_metrics.get(metric_name, float('inf'))
        prev_val = prev_metrics.get(metric_name, float('inf'))
        
        # Logic for Error Metrics (Lower is better)
        if metric_name in ['cmae', 'crmse', 'max_error']:
            # Degradation: Current Error > Previous Error
            # If tolerance is 0.05, we allow up to 5% increase.
            # threshold = prev * 1.05
            threshold = prev_val * (1.0 + self.degradation_tolerance)
            
            if curr_val > threshold:
                return True, f"Performance degraded: {curr_val:.4f} > {threshold:.4f} ({metric_name})"
                
        # Logic for Accuracy Metrics (Higher is better)
        elif 'accuracy' in metric_name or 'r2' in metric_name:
            # Degradation: Current Acc < Previous Acc
            # threshold = prev * 0.95
            threshold = prev_val * (1.0 - self.degradation_tolerance)
            
            if curr_val < threshold:
                return True, f"Performance degraded: {curr_val:.4f} < {threshold:.4f} ({metric_name})"

        return False, ""