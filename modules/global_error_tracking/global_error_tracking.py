import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

class GlobalErrorTrackingEngine:
    """
    Tracks sample-level error evolution across multiple Feature Selection rounds.
    Identifies persistent failures, breakpoints, and feature drop impacts.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.track_config = config.get('global_tracking', {})
        
    def track(self, 
              round_predictions: List[pd.DataFrame], 
              feature_history: List[Dict], 
              run_id: str) -> Dict[str, Any]:
        """
        Execute global tracking analysis.
        
        Parameters:
            round_predictions: List of DataFrames (one per round) containing 'index' and 'abs_error'.
            feature_history: List of dicts describing features dropped per round.
            run_id: Run identifier.
        """
        if not self.track_config.get('enabled', True):
            self.logger.info("Global Error Tracking disabled.")
            return {}
            
        if len(round_predictions) < 2:
            self.logger.info("Global Tracking skipped: Need at least 2 rounds of history.")
            return {}

        self.logger.info(f"Starting Global Error Tracking across {len(round_predictions)} rounds...")
        
        # 1. Setup Directory
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "GLOBAL_ERROR_TRACKING"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Build Evolution Matrix
        evolution_df = self._build_evolution_matrix(round_predictions)
        evolution_df.to_excel(output_dir / "error_evolution_matrix.xlsx", index=False)
        
        # 3. Identify Persistent Failures
        self._identify_persistent_failures(evolution_df, output_dir)
        
        # 4. Analyze Breakpoints (When did samples break?)
        self._analyze_breakpoints(evolution_df, feature_history, output_dir)
        
        self.logger.info("Global Error Tracking complete.")
        return {"output_dir": str(output_dir)}

    def _build_evolution_matrix(self, predictions_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Create a matrix where rows=samples, columns=rounds, values=abs_error.
        """
        # Start with the index from the first round
        # Assuming 'row_index' is the stable identifier
        matrix = pd.DataFrame({'row_index': predictions_list[0]['row_index']})
        
        for i, df in enumerate(predictions_list):
            round_col = f"round_{i+1}_error"
            # Merge on row_index to ensure alignment even if order changed
            temp = df[['row_index', 'abs_error']].rename(columns={'abs_error': round_col})
            matrix = matrix.merge(temp, on='row_index', how='left')
            
        return matrix

    def _identify_persistent_failures(self, matrix: pd.DataFrame, output_dir: Path) -> None:
        """
        Find samples that fail consistently across most rounds.
        """
        threshold = self.track_config.get('failure_threshold', 10.0)
        round_cols = [c for c in matrix.columns if c.startswith('round_')]
        
        # Calculate failure rate per sample
        # (Count how many rounds have error > threshold)
        failures = (matrix[round_cols] > threshold).sum(axis=1)
        failure_rate = failures / len(round_cols)
        
        matrix['failure_rate'] = failure_rate
        
        # Identify "Chronic" failures (>80% of rounds)
        chronic = matrix[matrix['failure_rate'] > 0.8].copy()
        
        if not chronic.empty:
            chronic.sort_values('failure_rate', ascending=False).to_excel(
                output_dir / "persistent_failures.xlsx", index=False
            )
            self.logger.warning(f"Found {len(chronic)} persistent failure samples (failed >80% rounds).")

    def _analyze_breakpoints(self, matrix: pd.DataFrame, feature_history: List[Dict], output_dir: Path) -> None:
        """
        Identify the specific round where a sample's error spiked significantly.
        Correlates breaks with feature drops.
        """
        threshold = self.track_config.get('failure_threshold', 10.0)
        round_cols = [c for c in matrix.columns if c.startswith('round_') and 'error' in c]
        
        breakpoints = []
        
        for idx, row in matrix.iterrows():
            errors = row[round_cols].values
            
            # Find first index where error jumps from < threshold to > threshold
            for i in range(1, len(errors)):
                prev_err = errors[i-1]
                curr_err = errors[i]
                
                # Check for "Break" event: previously good, now bad
                if prev_err <= threshold and curr_err > threshold:
                    # Get feature dropped in this round (round i+1 corresponds to index i in history?)
                    # History index mapping depends on implementation. 
                    # Usually history[0] describes what happened to create Round 2 (or end of Round 1).
                    # We assume feature_history[i-1] describes transition from Round i to i+1.
                    
                    dropped = "Unknown"
                    if i-1 < len(feature_history):
                        dropped = str(feature_history[i-1].get('dropped', []))
                    
                    breakpoints.append({
                        'row_index': row['row_index'],
                        'breakpoint_round': i + 1,
                        'prev_error': prev_err,
                        'new_error': curr_err,
                        'features_dropped_prior': dropped
                    })
                    break # Record first break only
                    
        if breakpoints:
            bp_df = pd.DataFrame(breakpoints)
            bp_df.to_excel(output_dir / "breakpoint_analysis.xlsx", index=False)
            self.logger.info(f"Identified {len(bp_df)} samples that 'broke' during feature elimination.")