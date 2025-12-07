import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils.circular_metrics import reconstruct_angle, wrap_angle

class EnsemblingEngine:
    """
    Combines predictions from multiple models/runs to improve performance.
    Strategies: Simple Average, Weighted Average, Hs-Sensitive Selection.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.ens_config = config.get('ensembling', {})
        self.enabled = self.ens_config.get('enabled', False)
        
    def ensemble(self, 
                 predictions_list: List[pd.DataFrame], 
                 metrics_list: List[Dict[str, float]], 
                 split_name: str, 
                 run_id: str) -> pd.DataFrame:
        """
        Execute ensembling strategy.
        
        Parameters:
            predictions_list: List of DataFrames (must have matching indices).
            metrics_list: List of metric dicts (for weighting).
            split_name: 'val' or 'test'.
            run_id: Run identifier.
            
        Returns:
            pd.DataFrame: Ensembled predictions.
        """
        if not self.enabled:
            self.logger.info("Ensembling disabled in config.")
            return pd.DataFrame()
            
        if len(predictions_list) < 2:
            self.logger.warning("Ensembling requires at least 2 prediction sets. Skipping.")
            return predictions_list[0] if predictions_list else pd.DataFrame()

        self.logger.info(f"Starting Ensembling for {split_name} set with {len(predictions_list)} models...")
        
        # 1. Setup Output Directory
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "09_ADVANCED_ANALYTICS" / "ensembling"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Validate Inputs
        self._validate_alignment(predictions_list)
        
        # 3. Select Strategy
        strategy = self.ens_config.get('strategy', 'weighted')
        
        if strategy == 'simple':
            ensemble_df = self._simple_average(predictions_list)
        elif strategy == 'weighted':
            ensemble_df = self._weighted_average(predictions_list, metrics_list)
        elif strategy == 'hs_sensitive':
            ensemble_df = self._hs_sensitive_ensemble(predictions_list, metrics_list)
        else:
            self.logger.warning(f"Unknown strategy '{strategy}', defaulting to simple average.")
            ensemble_df = self._simple_average(predictions_list)
            
        # 4. Recompute Errors for Ensemble
        # Assume truth comes from first model (validated alignment)
        true_angle = predictions_list[0]['true_angle'].values
        pred_angle = ensemble_df['pred_angle'].values
        
        error = wrap_angle(true_angle - pred_angle)
        ensemble_df['error'] = error
        ensemble_df['abs_error'] = np.abs(error)
        ensemble_df['true_angle'] = true_angle
        
        # Add metadata if present
        for col in ['Hs', 'hs_bin', 'angle_bin', 'combined_bin']:
            if col in predictions_list[0].columns:
                ensemble_df[col] = predictions_list[0][col].values
        
        # 5. Save Artifacts
        save_path = output_dir / f"ensemble_predictions_{split_name}.xlsx"
        ensemble_df.to_excel(save_path, index=False)
        
        self.logger.info(f"Ensembling complete. Saved to {save_path}")
        return ensemble_df

    def _validate_alignment(self, predictions_list: List[pd.DataFrame]):
        """Ensure all DataFrames have same length and index."""
        base_idx = predictions_list[0].index
        for i, df in enumerate(predictions_list[1:]):
            if len(df) != len(base_idx):
                raise ValueError(f"Model {i+1} has length {len(df)}, expected {len(base_idx)}.")
            # If indices don't match, we might be merging wrong rows
            # In production, we'd enforce exact index matching or sort.
            # Here we assume order is preserved from PredictionEngine.

    def _simple_average(self, predictions_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Average Sin and Cos components."""
        # Stack: (n_models, n_samples)
        sin_stack = np.stack([df['pred_sin'].values for df in predictions_list])
        cos_stack = np.stack([df['pred_cos'].values for df in predictions_list])
        
        avg_sin = np.mean(sin_stack, axis=0)
        avg_cos = np.mean(cos_stack, axis=0)
        
        angle = reconstruct_angle(avg_sin, avg_cos)
        
        return pd.DataFrame({
            'pred_sin': avg_sin,
            'pred_cos': avg_cos,
            'pred_angle': angle
        })

    def _weighted_average(self, predictions_list: List[pd.DataFrame], metrics_list: List[Dict]) -> pd.DataFrame:
        """Weighted average based on Inverse Error (Lower CMAE = Higher Weight)."""
        scheme = self.ens_config.get('weighting_scheme', 'inverse_error')
        
        if scheme == 'inverse_error':
            # Weight = 1 / CMAE (avoid div by zero)
            cmaes = np.array([m.get('cmae', 1.0) for m in metrics_list])
            weights = 1.0 / (cmaes + 1e-6)
        else:
            # Default to uniform if scheme unknown
            weights = np.ones(len(predictions_list))
            
        # Normalize weights
        weights = weights / np.sum(weights)
        
        self.logger.info(f"Ensemble Weights: {weights}")
        
        sin_stack = np.stack([df['pred_sin'].values for df in predictions_list])
        cos_stack = np.stack([df['pred_cos'].values for df in predictions_list])
        
        # Weighted sum across axis 0 (models)
        # weights[:, None] broadcasts shape (n_models, 1) against (n_models, n_samples)
        avg_sin = np.sum(sin_stack * weights[:, None], axis=0)
        avg_cos = np.sum(cos_stack * weights[:, None], axis=0)
        
        angle = reconstruct_angle(avg_sin, avg_cos)
        
        return pd.DataFrame({
            'pred_sin': avg_sin,
            'pred_cos': avg_cos,
            'pred_angle': angle
        })

    def _hs_sensitive_ensemble(self, predictions_list: List[pd.DataFrame], metrics_list: List[Dict]) -> pd.DataFrame:
        """
        Advanced: Select best model per Hs Bin?
        
        NOTE: This requires per-Hs metrics to be passed in, which makes the interface complex.
        Simplification: We will fallback to weighted average for now unless per-bin metrics are available.
        Implementation of full Hs-Sensitive logic requires refactoring EvaluationEngine to return granular per-bin metrics.
        
        Fallback: Return Weighted Average.
        """
        self.logger.warning("Hs-Sensitive ensembling requires per-bin metrics. Falling back to Weighted Average.")
        return self._weighted_average(predictions_list, metrics_list)