import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from utils.circular_metrics import compute_cmae, compute_crmse

class EvaluationEngine:
    """
    Computes comprehensive performance metrics for model predictions.
    Includes circular metrics, accuracy bands, component metrics, and extreme sample identification.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def evaluate(self, predictions: pd.DataFrame, split_name: str, run_id: str) -> Dict[str, float]:
        """
        Compute metrics and save evaluation reports.
        
        Parameters:
            predictions: DataFrame containing true/pred values and errors.
            split_name: 'val' or 'test'.
            run_id: Run identifier.
            
        Returns:
            dict: Dictionary of computed metrics.
        """
        self.logger.info(f"Starting Evaluation for {split_name} set...")
        
        # 1. Setup Output Directory
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        # UPDATED: Changed to 08 to match the new pipeline phase numbering
        output_dir = Path(base_dir) / "08_EVALUATION"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Compute All Metrics
        metrics = self.compute_metrics(predictions)
        
        # 3. Identify Extremes (Best/Worst samples)
        best_df, worst_df = self._identify_extremes(predictions)
        
        # 4. Save Artifacts
        # Save Metrics Table
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_excel(output_dir / f"metrics_{split_name}.xlsx", index=False)
        
        # Save Extremes (useful for debugging outlier cases)
        best_df.to_excel(output_dir / f"best_10_samples_{split_name}.xlsx", index=False)
        worst_df.to_excel(output_dir / f"worst_10_samples_{split_name}.xlsx", index=False)
        
        self.logger.info(f"Evaluation complete. {split_name} CMAE: {metrics['cmae']:.4f}°")
        return metrics

    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate full suite of metrics: Circular, Linear, Bands, Components.
        """
        # Extract columns
        true_angle = df['true_angle'].values
        pred_angle = df['pred_angle'].values
        abs_error = df['abs_error'].values
        error = df['error'].values
        
        true_sin = df['true_sin'].values
        pred_sin = df['pred_sin'].values
        true_cos = df['true_cos'].values
        pred_cos = df['pred_cos'].values
        
        metrics = {}
        
        # --- 1. Circular Metrics ---
        metrics['cmae'] = compute_cmae(true_angle, pred_angle)
        metrics['crmse'] = compute_crmse(true_angle, pred_angle)
        
        # --- 2. Linear Error Statistics ---
        metrics['max_error'] = np.max(abs_error)
        metrics['mean_error'] = np.mean(error)  # Bias (Signed)
        metrics['std_error'] = np.std(error)
        metrics['median_abs_error'] = np.median(abs_error)
        
        # --- 3. Accuracy Bands ---
        # Returns keys like 'accuracy_at_5deg' scaled to 0-100%
        metrics.update(self._compute_accuracy_bands(abs_error))
        
        # --- 4. Percentiles ---
        for p in [50, 75, 90, 95, 99]:
            metrics[f'percentile_{p}'] = np.percentile(abs_error, p)
            
        # --- 5. Component Metrics (Sin/Cos) ---
        metrics['mae_sin'] = np.mean(np.abs(true_sin - pred_sin))
        metrics['rmse_sin'] = np.sqrt(np.mean((true_sin - pred_sin)**2))
        metrics['mae_cos'] = np.mean(np.abs(true_cos - pred_cos))
        metrics['rmse_cos'] = np.sqrt(np.mean((true_cos - pred_cos)**2))
        
        # Metadata
        metrics['n_samples'] = len(df)
        
        return metrics

    def _compute_accuracy_bands(self, abs_error: np.ndarray) -> Dict[str, float]:
        """Calculate percentage of predictions within specific error tolerances."""
        bands = [0, 5, 10, 20]
        results = {}
        total = len(abs_error)
        
        for b in bands:
            if b == 0:
                count = np.sum(abs_error == 0)
            else:
                count = np.sum(abs_error <= b)
            
            # CRITICAL: Scale to 0-100 for readability in Reporting Engine
            results[f'accuracy_at_{b}deg'] = (count / total) * 10