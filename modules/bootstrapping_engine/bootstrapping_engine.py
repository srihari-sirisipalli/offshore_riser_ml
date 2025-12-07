import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List

from utils.circular_metrics import compute_cmae, compute_crmse

class BootstrappingEngine:
    """
    Performs Non-Parametric Bootstrap resampling to quantify uncertainty
    in model performance metrics (Confidence Intervals).
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.boot_config = config.get('bootstrapping', {})
        self.enabled = self.boot_config.get('enabled', False)
        
    def bootstrap(self, predictions: pd.DataFrame, split_name: str, run_id: str) -> Dict[str, Any]:
        """
        Execute bootstrap analysis.
        
        Parameters:
            predictions: DataFrame containing 'true_angle', 'pred_angle', 'abs_error'.
            split_name: 'val' or 'test'.
            run_id: Run identifier.
            
        Returns:
            dict: Confidence Intervals and summary stats.
        """
        if not self.enabled:
            self.logger.info("Bootstrapping disabled in config.")
            return {}

        self.logger.info(f"Starting Bootstrap Analysis for {split_name} set...")
        
        # 1. Setup Directories
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "09_ADVANCED_ANALYTICS" / "bootstrapping" / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Configure Parameters
        n_samples = self.boot_config.get('num_samples', 500)
        confidence = self.boot_config.get('confidence_level', 0.95)
        # Default sample ratio is 1.0 (sample size = dataset size), typical for bootstrap
        sample_ratio = self.boot_config.get('sample_ratio', 1.0) 
        
        # 3. Perform Resampling
        bootstrap_metrics = self._perform_sampling(predictions, n_samples, sample_ratio)
        
        # 4. Compute Confidence Intervals
        ci_results = self._compute_ci(bootstrap_metrics, confidence)
        
        # 5. Save Artifacts
        # All samples (for transparency)
        pd.DataFrame(bootstrap_metrics).to_excel(output_dir / "bootstrap_samples.xlsx", index=False)
        
        # CI Summary
        pd.DataFrame(ci_results).T.to_excel(output_dir / "bootstrap_ci.xlsx")
        
        # 6. Visualize
        self._plot_bootstrap_distributions(bootstrap_metrics, ci_results, output_dir)
        
        self.logger.info(f"Bootstrapping complete. CMAE 95% CI: [{ci_results['cmae']['lower']:.2f}, {ci_results['cmae']['upper']:.2f}]")
        return ci_results

    def _perform_sampling(self, df: pd.DataFrame, n_samples: int, ratio: float) -> List[Dict[str, float]]:
        """Run the resampling loop."""
        results = []
        n_rows = len(df)
        size = int(n_rows * ratio)
        
        # Pre-convert to numpy for speed
        true_angles = df['true_angle'].values
        pred_angles = df['pred_angle'].values
        abs_errors = df['abs_error'].values
        
        for i in range(n_samples):
            # Resample indices with replacement
            indices = np.random.choice(n_rows, size=size, replace=True)
            
            sample_true = true_angles[indices]
            sample_pred = pred_angles[indices]
            sample_abs_err = abs_errors[indices]
            
            # Compute metrics for this sample
            metrics = {
                'sample_id': i,
                'cmae': compute_cmae(sample_true, sample_pred),
                'crmse': compute_crmse(sample_true, sample_pred),
                'accuracy_5deg': (np.sum(sample_abs_err <= 5) / size) * 100,
                'accuracy_10deg': (np.sum(sample_abs_err <= 10) / size) * 100
            }
            results.append(metrics)
            
        return results

    def _compute_ci(self, metrics_list: List[Dict], confidence: float) -> Dict[str, Dict[str, float]]:
        """Calculate percentiles for Confidence Intervals."""
        df = pd.DataFrame(metrics_list)
        stats_summary = {}
        
        alpha = 1.0 - confidence
        lower_p = (alpha / 2.0) * 100
        upper_p = (1.0 - alpha / 2.0) * 100
        
        metrics_to_analyze = ['cmae', 'crmse', 'accuracy_5deg', 'accuracy_10deg']
        
        for m in metrics_to_analyze:
            values = df[m].values
            stats_summary[m] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'lower': np.percentile(values, lower_p),
                'upper': np.percentile(values, upper_p),
                'confidence_level': confidence
            }
            
        return stats_summary

    def _plot_bootstrap_distributions(self, metrics_list: List[Dict], ci_results: Dict, output_dir: Path):
        """Generate histograms with CI bands."""
        df = pd.DataFrame(metrics_list)
        plt.switch_backend('Agg')
        
        for metric in ['cmae', 'crmse', 'accuracy_5deg']:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[metric], kde=True, color='purple', alpha=0.6)
            
            # Draw CI lines
            lower = ci_results[metric]['lower']
            upper = ci_results[metric]['upper']
            mean_val = ci_results[metric]['mean']
            
            plt.axvline(lower, color='red', linestyle='--', label=f'Lower CI: {lower:.2f}')
            plt.axvline(upper, color='red', linestyle='--', label=f'Upper CI: {upper:.2f}')
            plt.axvline(mean_val, color='black', linestyle='-', label=f'Mean: {mean_val:.2f}')
            
            plt.title(f'Bootstrap Distribution: {metric.upper()}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(output_dir / f"bootstrap_dist_{metric}.png", dpi=150)
            plt.close()