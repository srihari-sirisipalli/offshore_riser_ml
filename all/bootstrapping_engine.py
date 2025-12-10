import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional

from utils.circular_metrics import compute_cmae

class BootstrappingEngine:
    """
    Performs bootstrapping analysis to estimate the confidence intervals 
    of the model's performance metrics.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.boot_config = config.get('bootstrapping', {})
        self.enabled = self.boot_config.get('enabled', True)

    def bootstrap(self, predictions: pd.DataFrame, split_name: str, run_id: str) -> Dict[str, Any]:
        """
        Run bootstrapping on the predictions to generate confidence intervals for CMAE.
        """
        if not self.enabled:
            self.logger.info("Bootstrapping disabled in config.")
            return {}

        if predictions.empty:
            self.logger.warning("No predictions available for bootstrapping.")
            return {}

        self.logger.info(f"Starting Bootstrapping analysis for {split_name}...")

        # 1. Setup Output Directory
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "09_ADVANCED_ANALYTICS" / "bootstrapping" / split_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Configuration & Validation
        n_samples = self.boot_config.get('n_samples', 1000)
        sample_ratio = self.boot_config.get('sample_ratio', 1.0)
        confidence = self.boot_config.get('confidence_level', 0.95)
        
        if not (0 < confidence < 1):
            self.logger.warning(
                f"Invalid confidence level ({confidence}). Must be between 0 and 1 (exclusive). Defaulting to 0.95."
            )
            confidence = 0.95
            
        if sample_ratio <= 0:
            self.logger.warning(
                f"Sample ratio ({sample_ratio}) must be greater than 0. Defaulting to 1.0."
            )
            sample_ratio = 1.0

        master_seed = self.config['splitting']['seed']
        bootstrap_seed = self.config.get('_internal_seeds', {}).get('bootstrap', master_seed + 999)
        rng = np.random.RandomState(bootstrap_seed)

        # 3. Prepare Data
        y_true = predictions['true_angle'].values
        y_pred = predictions['pred_angle'].values
        n_rows = len(predictions)
        
        size = max(1, int(n_rows * sample_ratio))

        if size < 5:
            self.logger.warning(
                f"Calculated bootstrap sample size ({size}) is less than the recommended minimum (5). "
                "This may affect statistical significance. Consider increasing 'num_samples' or 'sample_ratio'."
            )
        
        metrics_list = []

        # 4. Bootstrap Loop
        for i in range(n_samples):
            indices = rng.choice(n_rows, size=size, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            metric_val = compute_cmae(y_true_boot, y_pred_boot)
            metrics_list.append({'cmae': metric_val})

        # 5. Compute Intervals
        ci_results = self._compute_ci(metrics_list, confidence)
        if not ci_results:
             self.logger.warning("Could not compute confidence intervals.")
             return {}
        cmae_results = ci_results['cmae']
        
        self.logger.info(f"Bootstrap CMAE ({confidence*100:.0f}% CI): {cmae_results['mean']:.4f} [{cmae_results['lower']:.4f}, {cmae_results['upper']:.4f}]")

        # 6. Save Results
        cmae_dist = np.array([m['cmae'] for m in metrics_list])
        stats = pd.DataFrame({
            'metric': ['CMAE'],
            'mean': [cmae_results['mean']],
            'std': [np.std(cmae_dist)],
            'ci_lower': [cmae_results['lower']],
            'ci_upper': [cmae_results['upper']],
            'confidence_level': [confidence],
            'n_bootstraps': [n_samples]
        })
        stats.to_excel(output_dir / "bootstrap_ci.xlsx", index=False)
        
        samples_df = pd.DataFrame(metrics_list)
        samples_df.to_excel(output_dir / "bootstrap_samples.xlsx", index=False)


        # 7. Visualize
        self._plot_distribution(cmae_dist, cmae_results['mean'], cmae_results['lower'], cmae_results['upper'], output_dir, confidence)
        
        return ci_results

    def _plot_distribution(self, data: np.ndarray, mean: float, lower: float, upper: float, output_dir: Path, confidence: float):
        """Plot the bootstrap distribution."""
        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 6))
        
        # FIX: Handle cases with no data variance, being robust to float precision
        num_unique = len(np.unique(np.round(data, decimals=5)))
        if num_unique == 1:
            bins = 1
        else:
            bins = min(50, len(np.unique(data)))

        plt.hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
        plt.axvline(lower, color='green', linestyle=':', linewidth=2, label=f'{confidence*100:.0f}% CI')
        plt.axvline(upper, color='green', linestyle=':', linewidth=2)
        plt.title('Bootstrap Distribution of CMAE')
        plt.xlabel('CMAE (degrees)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "bootstrap_dist_cmae.png")
        plt.close()

    def _compute_ci(self, metrics_list: list[dict], confidence: float) -> dict:
        """
        Computes confidence intervals for a distribution of metrics.
        """
        if not metrics_list:
            return {}

        # Transpose list of dicts to dict of lists
        metrics_dist = {
            metric: [sample[metric] for sample in metrics_list]
            for metric in metrics_list[0]
        }
        
        results = {}
        for metric, values in metrics_dist.items():
            lower_p = ((1.0 - confidence) / 2.0) * 100
            upper_p = (confidence + (1.0 - confidence) / 2.0) * 100
            
            # FIX: Use nanpercentile for robustness
            ci_lower = np.nanpercentile(values, lower_p)
            ci_upper = np.nanpercentile(values, upper_p)
            mean_est = np.nanmean(values)
            
            results[metric] = {
                'mean': mean_est,
                'lower': ci_lower,
                'upper': ci_upper
            }
        return results