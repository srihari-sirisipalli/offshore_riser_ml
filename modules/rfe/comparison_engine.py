import pandas as pd
import logging
import json
from pathlib import Path
from typing import Dict, Any, Tuple

class ComparisonEngine:
    """
    Compares the Baseline Model against the Selected Dropped Feature Model.
    
    Responsibilities:
    1. Calculate Deltas for all metrics (New - Baseline).
    2. Interpret Deltas (Improvement vs Degradation).
    3. Generate 'delta_metrics.xlsx'.
    4. Generate 'improvement_summary.txt'.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Define metric behavior
        # Lower is Better: Errors
        self.minimize_metrics = {'cmae', 'crmse', 'max_error', 'mae_sin', 'mae_cos'}
        # Higher is Better: Accuracies
        self.maximize_metrics = {'accuracy_at_5deg', 'accuracy_at_10deg', 'r2_score'}

    def compare(self, 
                round_dir: Path, 
                baseline_metrics: Dict[str, float], 
                dropped_metrics: Dict[str, float], 
                dropped_feature_name: str) -> Dict[str, Any]:
        """
        Executes the comparison logic and saves artifacts.
        
        Args:
            round_dir: Path to current ROUND_XXX folder.
            baseline_metrics: Metrics dictionary of the model with ALL active features.
            dropped_metrics: Metrics dictionary of the model WITHOUT the dropped feature.
            dropped_feature_name: Name of the feature being removed.
            
        Returns:
            Dictionary containing comparison summary.
        """
        self.logger.info(f"Comparing Baseline vs Dropped Feature ('{dropped_feature_name}')...")
        
        output_dir = round_dir / "06_COMPARISON"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Calculate Deltas
        comparison_data = []
        
        # Identify common keys to compare
        all_keys = set(baseline_metrics.keys()) | set(dropped_metrics.keys())
        # Filter for numeric metrics only
        metric_keys = [k for k in all_keys if isinstance(baseline_metrics.get(k, 0), (int, float))]
        
        summary_flags = []

        for metric in metric_keys:
            base_val = baseline_metrics.get(metric, 0.0)
            new_val = dropped_metrics.get(metric, 0.0)
            delta = new_val - base_val
            
            # Determine Status
            status = "NEUTRAL"
            if metric in self.minimize_metrics:
                if delta < 0: status = "IMPROVEMENT" # Error went down
                elif delta > 0: status = "DEGRADATION" # Error went up
            elif metric in self.maximize_metrics:
                if delta > 0: status = "IMPROVEMENT" # Acc went up
                elif delta < 0: status = "DEGRADATION" # Acc went down
            
            comparison_data.append({
                'metric': metric,
                'baseline': base_val,
                'dropped': new_val,
                'delta': delta,
                'status': status
            })
            
            # Track key metrics for log summary
            if metric in ['cmae', 'crmse', 'accuracy_at_5deg']:
                summary_flags.append(f"{metric}: {delta:+.4f} ({status})")

        # 2. Save Delta Metrics Excel
        df_comparison = pd.DataFrame(comparison_data)
        # Sort for readability (CMAE first)
        df_comparison['sort_key'] = df_comparison['metric'].apply(
            lambda x: 0 if 'cmae' in x else (1 if 'acc' in x else 2)
        )
        df_comparison = df_comparison.sort_values('sort_key').drop(columns='sort_key')
        
        df_comparison.to_excel(output_dir / "delta_metrics.xlsx", index=False)

        # 3. Generate Comprehensive Comparison Table (Val + Test for both models)
        self._generate_comprehensive_comparison_table(output_dir, baseline_metrics, dropped_metrics)

        # 4. Generate Summary Text
        summary_text = self._generate_summary_text(dropped_feature_name, summary_flags, df_comparison)
        with open(output_dir / "improvement_summary.txt", 'w') as f:
            f.write(summary_text)
            
        self.logger.info(f"Comparison Complete. Summary saved to {output_dir}")
        self.logger.info(f"  Impact Summary: {', '.join(summary_flags)}")
        
        return {
            'dropped_feature': dropped_feature_name,
            'deltas': df_comparison.set_index('metric')['delta'].to_dict()
        }

    def _generate_comprehensive_comparison_table(self, output_dir: Path, baseline_metrics: Dict, dropped_metrics: Dict):
        """Generate a comprehensive Excel table comparing all metrics for Val and Test sets."""

        # Define key metrics to compare
        key_metrics = ['cmae', 'crmse', 'max_error', 'accuracy_at_0deg', 'accuracy_at_5deg', 'accuracy_at_10deg']

        rows = []
        for metric in key_metrics:
            # Get values for all 4 combinations
            baseline_val = baseline_metrics.get(f'val_{metric}', 0)
            baseline_test = baseline_metrics.get(f'test_{metric}', 0)
            dropped_val = dropped_metrics.get(f'val_{metric}', 0)
            dropped_test = dropped_metrics.get(f'test_{metric}', 0)

            # Calculate deltas
            delta_val = dropped_val - baseline_val
            delta_test = dropped_test - baseline_test

            # Determine if lower/higher is better
            lower_is_better = metric in ['cmae', 'crmse', 'max_error']

            # Status for val
            if lower_is_better:
                status_val = "✓ Improved" if delta_val < 0 else ("✗ Degraded" if delta_val > 0 else "= Same")
            else:
                status_val = "✓ Improved" if delta_val > 0 else ("✗ Degraded" if delta_val < 0 else "= Same")

            # Status for test
            if lower_is_better:
                status_test = "✓ Improved" if delta_test < 0 else ("✗ Degraded" if delta_test > 0 else "= Same")
            else:
                status_test = "✓ Improved" if delta_test > 0 else ("✗ Degraded" if delta_test < 0 else "= Same")

            rows.append({
                'Metric': metric.replace('_', ' ').title(),
                'Baseline_Val': baseline_val,
                'Dropped_Val': dropped_val,
                'Delta_Val': delta_val,
                'Status_Val': status_val,
                'Baseline_Test': baseline_test,
                'Dropped_Test': dropped_test,
                'Delta_Test': delta_test,
                'Status_Test': status_test
            })

        df = pd.DataFrame(rows)
        df.to_excel(output_dir / "comprehensive_model_comparison.xlsx", index=False)
        self.logger.info(f"Generated comprehensive comparison table")

    def _generate_summary_text(self, feature_name: str, flags: list, df: pd.DataFrame) -> str:
        """Creates a human-readable summary string."""
        cmae_row = df[df['metric'] == 'cmae']
        cmae_delta = cmae_row.iloc[0]['delta'] if not cmae_row.empty else 0.0
        
        verdict = "BENEFICIAL" if cmae_delta <= 0 else "DETRIMENTAL (But selected as best option)"
        
        text = [
            f"Comparison Report: Dropping '{feature_name}'",
            f"============================================",
            f"Verdict: {verdict}",
            f"",
            f"Key Metric Changes (New - Baseline):",
        ]
        text.extend([f"- {flag}" for flag in flags])
        text.append("")
        text.append("Interpretation:")
        text.append("  - Negative Delta on Error metrics (CMAE) = Improvement")
        text.append("  - Positive Delta on Accuracy metrics = Improvement")
        
        return "\n".join(text)