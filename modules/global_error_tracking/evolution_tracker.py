import pandas as pd
import matplotlib.pyplot as plt
import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

class EvolutionTracker:
    """
    Manages the '01_GLOBAL_TRACKING' directory.
    
    Responsibilities:
    1. Aggregate metrics from every completed round into a master timeline.
    2. Track the order of feature elimination.
    3. Generate evolution plots (CMAE vs Rounds, Features vs Error).
    4. Maintain the 'metrics_all_rounds.xlsx' and 'features_eliminated_timeline.xlsx'.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.base_dir = Path(config.get('outputs', {}).get('base_results_dir', 'results'))
        self.tracking_dir = self.base_dir / "01_GLOBAL_TRACKING"
        
        # Ensure subdirectories exist
        (self.tracking_dir / "01_metrics").mkdir(parents=True, exist_ok=True)
        (self.tracking_dir / "02_features").mkdir(parents=True, exist_ok=True)
        (self.tracking_dir / "04_evolution_plots").mkdir(parents=True, exist_ok=True)

    def update_tracker(self, round_summary: Dict[str, Any]):
        """
        Updates global files with the latest round's data.
        
        Args:
            round_summary: Dictionary containing:
                - round (int)
                - n_features (int)
                - dropped_feature (str)
                - metrics (dict): {cmae, crmse, accuracy_at_5deg, ...}
                - hyperparameters (dict)
        """
        self.logger.info(f"Updating Global Evolution Tracker for Round {round_summary['round']}...")
        
        # 1. Update Metrics History
        self._update_metrics_history(round_summary)
        
        # 2. Update Feature Timeline
        self._update_feature_timeline(round_summary)
        
        # 3. Generate Evolution Plots
        # (Only if we have at least 2 points to plot a line)
        if round_summary['round'] > 0:
            self._generate_evolution_plots()

    def _update_metrics_history(self, summary: Dict):
        """Appends row to metrics_all_rounds.xlsx with comprehensive metrics."""
        file_path = self.tracking_dir / "01_metrics" / "metrics_all_rounds.xlsx"

        # Flatten structure
        metrics = summary.get('metrics', {})

        # Use np.nan for missing values to ensure numeric columns in DataFrame
        row = {
            'round_number': summary['round'],
            'num_features': summary['n_features'],
            'feature_removed': summary.get('dropped_feature', 'None'),
            # Validation metrics
            'val_cmae': metrics.get('val_cmae', np.nan),
            'val_crmse': metrics.get('val_crmse', np.nan),
            'val_max_error_deg': metrics.get('val_max_error', np.nan),
            'val_acc0': metrics.get('val_accuracy_at_0deg', np.nan),
            'val_acc5': metrics.get('val_accuracy_at_5deg', np.nan),
            'val_acc10': metrics.get('val_accuracy_at_10deg', np.nan),
            # Test metrics
            'test_cmae': metrics.get('test_cmae', np.nan),
            'test_crmse': metrics.get('test_crmse', np.nan),
            'test_max_error_deg': metrics.get('test_max_error', np.nan),
            'test_acc0': metrics.get('test_accuracy_at_0deg', np.nan),
            'test_acc5': metrics.get('test_accuracy_at_5deg', np.nan),
            'test_acc10': metrics.get('test_accuracy_at_10deg', np.nan),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        new_df = pd.DataFrame([row])

        if file_path.exists():
            existing_df = pd.read_excel(file_path)

            # Check if round already exists to avoid duplicates (resume logic)
            if summary['round'] in existing_df['round_number'].values:
                # Remove the old row and append the new one (handles schema changes)
                existing_df = existing_df[existing_df['round_number'] != summary['round']]
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Sort by round number to maintain order
                updated_df = updated_df.sort_values('round_number').reset_index(drop=True)
            else:
                # Append new row
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = new_df

        updated_df.to_excel(file_path, index=False)

    def _update_feature_timeline(self, summary: Dict):
        """Appends to features_eliminated_timeline.xlsx."""
        file_path = self.tracking_dir / "02_features" / "features_eliminated_timeline.xlsx"
        
        # If no feature dropped (e.g. Round 0 start), skip or log 'Start'
        dropped = summary.get('dropped_feature')
        if not dropped:
            return

        row = {
            'round_number': summary['round'],
            'feature_dropped': dropped,
            'remaining_features': summary['n_features'] - 1 # Count AFTER drop
        }
        
        new_df = pd.DataFrame([row])
        
        if file_path.exists():
            existing_df = pd.read_excel(file_path)
            if summary['round'] in existing_df['round_number'].values:
                # Only update if feature changed (unlikely in deterministic flow)
                return 
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = new_df
            
        updated_df.to_excel(file_path, index=False)

    def _generate_evolution_plots(self):
        """Regenerates plots based on the current history file."""
        history_path = self.tracking_dir / "01_metrics" / "metrics_all_rounds.xlsx"
        if not history_path.exists():
            return

        df = pd.read_excel(history_path)
        if len(df) < 2:
            return

        plot_dir = self.tracking_dir / "04_evolution_plots"

        # --- VALIDATION SET PLOTS ---
        # 1. Val CMAE vs Rounds (Lower is better)
        self._plot_metric_vs_round(df, 'val_cmae', 'Val CMAE (deg)', plot_dir / "val_cmae_evolution.png", inverse=True)

        # 2. Val CRMSE vs Rounds (Lower is better)
        self._plot_metric_vs_round(df, 'val_crmse', 'Val CRMSE (deg)', plot_dir / "val_crmse_evolution.png", inverse=True)

        # 3. Val Max Error vs Rounds (Lower is better)
        self._plot_metric_vs_round(df, 'val_max_error_deg', 'Val Max Error (deg)', plot_dir / "val_max_error_evolution.png", inverse=True)

        # 4. Val Accuracy 0 vs Rounds (Higher is better)
        self._plot_metric_vs_round(df, 'val_acc0', 'Val Accuracy @ 0° (%)', plot_dir / "val_acc0_evolution.png", inverse=False)

        # 5. Val Accuracy 5 vs Rounds (Higher is better)
        self._plot_metric_vs_round(df, 'val_acc5', 'Val Accuracy @ 5° (%)', plot_dir / "val_acc5_evolution.png", inverse=False)

        # 6. Val Accuracy 10 vs Rounds (Higher is better)
        self._plot_metric_vs_round(df, 'val_acc10', 'Val Accuracy @ 10° (%)', plot_dir / "val_acc10_evolution.png", inverse=False)

        # --- TEST SET PLOTS ---
        # 7. Test CMAE vs Rounds (Lower is better)
        self._plot_metric_vs_round(df, 'test_cmae', 'Test CMAE (deg)', plot_dir / "test_cmae_evolution.png", inverse=True)

        # 8. Test CRMSE vs Rounds (Lower is better)
        self._plot_metric_vs_round(df, 'test_crmse', 'Test CRMSE (deg)', plot_dir / "test_crmse_evolution.png", inverse=True)

        # 9. Test Max Error vs Rounds (Lower is better)
        self._plot_metric_vs_round(df, 'test_max_error_deg', 'Test Max Error (deg)', plot_dir / "test_max_error_evolution.png", inverse=True)

        # 10. Test Accuracy 0 vs Rounds (Higher is better)
        self._plot_metric_vs_round(df, 'test_acc0', 'Test Accuracy @ 0° (%)', plot_dir / "test_acc0_evolution.png", inverse=False)

        # 11. Test Accuracy 5 vs Rounds (Higher is better)
        self._plot_metric_vs_round(df, 'test_acc5', 'Test Accuracy @ 5° (%)', plot_dir / "test_acc5_evolution.png", inverse=False)

        # 12. Test Accuracy 10 vs Rounds (Higher is better)
        self._plot_metric_vs_round(df, 'test_acc10', 'Test Accuracy @ 10° (%)', plot_dir / "test_acc10_evolution.png", inverse=False)

        # 13. Features vs Error (The classic "Elbow" plot) - using val_cmae
        self._plot_features_vs_error(df, plot_dir / "feature_count_vs_val_cmae.png", 'val_cmae')

        # 14. Features vs Error (Test set)
        self._plot_features_vs_error(df, plot_dir / "feature_count_vs_test_cmae.png", 'test_cmae')

    def _plot_metric_vs_round(self, df: pd.DataFrame, col: str, ylabel: str, path: Path, inverse: bool):
        """Helper to plot a metric timeline."""
        # FIX: Check if column exists and has non-null data
        if col not in df.columns or df[col].dropna().empty:
            return

        plt.figure(figsize=(10, 6))
        
        # Plot data
        plt.plot(df['round_number'], df[col], marker='o', linestyle='-', linewidth=2, markersize=8)
        
        # Highlight best point (Robustly)
        try:
            if inverse:
                best_idx = df[col].idxmin()
                color = 'green'
            else:
                best_idx = df[col].idxmax()
                color = 'green'
                
            if pd.notna(best_idx):
                best_round = df.loc[best_idx, 'round_number']
                best_val = df.loc[best_idx, col]
                
                plt.scatter([best_round], [best_val], color=color, s=150, zorder=5, label=f'Best ({best_val:.4f})')
                plt.legend()
        except Exception:
            pass # Skip highlighting if calculation fails
        
        plt.title(f'Pipeline Evolution: {ylabel}')
        plt.xlabel('Round Number')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _plot_features_vs_error(self, df: pd.DataFrame, path: Path, metric_col: str):
        """Plots Number of Features (X) vs CMAE (Y)."""
        # FIX: Check for required columns
        if 'num_features' not in df.columns or metric_col not in df.columns:
            return
        if df[metric_col].dropna().empty:
            return

        plt.figure(figsize=(10, 6))

        color = 'purple' if 'val' in metric_col else 'orange'
        split_name = 'Validation' if 'val' in metric_col else 'Test'

        plt.plot(df['num_features'], df[metric_col], marker='s', linestyle='-', color=color, linewidth=2, markersize=8)
        plt.gca().invert_xaxis() # High on left, low on right

        plt.title(f'Feature Reduction Curve ({split_name} Set)')
        plt.xlabel('Number of Features')
        plt.ylabel('CMAE (deg)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()