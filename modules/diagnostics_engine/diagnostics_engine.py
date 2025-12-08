import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from pathlib import Path
from scipy import stats
from typing import Optional
from contextlib import contextmanager # Added for FIX #78

class DiagnosticsEngine:
    """
    Generates comprehensive diagnostic visualizations for model predictions.
    Includes scatter plots, error distributions, residual analysis, and per-Hs performance.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.diag_config = config.get('diagnostics', {})
        self.dpi = self.diag_config.get('dpi', 200)
        self.fmt = self.diag_config.get('save_format', 'png')
        
        # FIX #78: Removed global style setting from __init__. Now managed by _plot_context.
        # sns.set_theme(style="whitegrid")
        # plt.rcParams.update({'figure.max_open_warning': 0})

    @contextmanager
    def _plot_context(self):
        """
        FIX #78: Context manager to apply and then reset plot settings.
        Ensures global matplotlib/seaborn settings don't leak.
        """
        # Save current rcParams and seaborn theme
        original_rcParams = plt.rcParams.copy()
        original_seaborn_theme = sns.axes_style() # Captures current seaborn style
        try:
            # Apply engine's specific settings
            sns.set_theme(style="whitegrid")
            plt.rcParams.update({'figure.max_open_warning': 0})
            yield
        finally:
            # Restore original rcParams and seaborn theme
            plt.rcParams.update(original_rcParams)
            sns.set_theme(style=original_seaborn_theme) # Restore seaborn theme
            plt.close('all') # Close all figures created within this context

    def generate_all(self, predictions: pd.DataFrame, split_name: str, run_id: str) -> None:
        """
        Orchestrate generation of all configured plots.
        """
        self.logger.info(f"Generating diagnostics for {split_name} set...")
        
        # FIX #78: Wrap all plotting logic in the context manager.
        with self._plot_context():
            # 1. Setup Directories
            base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
            root_dir = Path(base_dir) / "09_DIAGNOSTICS"
            
            dirs = {
                'index': root_dir / "index_plots",
                'scatter': root_dir / "scatter_plots",
                'residual': root_dir / "residual_plots",
                'dist': root_dir / "distribution_plots",
                'qq': root_dir / "qq_plots",
                'per_hs': root_dir / "per_hs_plots"
            }
            
            for d in dirs.values():
                d.mkdir(parents=True, exist_ok=True)
                
            # 2. Generate Plots based on flags
            if self.diag_config.get('generate_index_plots', True):
                self._plot_index_vs_values(predictions, split_name, dirs['index'])
                
            if self.diag_config.get('generate_scatter_plots', True):
                self._plot_scatter(predictions, split_name, dirs['scatter'])
                
            if self.diag_config.get('generate_residual_plots', True):
                self._plot_residuals(predictions, split_name, dirs['residual'])
                
            if self.diag_config.get('generate_distribution_plots', True):
                self._plot_distributions(predictions, split_name, dirs['dist'])
                
            if self.diag_config.get('generate_qq_plots', True):
                self._plot_qq(predictions, split_name, dirs['qq'])
                
            if self.diag_config.get('generate_per_hs_accuracy', True):
                self._plot_per_hs_analysis(predictions, split_name, dirs['per_hs'])
                
            self.logger.info(f"Diagnostics generation complete for {split_name}.")

    def _save_fig(self, output_dir: Path, filename: str):
        """Helper to save and close figures."""
        path = output_dir / f"{filename}.{self.fmt}"
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def _plot_index_vs_values(self, df: pd.DataFrame, split: str, output_dir: Path):
        """
        Plot Actual/Predicted/Error vs Index using SCATTER plots.
        """
        # 1. Index vs Actual & Predicted (Scatter)
        plt.figure(figsize=(12, 6))
        # Actuals as small blue dots
        plt.scatter(df.index, df['true_angle'], label='True', alpha=0.5, s=15, color='blue', marker='o')
        # Preds as small red crosses
        plt.scatter(df.index, df['pred_angle'], label='Pred', alpha=0.5, s=15, color='red', marker='x')
        
        plt.title(f'{split.upper()}: Index vs Angle')
        plt.xlabel('Sample Index')
        plt.ylabel('Angle (deg)')
        plt.legend()
        self._save_fig(output_dir, f"index_vs_values_{split}")
        
        # 2. Index vs Error (Scatter)
        plt.figure(figsize=(12, 4))
        plt.scatter(df.index, df['error'], color='purple', alpha=0.6, s=10)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.title(f'{split.upper()}: Index vs Error')
        plt.ylabel('Error (deg)')
        self._save_fig(output_dir, f"index_vs_error_{split}")
        
        # 3. Index vs Abs Error (Scatter)
        plt.figure(figsize=(12, 4))
        plt.scatter(df.index, df['abs_error'], color='darkorange', alpha=0.6, s=10)
        plt.title(f'{split.upper()}: Index vs Absolute Error')
        plt.ylabel('Abs Error (deg)')
        self._save_fig(output_dir, f"index_vs_abs_error_{split}")

    def _plot_scatter(self, df: pd.DataFrame, split: str, output_dir: Path):
        """Scatter plot: Actual vs Predicted."""
        plt.figure(figsize=(8, 8))
        
        # Color points by absolute error
        sc = plt.scatter(df['true_angle'], df['pred_angle'], 
                         c=df['abs_error'], cmap='viridis', s=15, alpha=0.6)
        
        # Perfect line
        lims = [0, 360]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect')
        
        # +/- 5 deg bounds
        plt.fill_between(lims, [x-5 for x in lims], [x+5 for x in lims], color='green', alpha=0.1, label='±5° Band')
        
        plt.xlim(lims)
        plt.ylim(lims)
        plt.colorbar(sc, label='Abs Error (deg)')
        plt.title(f'{split.upper()}: Actual vs Predicted')
        plt.xlabel('True Angle (deg)')
        plt.ylabel('Predicted Angle (deg)')
        plt.legend()
        self._save_fig(output_dir, f"actual_vs_pred_{split}")

    def _plot_residuals(self, df: pd.DataFrame, split: str, output_dir: Path):
        """Residuals vs Predicted Value."""
        plt.figure(figsize=(10, 6))
        plt.scatter(df['pred_angle'], df['error'], alpha=0.5, s=15)
        plt.axhline(0, color='r', linestyle='--')
        plt.title(f'{split.upper()}: Residuals vs Predicted')
        plt.xlabel('Predicted Angle (deg)')
        plt.ylabel('Residual (Error) (deg)')
        self._save_fig(output_dir, f"residuals_vs_pred_{split}")

    def _plot_distributions(self, df: pd.DataFrame, split: str, output_dir: Path):
        """Error Histogram and Boxplot."""
        # Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(df['error'], kde=True, bins=50, color='teal')
        plt.title(f'{split.upper()}: Error Distribution')
        plt.xlabel('Error (deg)')
        self._save_fig(output_dir, f"error_hist_{split}")
        
        # Boxplot (Abs Error)
        plt.figure(figsize=(6, 6))
        sns.boxplot(y=df['abs_error'], color='orange')
        plt.title(f'{split.upper()}: Absolute Error Boxplot')
        plt.ylabel('Absolute Error (deg)')
        self._save_fig(output_dir, f"error_boxplot_{split}")

    def _plot_qq(self, df: pd.DataFrame, split: str, output_dir: Path):
        """Q-Q Plot to check normality of residuals."""
        plt.figure(figsize=(8, 8))
        stats.probplot(df['error'], dist="norm", plot=plt)
        plt.title(f'{split.upper()}: Q-Q Plot')
        self._save_fig(output_dir, f"qq_plot_{split}")

    def _plot_per_hs_analysis(self, df: pd.DataFrame, split: str, output_dir: Path):
        """Plot Error vs Significant Wave Height (Hs)."""
        hs_col = self.config['data']['hs_column']
        
        if hs_col not in df.columns:
            self.logger.warning(f"Hs column '{hs_col}' not found in predictions. Skipping Per-Hs plots.")
            return

        # Scatter: Abs Error vs Hs
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=hs_col, y='abs_error', alpha=0.5, s=20)
        plt.title(f'{split.upper()}: Absolute Error vs Hs')
        plt.xlabel('Significant Wave Height (m)')
        plt.ylabel('Abs Error (deg)')
        self._save_fig(output_dir, f"error_vs_hs_scatter_{split}")
        
        # Boxplot by Hs Bin (if available)
        if 'hs_bin' in df.columns:
            # Ensure bins are proper categorical labels
            df['hs_bin'] = df['hs_bin'].astype(str)

            plt.figure(figsize=(12, 6))

            # FIXED: Removed deprecated hue parameter that was causing warnings
            sns.boxplot(
                data=df,
                x='hs_bin',
                y='abs_error',
                palette="Blues"
            )

            plt.title(f'{split.upper()}: Error Distribution by Hs Bin')
            plt.xlabel('Hs Bin')
            plt.ylabel('Abs Error (deg)')
            self._save_fig(output_dir, f"error_vs_hs_bin_{split}")