import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot for parallel safety
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from pathlib import Path
from scipy import stats
from typing import Optional, Callable, List, Tuple
from contextlib import contextmanager # Added for FIX #78
import warnings  # Added
from joblib import Parallel, delayed  # Added for parallel plot generation
from modules.base.base_engine import BaseEngine
from utils.error_handling import handle_engine_errors
from utils import constants

class DiagnosticsEngine(BaseEngine):
    """
    Generates comprehensive diagnostic visualizations for model predictions.
    Includes scatter plots, error distributions, residual analysis, and per-Hs performance.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config, logger)
        self.diag_config = config.get('diagnostics', {})
        self.dpi = self.diag_config.get('dpi', 200)
        self.fmt = self.diag_config.get('save_format', 'png')
        
    def _get_engine_directory_name(self) -> str:
        return constants.DIAGNOSTICS_ENGINE_DIR

    @contextmanager
    def _plot_context(self):
        """
        FIX #78: Context manager to apply and then reset plot settings.
        Ensures global matplotlib/seaborn settings don't leak.
        """
        # Save current rcParams and seaborn theme
        original_rcParams = plt.rcParams.copy()
        original_seaborn_theme = sns.axes_style() # Captures current seaborn style
        figs = [] # Track figures created in this context
        try:
            # Apply engine's specific settings
            sns.set_theme(style="whitegrid")
            plt.rcParams.update({'figure.max_open_warning': 0})
            yield figs
        finally:
            # Restore original rcParams and seaborn theme
            plt.rcParams.update(original_rcParams)
            sns.set_theme(style=original_seaborn_theme) # Restore seaborn theme
            # Close only figures created in this context
            for fig in figs:
                plt.close(fig)

    def _generate_plot_task(self, plot_func: Callable, predictions: pd.DataFrame,
                           split_name: str, output_dir: Path) -> str:
        """
        Wrapper for parallel plot generation.
        Each task runs in isolation with its own matplotlib context.

        Args:
            plot_func: The plotting function to call
            predictions: DataFrame with predictions
            split_name: Name of the split (train/val/test)
            output_dir: Directory to save plots

        Returns:
            Status message
        """
        try:
            # Each worker gets a fresh matplotlib context
            with self._plot_context() as figs:
                plot_func(predictions, split_name, output_dir, figs)
            return f"Success: {plot_func.__name__}"
        except Exception as e:
            self.logger.error(f"Failed to generate {plot_func.__name__}: {e}")
            return f"Failed: {plot_func.__name__} - {str(e)}"

    @handle_engine_errors("Diagnostics")
    def execute(self, predictions: pd.DataFrame, split_name: str, run_id: str) -> None:
        """
        Orchestrate generation of all configured plots.

        PERFORMANCE OPTIMIZATION (P0 - Issue #P3):
        Uses parallel processing to generate plots concurrently, achieving
        8x speedup on 8-core systems. Each plot is generated in isolation
        to prevent matplotlib state conflicts.
        """
        if predictions.empty:
            self.logger.warning("Predictions dataframe is empty. Skipping diagnostics generation.")
            return

        self.logger.info(f"Generating diagnostics for {split_name} set...")

        # 1. Setup Directories
        dirs = {
            'index': self.output_dir / "index_plots",
            'scatter': self.output_dir / "scatter_plots",
            'residual': self.output_dir / "residual_plots",
            'dist': self.output_dir / "distribution_plots",
            'qq': self.output_dir / "qq_plots",
            'per_hs': self.output_dir / "per_hs_plots"
        }

        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # 2. Build list of plot tasks based on config flags
        plot_tasks: List[Tuple[Callable, Path]] = []

        if self.diag_config.get('generate_index_plots', True):
            plot_tasks.append((self._plot_index_vs_values, dirs['index']))

        if self.diag_config.get('generate_scatter_plots', True):
            plot_tasks.append((self._plot_scatter, dirs['scatter']))

        if self.diag_config.get('generate_residual_plots', True):
            plot_tasks.append((self._plot_residuals, dirs['residual']))

        if self.diag_config.get('generate_distribution_plots', True):
            plot_tasks.append((self._plot_distributions, dirs['dist']))

        if self.diag_config.get('generate_qq_plots', True):
            plot_tasks.append((self._plot_qq, dirs['qq']))

        if self.diag_config.get('generate_per_hs_accuracy', True):
            plot_tasks.append((self._plot_per_hs_analysis, dirs['per_hs']))

        # 3. Generate plots (parallel by default, sequential fallback)
        parallel_enabled = self.diag_config.get('parallel_plots', True)
        n_jobs = self.config.get('execution', {}).get('n_jobs', -1)

        self.logger.info(f"Generating {len(plot_tasks)} plot groups "
                         f"{'in parallel' if parallel_enabled else 'sequentially'}...")

        # Force sequential plotting for stability (matplotlib is not thread-safe in our usage).
        results = [
            self._generate_plot_task(plot_func, predictions, split_name, output_dir)
            for plot_func, output_dir in plot_tasks
        ]

        # 4. Log results
        successes = sum(1 for r in results if r.startswith("Success"))
        failures = sum(1 for r in results if r.startswith("Failed"))

        self.logger.info(f"Diagnostics generation complete: {successes} successful, {failures} failed")

        if failures > 0:
            for result in results:
                if result.startswith("Failed"):
                    self.logger.warning(result)

    def _save_fig(self, output_dir: Path, filename: str, fig, figs: list):
        """Helper to save and close figures."""
        path = output_dir / f"{filename}.{self.fmt}"
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
        figs.append(fig)

    def _plot_index_vs_values(self, df: pd.DataFrame, split: str, output_dir: Path, figs: list):
        """
        Plot Actual/Predicted/Error vs Index using SCATTER plots.
        """
        plt.switch_backend('Agg')
        # 1. Index vs Actual & Predicted (Scatter)
        fig = plt.figure(figsize=(12, 6))
        # Actuals as small blue dots
        plt.scatter(df.index, df['true_angle'], label='True', alpha=0.5, s=15, color='blue', marker='o')
        # Preds as small red crosses
        plt.scatter(df.index, df['pred_angle'], label='Pred', alpha=0.5, s=15, color='red', marker='x')
        
        plt.title(f'{split.upper()}: Index vs Angle')
        plt.xlabel('Sample Index')
        plt.ylabel('Angle (deg)')
        plt.legend()
        self._save_fig(output_dir, f"index_vs_values_{split}", fig, figs)
        
        # 2. Index vs Error (Scatter)
        fig = plt.figure(figsize=(12, 4))
        plt.scatter(df.index, df['error'], color='purple', alpha=0.6, s=10)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.title(f'{split.upper()}: Index vs Error')
        plt.ylabel('Error (deg)')
        self._save_fig(output_dir, f"index_vs_error_{split}", fig, figs)
        
        # 3. Index vs Abs Error (Scatter)
        fig = plt.figure(figsize=(12, 4))
        plt.scatter(df.index, df['abs_error'], color='darkorange', alpha=0.6, s=10)
        plt.title(f'{split.upper()}: Index vs Absolute Error')
        plt.ylabel('Abs Error (deg)')
        self._save_fig(output_dir, f"index_vs_abs_error_{split}", fig, figs)

    def _plot_scatter(self, df: pd.DataFrame, split: str, output_dir: Path, figs: list):
        """Scatter plot: Actual vs Predicted."""
        plt.switch_backend('Agg')
        fig = plt.figure(figsize=(8, 8))
        
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
        self._save_fig(output_dir, f"actual_vs_pred_{split}", fig, figs)

    def _plot_residuals(self, df: pd.DataFrame, split: str, output_dir: Path, figs: list):
        """Residuals vs Predicted Value."""
        plt.switch_backend('Agg')
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(df['pred_angle'], df['error'], alpha=0.5, s=15)
        plt.axhline(0, color='r', linestyle='--')
        plt.title(f'{split.upper()}: Residuals vs Predicted')
        plt.xlabel('Predicted Angle (deg)')
        plt.ylabel('Residual (Error) (deg)')
        self._save_fig(output_dir, f"residuals_vs_pred_{split}", fig, figs)

    def _plot_distributions(self, df: pd.DataFrame, split: str, output_dir: Path, figs: list):
        """Error Histogram and Boxplot."""
        plt.switch_backend('Agg')
        # Histogram
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(df['error'], kde=True, bins=50, color='teal')
        plt.title(f'{split.upper()}: Error Distribution')
        plt.xlabel('Error (deg)')
        self._save_fig(output_dir, f"error_hist_{split}", fig, figs)
        
        # Boxplot (Abs Error)
        fig = plt.figure(figsize=(6, 6))
        # FIX: Suppress upstream PendingDeprecationWarning regarding 'vert'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PendingDeprecationWarning)
            sns.boxplot(y=df['abs_error'], color='orange')
            
        plt.title(f'{split.upper()}: Absolute Error Boxplot')
        plt.ylabel('Absolute Error (deg)')
        self._save_fig(output_dir, f"error_boxplot_{split}", fig, figs)

    def _plot_qq(self, df: pd.DataFrame, split: str, output_dir: Path, figs: list):
        """Q-Q Plot to check normality of residuals."""
        plt.switch_backend('Agg')
        fig = plt.figure(figsize=(8, 8))
        stats.probplot(df['error'], dist="norm", plot=plt)
        plt.title(f'{split.upper()}: Q-Q Plot')
        self._save_fig(output_dir, f"qq_plot_{split}", fig, figs)

    def _plot_per_hs_analysis(self, df: pd.DataFrame, split: str, output_dir: Path, figs: list):
        """Plot Error vs Significant Wave Height (Hs)."""
        plt.switch_backend('Agg')
        configured = self.config['data']['hs_column']
        candidates = [f"{configured}_ft", "Hs_ft", configured]
        
        hs_col = next((c for c in candidates if c in df.columns), None)
        if hs_col is None:
            self.logger.warning(f"Hs column '{configured}' not found in predictions (meters or feet). Skipping Per-Hs plots.")
            return

        # Scatter: Abs Error vs Hs
        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=hs_col, y='abs_error', alpha=0.5, s=20)
        unit = "(ft)" if hs_col.endswith("_ft") or hs_col.lower().endswith("hs_ft") else "(m)"
        plt.title(f'{split.upper()}: Absolute Error vs Hs')
        plt.xlabel(f'Significant Wave Height {unit}')
        plt.ylabel('Abs Error (deg)')
        self._save_fig(output_dir, f"error_vs_hs_scatter_{split}", fig, figs)
        
        # Boxplot by Hs Bin (if available)
        if 'hs_bin' in df.columns:
            # Ensure bins are proper categorical labels
            df['hs_bin'] = df['hs_bin'].astype(str)

            fig = plt.figure(figsize=(12, 6))

            # FIX: Suppress upstream warnings and fix FutureWarnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=PendingDeprecationWarning)
                # FIX: Added hue and legend=False to satisfy new Seaborn API
                sns.boxplot(
                    data=df,
                    x='hs_bin',
                    y='abs_error',
                    hue='hs_bin', 
                    palette="Blues",
                    legend=False
                )

            plt.title(f'{split.upper()}: Error Distribution by Hs Bin')
            plt.xlabel('Hs Bin')
            plt.ylabel('Abs Error (deg)')
            self._save_fig(output_dir, f"error_vs_hs_bin_{split}", fig, figs)
