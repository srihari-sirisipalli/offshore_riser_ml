"""
Advanced Visualizations Module
Implements the top 10 critical missing visualizations from audit.

Covers:
1. 3D Error Surface Maps
2. Optimal Performance Zone Maps
3. Error Response Curves (Hs and Angle)
4. Faceted Multi-Panel Views
5. High-Error Region Zooms
6. Round Progression Tracking
7. Delta/Improvement Visualizations
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable, List
from scipy.interpolate import griddata
import warnings
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore', category=UserWarning)


class AdvancedVisualizer:
    """
    Generates advanced publication-ready visualizations.
    """

    # Standard color schemes
    ERROR_COLORS = {
        'excellent': '#006400',  # Dark Green (0-3°)
        'good': '#90EE90',      # Light Green (3-5°)
        'acceptable': '#FFFF00', # Yellow (5-10°)
        'caution': '#FFA500',   # Orange (10-15°)
        'critical': '#FF0000'    # Red (15+°)
    }

    DATASET_COLORS = {
        'test': '#1f77b4',   # Blue
        'val': '#ff7f0e',    # Orange
        'train': '#2ca02c'   # Green
    }

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.dpi'] = 150  # High resolution
        vis_cfg = config.get("visualization", {})
        self.parallel = vis_cfg.get("parallel_plots", False)
        self.n_jobs = config.get("execution", {}).get("n_jobs", -1)

    # ------------------------------------------------------------------ #
    # Batch runner to orchestrate multiple plots with optional threading #
    # ------------------------------------------------------------------ #
    def run_batch(self, tasks: List[Tuple[str, Callable[[], None]]]) -> None:
        """
        Execute a list of plotting tasks, optionally in parallel.

        Each task is a tuple of (name, zero-arg callable). Parallelism uses
        a threading backend for matplotlib safety. Fallback to sequential if
        disabled or only one task provided.
        """
        if not tasks:
            return

        def _safe(name: str, fn: Callable[[], None]) -> str:
            try:
                plt.switch_backend("Agg")
                fn()
                return f"Success:{name}"
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"Advanced viz failed '{name}': {exc}")
                return f"Failed:{name}"

        if self.parallel and len(tasks) > 1:
            results = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(_safe)(name, fn) for name, fn in tasks
            )
        else:
            results = [_safe(name, fn) for name, fn in tasks]

        failures = [r for r in results if r.startswith("Failed")]
        if failures:
            self.logger.warning(f"{len(failures)} advanced visualization task(s) failed: {failures}")

    # ------------------------------------------------------------------ #
    # Convenience: run a standard suite of advanced plots for a split    #
    # ------------------------------------------------------------------ #
    def run_default_suite(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        split_name: str = "test",
        hs_col: str = "Hs_ft",
        metrics_history: Optional[pd.DataFrame] = None,
        comparison_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Generate the standard advanced visualization set for a given split.

        Args:
            df: DataFrame with columns [hs_col, true_angle, abs_error, pred_angle] (as needed).
            output_dir: Base directory to write plots into.
            split_name: Identifier for titles and filenames.
            hs_col: Column name for significant wave height.
            metrics_history: Optional metrics DataFrame for round progression plot.
            comparison_df: Optional DataFrame with model/scenario column for delta grids.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tasks: List[Tuple[str, Callable[[], None]]] = []

        tasks.append((
            "error_surface_3d",
            lambda: self.plot_error_surface_3d(df, output_dir / f"error_surface_3d_{split_name}.png", hs_col=hs_col, split_name=split_name),
        ))
        tasks.append((
            "optimal_zone_map",
            lambda: self.plot_optimal_zone_map(df, output_dir / f"optimal_zone_map_{split_name}.png", hs_col=hs_col, split_name=split_name),
        ))
        tasks.append((
            "error_vs_hs_response",
            lambda: self.plot_error_vs_hs_response(df.copy(), output_dir / f"error_vs_hs_response_{split_name}.png", hs_col=hs_col, split_name=split_name),
        ))
        tasks.append((
            "circular_error_vs_angle",
            lambda: self.plot_circular_error_vs_angle(df, output_dir / f"circular_error_vs_angle_{split_name}.png", split_name=split_name),
        ))
        tasks.append((
            "faceted_error_by_hs_bins",
            lambda: self.plot_faceted_error_by_hs_bins(df.copy(), output_dir / f"faceted_error_by_hs_bins_{split_name}.png", hs_col=hs_col, split_name=split_name),
        ))
        tasks.append((
            "faceted_error_by_angle_bins",
            lambda: self.plot_faceted_error_by_angle_bins(df.copy(), output_dir / f"faceted_error_by_angle_bins_{split_name}.png", split_name=split_name),
        ))

        # Optional additional plots (if required columns present)
        tasks.append((
            "residual_diagnostics",
            lambda: self.plot_residual_diagnostics(df, output_dir / f"residual_diagnostics_{split_name}.png"),
        ))
        tasks.append((
            "boundary_analysis",
            lambda: self.plot_boundary_analysis(df, output_dir / f"boundary_analysis_{split_name}.png"),
        ))
        tasks.append((
            "error_distribution_by_hs_bins",
            lambda: self.plot_error_distribution_by_hs_bins(df, output_dir / f"error_distribution_by_hs_bins_{split_name}.png", hs_col=hs_col),
        ))
        tasks.append((
            "performance_contour_map",
            lambda: self.plot_performance_contour_map(df, output_dir / f"performance_contour_map_{split_name}.png", hs_col=hs_col),
        ))
        tasks.append((
            "high_error_zoom",
            lambda: self.plot_high_error_zoom(df, output_dir / f"high_error_zoom_{split_name}.png", hs_col=hs_col, top_n=50),
        ))
        tasks.append((
            "high_error_zoom_facets",
            lambda: self.plot_high_error_zoom_facets(df, output_dir / f"high_error_zoom_facets_{split_name}.png", hs_col=hs_col, top_n=100),
        ))
        tasks.append((
            "boundary_gradient",
            lambda: self.plot_boundary_gradient(df, output_dir / f"boundary_gradient_{split_name}.png"),
        ))
        tasks.append((
            "filtered_error_dashboard",
            lambda: self.plot_filtered_error_dashboard(df, output_dir / f"filtered_error_dashboard_{split_name}.png", hs_col=hs_col),
        ))
        tasks.append((
            "operating_envelope_overlay",
            lambda: self.plot_operating_envelope_overlay(df, output_dir / f"operating_envelope_overlay_{split_name}.png", hs_col=hs_col),
        ))
        if comparison_df is not None:
            tasks.append((
                "faceted_delta_grid",
                lambda: self.plot_faceted_delta_grid(comparison_df, output_dir / f"faceted_delta_grid_{split_name}.png"),
            ))
        tasks.append((
            "cluster_evolution_overlay",
            lambda: self.plot_cluster_evolution_overlay(df, output_dir / f"cluster_evolution_overlay_{split_name}.png"),
        ))

        if metrics_history is not None and not metrics_history.empty:
            tasks.append((
                "round_progression",
                lambda: self.plot_round_progression(metrics_history, output_dir / f"round_progression_{split_name}.png"),
            ))

        self.run_batch(tasks)

    # =========================================================================
    # #1: 3D ERROR SURFACE MAP
    # =========================================================================
    def plot_error_surface_3d(self, df: pd.DataFrame, output_path: Path,
                               hs_col: str = 'Hs_ft', split_name: str = 'test'):
        """
        Creates 3D surface plot of error landscape over (Hs, Angle) space.

        Args:
            df: DataFrame with columns [Hs_ft, true_angle, abs_error]
            output_path: Where to save the plot
            hs_col: Name of Hs column
            split_name: test/val/train for title
        """
        self.logger.info(f"Generating 3D error surface for {split_name}...")

        if hs_col not in df.columns or 'true_angle' not in df.columns or 'abs_error' not in df.columns:
            self.logger.warning(f"Missing required columns for 3D surface. Skipping.")
            return

        # Prepare data
        hs = df[hs_col].values
        angle = df['true_angle'].values
        error = df['abs_error'].values

        # Create grid for interpolation
        grid_hs, grid_angle = np.mgrid[
            hs.min():hs.max():50j,
            angle.min():angle.max():50j
        ]

        # Interpolate error surface
        grid_error = griddata(
            (hs, angle), error,
            (grid_hs, grid_angle),
            method='cubic'
        )

        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Surface plot
        surf = ax.plot_surface(
            grid_hs, grid_angle, grid_error,
            cmap='RdYlGn_r',  # Red (high error) to Green (low error)
            alpha=0.8,
            edgecolor='none'
        )

        # Contour lines at key error thresholds
        contours = ax.contour(
            grid_hs, grid_angle, grid_error,
            levels=[5, 10, 15],
            colors='black',
            linewidths=1,
            alpha=0.5
        )

        # Labels and title
        ax.set_xlabel('Hs (ft)', fontsize=11)
        ax.set_ylabel('Angle (degrees)', fontsize=11)
        ax.set_zlabel('Absolute Error (degrees)', fontsize=11)
        ax.set_title(
            f'3D Error Surface Map - {split_name.upper()} Set\n'
            f'N={len(df):,} points',
            fontsize=13,
            fontweight='bold'
        )

        # Colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Error (deg)')

        # Set viewing angle
        ax.view_init(elev=25, azim=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved 3D surface to {output_path}")

    # =========================================================================
    # #2: OPTIMAL PERFORMANCE ZONE MAP (2D)
    # =========================================================================
    def plot_optimal_zone_map(self, df: pd.DataFrame, output_path: Path,
                               hs_col: str = 'Hs_ft', split_name: str = 'test'):
        """
        2D contour map with colored zones indicating where model is reliable.
        """
        self.logger.info(f"Generating optimal performance zone map for {split_name}...")

        if hs_col not in df.columns or 'true_angle' not in df.columns or 'abs_error' not in df.columns:
            self.logger.warning(f"Missing required columns for zone map. Skipping.")
            return

        # Prepare data
        hs = df[hs_col].values
        angle = df['true_angle'].values
        error = df['abs_error'].values

        # Create grid
        grid_hs, grid_angle = np.mgrid[
            hs.min():hs.max():100j,
            angle.min():angle.max():100j
        ]

        # Interpolate
        grid_error = griddata(
            (hs, angle), error,
            (grid_hs, grid_angle),
            method='cubic'
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Filled contour with custom levels and colors
        levels = [0, 3, 5, 10, 15, np.inf]
        colors = ['#006400', '#90EE90', '#FFFF00', '#FFA500', '#FF0000']

        contourf = ax.contourf(
            grid_hs, grid_angle, grid_error,
            levels=levels,
            colors=colors,
            alpha=0.7
        )

        # Contour lines
        contour = ax.contour(
            grid_hs, grid_angle, grid_error,
            levels=[3, 5, 10, 15],
            colors='black',
            linewidths=1.5,
            alpha=0.8
        )
        ax.clabel(contour, inline=True, fontsize=9, fmt='%1.0f°')

        # Scatter actual data points
        ax.scatter(hs, angle, c='white', s=1, alpha=0.3, edgecolors='none')

        # Labels
        ax.set_xlabel('Hs (ft)', fontsize=12)
        ax.set_ylabel('Angle (degrees)', fontsize=12)
        ax.set_title(
            f'Optimal Performance Zone Map - {split_name.upper()} Set\n'
            f'Green: Excellent (<3°) | Yellow: Acceptable (5-10°) | Red: Critical (>15°)',
            fontsize=13,
            fontweight='bold'
        )

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#006400', label='Excellent (<3°)'),
            Patch(facecolor='#90EE90', label='Good (3-5°)'),
            Patch(facecolor='#FFFF00', label='Acceptable (5-10°)'),
            Patch(facecolor='#FFA500', label='Caution (10-15°)'),
            Patch(facecolor='#FF0000', label='Critical (>15°)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved optimal zone map to {output_path}")

    # =========================================================================
    # #3: ERROR VS HS RESPONSE CURVE
    # =========================================================================
    def plot_error_vs_hs_response(self, df: pd.DataFrame, output_path: Path,
                                   hs_col: str = 'Hs_ft', split_name: str = 'test'):
        """
        Detailed response curve showing how error varies with Hs.
        Includes mean, std bands, percentiles, and annotations.
        """
        self.logger.info(f"Generating error vs Hs response curve for {split_name}...")

        if hs_col not in df.columns or 'abs_error' not in df.columns:
            self.logger.warning(f"Missing required columns. Skipping.")
            return

        # Bin Hs into intervals for aggregation
        # Ensure bins cover the full range and have sufficient steps
        min_hs = np.floor(df[hs_col].min())
        max_hs = np.ceil(df[hs_col].max())
        hs_bins = np.arange(min_hs, max_hs + 1, 1)

        # Ensure there are at least two bins
        if len(hs_bins) < 2:
            # If range is too small, create a default set of bins with broader range
            hs_bins = np.linspace(min_hs, max_hs + 1, 5) # Create 4 bin edges for 3 bins
            if len(hs_bins) < 2: # Still not enough bins (e.g. all data points are same)
                self.logger.warning(f"Insufficient range for Hs_ft to create meaningful bins. Skipping error vs Hs response plot.")
                return

        # Ensure bins are unique and sorted
        hs_bins = np.unique(hs_bins)
        if len(hs_bins) < 2:
            self.logger.warning(f"Could not create valid bins for Hs_ft. Skipping error vs Hs response plot.")
            return

        df['hs_bin'] = pd.cut(df[hs_col], bins=hs_bins, include_lowest=True, right=True)

        # Handle cases where some bins might be empty after cutting
        df = df.dropna(subset=['hs_bin'])
        if df.empty:
            self.logger.warning(f"No data points left after binning Hs_ft. Skipping error vs Hs response plot.")
            return

        # Aggregate statistics
        agg = df.groupby('hs_bin', observed=False)['abs_error'].agg(['mean', 'std', 'count']) # observed=False for future compatibility
        agg['hs_center'] = [interval.mid for interval in agg.index]
        agg = agg.dropna()
        if agg.empty: # If all bins are empty or dropna makes it empty
             self.logger.warning(f"Aggregated data for Hs_ft is empty. Skipping error vs Hs response plot.")
             return

        # Percentiles - ensure they are based on the binned df
        # Percentiles - calculate from the same grouped data as agg, and ensure index alignment
        p25 = df.groupby('hs_bin', observed=False)['abs_error'].quantile(0.25).reindex(agg.index)
        p75 = df.groupby('hs_bin', observed=False)['abs_error'].quantile(0.75).reindex(agg.index)
        
        # Fill NaN for percentiles if any bin in agg had no corresponding percentile (unlikely if agg is not empty)
        p25 = p25.ffill().bfill()
        p75 = p75.ffill().bfill()

        if p25.empty or p75.empty:
            self.logger.warning(f"Percentile data for Hs_ft is empty. Skipping error vs Hs response plot.")
            return

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 7))

        # Scatter raw data
        ax.scatter(df[hs_col], df['abs_error'], alpha=0.1, s=10, color='gray', label='Raw Data')

        # Mean line
        ax.plot(agg['hs_center'], agg['mean'], linewidth=2.5, color='blue', label='Mean Error')

        # Std band
        ax.fill_between(
            agg['hs_center'],
            agg['mean'] - agg['std'],
            agg['mean'] + agg['std'],
            alpha=0.3,
            color='blue',
            label='± 1 Std Dev'
        )

        # Percentile lines
        ax.plot(agg['hs_center'], p25, linestyle='--', color='green', alpha=0.7, label='25th Percentile')
        ax.plot(agg['hs_center'], p75, linestyle='--', color='red', alpha=0.7, label='75th Percentile')

        # Threshold lines
        ax.axhline(5, color='yellow', linestyle=':', linewidth=2, label='Acceptable Threshold (5°)')
        ax.axhline(10, color='orange', linestyle=':', linewidth=2, label='Warning Threshold (10°)')
        ax.axhline(15, color='red', linestyle=':', linewidth=2, label='Critical Threshold (15°)')

        # Find optimal Hs range (lowest mean error)
        if len(agg) > 0:
            optimal_idx = agg['mean'].idxmin()
            optimal_hs = agg.loc[optimal_idx, 'hs_center']
            optimal_error = agg.loc[optimal_idx, 'mean']

            ax.axvline(optimal_hs, color='green', linestyle='-.', alpha=0.5, linewidth=2)
            ax.annotate(
                f'Optimal Zone\nHs ≈ {optimal_hs:.1f} ft\nError = {optimal_error:.2f}°',
                xy=(optimal_hs, optimal_error),
                xytext=(optimal_hs + 2, optimal_error + 3),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                fontsize=10
            )

        # Labels
        ax.set_xlabel('Hs (ft)', fontsize=12)
        ax.set_ylabel('Absolute Error (degrees)', fontsize=12)
        ax.set_title(
            f'Error vs Hs Response Curve - {split_name.upper()} Set\n'
            f'N={len(df):,} points',
            fontsize=13,
            fontweight='bold'
        )
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, min(30, df['abs_error'].quantile(0.99)))

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved error vs Hs curve to {output_path}")

    # =========================================================================
    # #4: CIRCULAR ERROR VS ANGLE PLOT
    # =========================================================================
    def plot_circular_error_vs_angle(self, df: pd.DataFrame, output_path: Path,
                                      split_name: str = 'test'):
        """
        Polar plot showing how error varies by angle direction.
        """
        self.logger.info(f"Generating circular error vs angle plot for {split_name}...")

        if 'true_angle' not in df.columns or 'abs_error' not in df.columns:
            self.logger.warning(f"Missing required columns. Skipping.")
            return

        # Bin angles into sectors (8 directions)
        angle_bins = np.arange(0, 361, 45)
        df['angle_sector'] = pd.cut(df['true_angle'], bins=angle_bins)

        # Aggregate
        agg = df.groupby('angle_sector', observed=False)['abs_error'].agg(['mean', 'count'])
        agg['angle_mid'] = [interval.mid for interval in agg.index]
        agg = agg.dropna()

        # Convert to radians
        theta = np.deg2rad(agg['angle_mid'].values)
        radii = agg['mean'].values

        # Create polar plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

        # Bar chart
        bars = ax.bar(theta, radii, width=np.deg2rad(45), alpha=0.7, edgecolor='black', linewidth=1.5)

        # Color code by error magnitude
        for bar, r in zip(bars, radii):
            if r < 5:
                bar.set_facecolor('#2ca02c')  # Green
            elif r < 10:
                bar.set_facecolor('#FFFF00')  # Yellow
            else:
                bar.set_facecolor('#FF0000')  # Red

        # Reference circles
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)  # Clockwise

        # Labels
        direction_labels = ['N (0°)', 'NE (45°)', 'E (90°)', 'SE (135°)',
                            'S (180°)', 'SW (225°)', 'W (270°)', 'NW (315°)']
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
        ax.set_xticklabels(direction_labels, fontsize=10)

        ax.set_title(
            f'Error by Directional Sector - {split_name.upper()} Set\n'
            f'Green: <5° | Yellow: 5-10° | Red: >10°',
            fontsize=13,
            fontweight='bold',
            pad=20
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved circular angle plot to {output_path}")

    # =========================================================================
    # #5: FACETED ERROR BY HS BINS (8 PANELS)
    # =========================================================================
    def plot_faceted_error_by_hs_bins(self, df: pd.DataFrame, output_path: Path,
                                       hs_col: str = 'Hs_ft', split_name: str = 'test'):
        """
        8-panel faceted view showing error distributions for different Hs ranges.
        """
        self.logger.info(f"Generating faceted error by Hs bins for {split_name}...")

        if hs_col not in df.columns or 'abs_error' not in df.columns:
            self.logger.warning(f"Missing required columns. Skipping.")
            return

        # Define Hs bins
        hs_max = df[hs_col].max()
        bin_edges = np.linspace(df[hs_col].min(), hs_max, 9)  # 8 bins
        df['hs_bin'] = pd.cut(df[hs_col], bins=bin_edges)

        # Create faceted plot
        fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, (bin_label, group) in enumerate(df.groupby('hs_bin', observed=False)):
            if i >= 8:
                break

            ax = axes[i]

            if len(group) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_title(f'{bin_label}', fontsize=10)
                continue

            # Histogram
            ax.hist(group['abs_error'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)

            # Mean line
            mean_err = group['abs_error'].mean()
            ax.axvline(mean_err, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.2f}°')

            # Threshold line
            ax.axvline(10, color='red', linestyle=':', linewidth=2, label='Threshold: 10°')

            # Count high errors
            high_err_count = (group['abs_error'] > 10).sum()

            # Title and labels
            ax.set_title(f'{bin_label}\nN={len(group):,} | >10°: {high_err_count}', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Common labels
        fig.text(0.5, 0.02, 'Absolute Error (degrees)', ha='center', fontsize=12)
        fig.text(0.02, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=12)
        fig.suptitle(
            f'Error Distribution by Hs Bins - {split_name.upper()} Set',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )

        plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved faceted Hs bins plot to {output_path}")

    # =========================================================================
    # #6: FACETED ERROR BY ANGLE BINS (8 PANELS)
    # =========================================================================
    def plot_faceted_error_by_angle_bins(self, df: pd.DataFrame, output_path: Path,
                                          split_name: str = 'test'):
        """
        8-panel faceted view showing error distributions for 8 directional sectors.
        """
        self.logger.info(f"Generating faceted error by angle bins for {split_name}...")

        if 'true_angle' not in df.columns or 'abs_error' not in df.columns:
            self.logger.warning(f"Missing required columns. Skipping.")
            return

        # Define angle bins (8 sectors)
        angle_bins = np.arange(0, 361, 45)
        df['angle_sector'] = pd.cut(df['true_angle'], bins=angle_bins)

        sector_names = ['N (0-45°)', 'NE (45-90°)', 'E (90-135°)', 'SE (135-180°)',
                        'S (180-225°)', 'SW (225-270°)', 'W (270-315°)', 'NW (315-360°)']

        # Create faceted plot
        fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, (bin_label, group) in enumerate(df.groupby('angle_sector', observed=False)):
            if i >= 8:
                break

            ax = axes[i]

            if len(group) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_title(sector_names[i] if i < len(sector_names) else str(bin_label), fontsize=10)
                continue

            # Histogram
            ax.hist(group['abs_error'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)

            # Mean line
            mean_err = group['abs_error'].mean()
            ax.axvline(mean_err, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.2f}°')

            # Threshold line
            ax.axvline(10, color='red', linestyle=':', linewidth=2, label='Threshold: 10°')

            # Count high errors
            high_err_count = (group['abs_error'] > 10).sum()

            # Title and labels
            title = sector_names[i] if i < len(sector_names) else str(bin_label)
            ax.set_title(f'{title}\nN={len(group):,} | >10°: {high_err_count}', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Common labels
        fig.text(0.5, 0.02, 'Absolute Error (degrees)', ha='center', fontsize=12)
        fig.text(0.02, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=12)
        fig.suptitle(
            f'Error Distribution by Directional Sector - {split_name.upper()} Set',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )

        plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved faceted angle bins plot to {output_path}")

    # =========================================================================
    # #7: ROUND PROGRESSION PLOT
    # =========================================================================
    def plot_round_progression(self, metrics_history: pd.DataFrame, output_path: Path):
        """
        Line plot showing how metrics evolve across RFE rounds.

        Args:
            metrics_history: DataFrame with columns [round, metric_name, val_value, test_value]
        """
        self.logger.info("Generating round progression plot...")

        if 'round' not in metrics_history.columns:
            self.logger.warning("Missing 'round' column. Skipping.")
            return

        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Key metrics to track
        error_metrics = ['cmae', 'crmse', 'max_error']
        acc_metrics = ['accuracy_at_5deg', 'accuracy_at_10deg']

        # --- SUBPLOT 1: Error Metrics ---
        ax1 = axes[0]
        for metric in error_metrics:
            val_data = metrics_history[metrics_history['metric_name'] == f'val_{metric}']
            test_data = metrics_history[metrics_history['metric_name'] == f'test_{metric}']

            if not val_data.empty:
                ax1.plot(val_data['round'], val_data['value'], marker='o', label=f'{metric.upper()} Val', linewidth=2)
            if not test_data.empty:
                ax1.plot(test_data['round'], test_data['value'], marker='s', linestyle='--', label=f'{metric.upper()} Test', linewidth=2, alpha=0.7)

        ax1.set_ylabel('Error (degrees)', fontsize=12)
        ax1.set_title('Error Metrics Evolution Across Rounds (Lower is Better)', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # --- SUBPLOT 2: Accuracy Metrics ---
        ax2 = axes[1]
        for metric in acc_metrics:
            val_data = metrics_history[metrics_history['metric_name'] == f'val_{metric}']
            test_data = metrics_history[metrics_history['metric_name'] == f'test_{metric}']

            if not val_data.empty:
                ax2.plot(val_data['round'], val_data['value'], marker='o', label=f'{metric.upper()} Val', linewidth=2)
            if not test_data.empty:
                ax2.plot(test_data['round'], test_data['value'], marker='s', linestyle='--', label=f'{metric.upper()} Test', linewidth=2, alpha=0.7)

        ax2.set_xlabel('Round Number', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Accuracy Metrics Evolution Across Rounds (Higher is Better)', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved round progression plot to {output_path}")

    # =========================================================================
    # #8: Improvement Heatmap (Hs vs Angle)
    # =========================================================================
    def plot_improvement_heatmap(self, df: pd.DataFrame, output_path: Path, hs_col: str = 'Hs_ft'):
        """
        Heatmap of improvement = baseline_abs_error - candidate_abs_error across Hs/angle bins.
        Expects columns: true_angle, abs_error_baseline, abs_error_candidate, and hs_col.
        """
        required = {hs_col, 'true_angle', 'abs_error_baseline', 'abs_error_candidate'}
        if not required.issubset(df.columns):
            self.logger.warning("Improvement heatmap skipped: missing required columns.")
            return

        df = df.copy()
        df['improvement'] = df['abs_error_baseline'] - df['abs_error_candidate']

        hs_bins = np.linspace(df[hs_col].min(), df[hs_col].max(), 10)
        angle_bins = np.linspace(0, 360, 13)

        df['hs_bin'] = pd.cut(df[hs_col], bins=hs_bins)
        df['angle_bin'] = pd.cut(df['true_angle'], bins=angle_bins)

        pivot = df.pivot_table(index='angle_bin', columns='hs_bin', values='improvement', aggfunc='mean')
        if pivot.empty:
            self.logger.warning("Improvement heatmap skipped: empty pivot.")
            return

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, cmap='RdYlGn', center=0, annot=False)
        plt.title("Improvement Heatmap (Baseline - Candidate Abs Error)\nGreen = Improvement, Red = Degradation")
        plt.xlabel("Hs bin")
        plt.ylabel("Angle bin")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved improvement heatmap to {output_path}")

    # =========================================================================
    # #9: Residual Diagnostics (4-panel)
    # =========================================================================
    def plot_residual_diagnostics(self, df: pd.DataFrame, output_path: Path):
        """
        Four-panel residual diagnostics:
        1) Residuals vs Fitted
        2) Normal Q-Q
        3) Scale-Location
        4) Residuals vs Leverage (proxy leverage: ranked index)
        Expects columns: pred_angle, true_angle, error, abs_error.
        """
        required = {'pred_angle', 'true_angle', 'error', 'abs_error'}
        if not required.issubset(df.columns):
            self.logger.warning("Residual diagnostics skipped: missing required columns.")
            return

        residuals = df['error'].values
        fitted = df['pred_angle'].values

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(fitted, residuals, alpha=0.4, color="#1f77b4")
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        ax1.set_title("Residuals vs Fitted")
        ax1.set_xlabel("Fitted (pred_angle)")
        ax1.set_ylabel("Residuals")

        ax2 = fig.add_subplot(gs[0, 1])
        sorted_resid = np.sort(residuals)
        quantiles = np.linspace(0, 1, len(sorted_resid), endpoint=False)
        try:
            # Import stats locally to avoid circular dependencies if stats is also importing from advanced_viz.
            # Or ensure scipy is available globaly
            from scipy import stats, special # Import special here
            norm_quantiles = np.sqrt(2) * special.erfinv(2 * quantiles - 1) # Use special.erfinv
        except Exception as exc:
            self.logger.warning(f"Failed to calculate normal quantiles for residual diagnostics: {exc}. Skipping plot.")
            plt.close(fig) # Close the figure to avoid resource leak
            return # Exit function early
        ax2.scatter(norm_quantiles, sorted_resid, alpha=0.5, color="#ff7f0e")
        ax2.plot([norm_quantiles.min(), norm_quantiles.max()],
                 [sorted_resid.min(), sorted_resid.max()], 'k--', linewidth=1)
        ax2.set_title("Normal Q-Q")
        ax2.set_xlabel("Theoretical Quantiles")
        ax2.set_ylabel("Residuals")

        ax3 = fig.add_subplot(gs[1, 0])
        scale_location = np.sqrt(np.abs(residuals))
        ax3.scatter(fitted, scale_location, alpha=0.4, color="#2ca02c")
        ax3.set_title("Scale-Location")
        ax3.set_xlabel("Fitted (pred_angle)")
        ax3.set_ylabel("Sqrt(|Residuals|)")

        ax4 = fig.add_subplot(gs[1, 1])
        leverage_proxy = np.arange(len(df)) / len(df)
        ax4.scatter(leverage_proxy, residuals, alpha=0.4, color="#9467bd")
        ax4.axhline(0, color='black', linestyle='--', linewidth=1)
        ax4.set_title("Residuals vs Leverage (proxy)")
        ax4.set_xlabel("Leverage proxy (ranked index)")
        ax4.set_ylabel("Residuals")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved residual diagnostics to {output_path}")

    # =========================================================================
    # #10: Boundary Region Analysis (0/360 wrap)
    # =========================================================================
    def plot_boundary_analysis(self, df: pd.DataFrame, output_path: Path):
        """
        Focused view near circular boundary (0/360). Expects true_angle, pred_angle, abs_error.
        """
        required = {'true_angle', 'pred_angle', 'abs_error'}
        if not required.issubset(df.columns):
            self.logger.warning("Boundary analysis skipped: missing required columns.")
            return

        df_boundary = df[(df['true_angle'] <= 10) | (df['true_angle'] >= 350)]
        if df_boundary.empty:
            self.logger.warning("Boundary analysis skipped: no boundary samples.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(df_boundary['true_angle'], df_boundary['pred_angle'],
                        c=df_boundary['abs_error'], cmap='viridis', alpha=0.7)
        axes[0].set_title("True vs Predicted near boundary")
        axes[0].set_xlabel("True Angle (deg)")
        axes[0].set_ylabel("Pred Angle (deg)")
        axes[0].axline((0, 0), slope=1, color='k', linestyle='--', linewidth=1)

        axes[1].scatter(df_boundary['true_angle'], df_boundary['abs_error'],
                        color="#d62728", alpha=0.7)
        axes[1].set_title("Abs Error near boundary")
        axes[1].set_xlabel("True Angle (deg)")
        axes[1].set_ylabel("Abs Error (deg)")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved boundary analysis to {output_path}")

    def plot_boundary_gradient(self, df: pd.DataFrame, output_path: Path):
        """
        Gradient map of errors around angle boundaries to spot wrap-around issues.
        """
        if df.empty or 'true_angle' not in df.columns or 'abs_error' not in df.columns:
            self.logger.warning("Boundary gradient skipped: missing true_angle/abs_error.")
            return

        plt.figure(figsize=(10, 4))
        sns.kdeplot(
            data=df,
            x="true_angle",
            y="abs_error",
            fill=True,
            cmap="mako",
            levels=30,
            thresh=0.05,
        )
        plt.axvline(0, color="red", linestyle="--", alpha=0.7)
        plt.axvline(360, color="red", linestyle="--", alpha=0.7)
        plt.title("Boundary Gradient (wrap-around sensitivity)")
        plt.xlabel("True Angle (deg)")
        plt.ylabel("Abs Error (deg)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved boundary gradient to {output_path}")

    # =========================================================================
    # #11: Cluster Characteristics Summary
    # =========================================================================
    def plot_cluster_summary(self, df: pd.DataFrame, output_path: Path, hs_col: str = 'Hs_ft'):
        """
        Scatter view colored by cluster_id (if present) or high-error flag.
        Expects columns: true_angle, abs_error, optional cluster_id, hs_col.
        """
        required = {'true_angle', 'abs_error'}
        if not required.issubset(df.columns):
            self.logger.warning("Cluster summary skipped: missing required columns.")
            return

        df = df.copy()
        has_cluster = 'cluster_id' in df.columns
        if not has_cluster:
            thresh = np.nanpercentile(df['abs_error'], 90)
            df['cluster_id'] = np.where(df['abs_error'] >= thresh, 'high_error', 'normal')

        fig = plt.figure(figsize=(12, 6))
        if hs_col in df.columns:
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            sc = ax.scatter(df[hs_col], df['true_angle'], df['abs_error'],
                            c=df['cluster_id'].astype('category').cat.codes,
                            cmap='tab20', alpha=0.7)
            ax.set_xlabel('Hs (ft)')
            ax.set_ylabel('True Angle')
            ax.set_zlabel('Abs Error')
            ax.set_title('Cluster Overview (3D)')
        else:
            ax = fig.add_subplot(1, 1, 1)
            sc = ax.scatter(df['true_angle'], df['abs_error'],
                            c=df['cluster_id'].astype('category').cat.codes,
                            cmap='tab20', alpha=0.7)
            ax.set_xlabel('True Angle')
            ax.set_ylabel('Abs Error')
            ax.set_title('Cluster Overview')

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved cluster summary to {output_path}")

    # =========================================================================
    # #12: Error Distribution by Hs Bins (violin)
    # =========================================================================
    def plot_error_distribution_by_hs_bins(self, df: pd.DataFrame, output_path: Path, hs_col: str = "Hs_ft"):
        """
        Violin plots of abs_error across Hs bins.
        """
        required = {hs_col, "abs_error"}
        if not required.issubset(df.columns):
            self.logger.warning("Error by Hs bins skipped: missing required columns.")
            return

        df = df.copy()
        bin_edges = np.linspace(df[hs_col].min(), df[hs_col].max(), 7)
        df["hs_bin"] = pd.cut(df[hs_col], bins=bin_edges, include_lowest=True)
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x="hs_bin", y="abs_error", inner="box", palette="viridis")
        plt.xticks(rotation=30, ha="right")
        plt.title("Abs Error Distribution by Hs Bin")
        plt.xlabel("Hs bin")
        plt.ylabel("Abs Error (deg)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved error distribution by Hs bins to {output_path}")

    # =========================================================================
    # #13: Performance Contour Map (Error Topography)
    # =========================================================================
    def plot_performance_contour_map(self, df: pd.DataFrame, output_path: Path, hs_col: str = "Hs_ft"):
        """
        Contour map of abs_error over (Hs, true_angle) space.
        """
        required = {hs_col, "true_angle", "abs_error"}
        if not required.issubset(df.columns):
            self.logger.warning("Performance contour map skipped: missing required columns.")
            return

        hs = df[hs_col].values
        ang = df["true_angle"].values
        err = df["abs_error"].values

        grid_hs, grid_angle = np.mgrid[
            hs.min():hs.max():100j,
            0:360:100j,
        ]
        grid_error = griddata((hs, ang), err, (grid_hs, grid_angle), method="linear")

        plt.figure(figsize=(12, 8))
        cs = plt.contourf(grid_hs, grid_angle, grid_error, levels=20, cmap="magma")
        plt.colorbar(cs, label="Abs Error (deg)")
        plt.xlabel("Hs (ft)")
        plt.ylabel("Angle (deg)")
        plt.title("Performance Contour Map (Abs Error)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved performance contour map to {output_path}")

    # =========================================================================
    # #13B: Faceted Delta Grid (baseline vs candidate)
    # =========================================================================
    def plot_faceted_delta_grid(self, df: pd.DataFrame, output_path: Path):
        """
        Faceted heatmap of delta errors between models across hs/angle bins.

        Expects columns: model (or scenario), hs_bin, angle_bin, abs_error.
        """
        required = {"hs_bin", "angle_bin", "abs_error"}
        if df is None or df.empty or not required.issubset(df.columns):
            self.logger.warning("Faceted delta grid skipped: missing hs_bin/angle_bin/abs_error.")
            return
        if "model" not in df.columns and "scenario" not in df.columns:
            self.logger.warning("Faceted delta grid skipped: missing model/scenario column.")
            return

        model_col = "model" if "model" in df.columns else "scenario"
        pivots = []
        for name, group in df.groupby(model_col):
            try:
                pivot = group.pivot_table(values="abs_error", index="hs_bin", columns="angle_bin", aggfunc="mean")
                pivots.append((name, pivot))
            except Exception:
                continue

        if len(pivots) < 2:
            self.logger.warning("Faceted delta grid skipped: need at least two models.")
            return

        base_name, base = pivots[0]
        n = len(pivots) - 1
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharex=False, sharey=False)
        if n == 1:
            axes = [axes]

        for ax, (name, mat) in zip(axes, pivots[1:]):
            common_index = base.index.intersection(mat.index)
            common_cols = base.columns.intersection(mat.columns)
            if common_index.empty or common_cols.empty:
                self.logger.warning(f"Delta grid skipped for {name}: no common bins.")
                continue
            delta = mat.loc[common_index, common_cols] - base.loc[common_index, common_cols]
            sns.heatmap(delta, ax=ax, cmap="coolwarm", center=0, cbar=True)
            ax.set_title(f"{name} vs {base_name} (Δ abs_error)")
            ax.set_xlabel("angle_bin")
            ax.set_ylabel("hs_bin")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved faceted delta grid to {output_path}")

    # =========================================================================
    # #14: Operating Envelope Diagram
    # =========================================================================
    def plot_operating_envelope(self, df: pd.DataFrame, output_path: Path, hs_col: str = "Hs_ft"):
        """
        Contour-style operating envelope: zones colored by error tiers with training density overlay.
        """
        required = {hs_col, "true_angle", "abs_error"}
        if not required.issubset(df.columns):
            self.logger.warning("Operating envelope skipped: missing required columns.")
            return

        hs = df[hs_col].values
        ang = df["true_angle"].values
        err = df["abs_error"].values

        grid_hs, grid_angle = np.mgrid[
            hs.min():hs.max():150j,
            0:360:150j,
        ]
        grid_error = griddata((hs, ang), err, (grid_hs, grid_angle), method="linear")

        levels = [0, 3, 5, 10, 15, np.inf]
        colors = ['#006400', '#90EE90', '#FFFF00', '#FFA500', '#FF0000']

        plt.figure(figsize=(12, 8))
        cf = plt.contourf(grid_hs, grid_angle, grid_error, levels=levels, colors=colors, alpha=0.7)
        cs = plt.contour(grid_hs, grid_angle, grid_error, levels=[3, 5, 10, 15], colors='k', linewidths=1, alpha=0.8)
        plt.clabel(cs, inline=True, fmt="%1.0f°")
        plt.scatter(hs, ang, c="white", s=4, alpha=0.15, edgecolors="none")
        plt.xlabel("Hs (ft)")
        plt.ylabel("Angle (deg)")
        plt.title("Operating Envelope (Error Tiers)")
        plt.colorbar(cf, label="Abs Error (deg)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved operating envelope to {output_path}")

    def plot_operating_envelope_overlay(self, df: pd.DataFrame, output_path: Path, hs_col: str = "Hs_ft"):
        """
        Overlay operating envelope tiers with high-error markers and density contours.
        """
        if df.empty or {hs_col, "true_angle", "abs_error"}.intersection(df.columns) != {hs_col, "true_angle", "abs_error"}:
            self.logger.warning("Operating envelope overlay skipped: missing required columns.")
            return

        plt.figure(figsize=(12, 6))
        try:
            sns.kdeplot(
                data=df,
                x=hs_col,
                y="true_angle",
                fill=True,
                cmap="Blues",
                alpha=0.5,
                levels=30,
            )
        except Exception:
            pass

        tiers = [
            (df["abs_error"] <= 3, "#006400", "excellent"),
            ((df["abs_error"] > 3) & (df["abs_error"] <= 5), "#90EE90", "good"),
            ((df["abs_error"] > 5) & (df["abs_error"] <= 10), "#FFD700", "acceptable"),
            ((df["abs_error"] > 10) & (df["abs_error"] <= 15), "#FFA500", "caution"),
            (df["abs_error"] > 15, "#FF0000", "critical"),
        ]
        for mask, color, label in tiers:
            subset = df[mask]
            if subset.empty:
                continue
            plt.scatter(subset[hs_col], subset["true_angle"], s=12, alpha=0.6, color=color, label=label)

        plt.xlabel("Hs (ft)")
        plt.ylabel("True Angle (deg)")
        plt.title("Operating Envelope Overlay")
        plt.legend(title="Error tier", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved operating envelope overlay to {output_path}")

    def plot_filtered_error_dashboard(self, df: pd.DataFrame, output_path: Path, hs_col: str = "Hs_ft"):
        """
        Static dashboard-style grid with filtered slices by Hs quantiles.
        """
        if df.empty or {hs_col, "true_angle", "abs_error"}.intersection(df.columns) != {hs_col, "true_angle", "abs_error"}:
            self.logger.warning("Filtered error dashboard skipped: missing required columns.")
            return

        quantiles = np.linspace(0, 1, 5)
        thresholds = df[hs_col].quantile(quantiles).unique()
        thresholds.sort()
        panels = []
        for i in range(len(thresholds) - 1):
            low, high = thresholds[i], thresholds[i + 1]
            mask = (df[hs_col] >= low) & (df[hs_col] <= high)
            slice_df = df[mask].copy()
            if slice_df.empty:
                continue
            slice_df["hs_range"] = f"{low:.2f}-{high:.2f}"
            panels.append(slice_df)
        if not panels:
            self.logger.warning("Filtered error dashboard skipped: no slices to plot.")
            return

        plot_df = pd.concat(panels, ignore_index=True)
        g = sns.FacetGrid(plot_df, col="hs_range", col_wrap=3, height=3, sharex=False, sharey=False)
        g.map_dataframe(sns.scatterplot, x="true_angle", y="abs_error", hue="abs_error", palette="magma", s=10, alpha=0.8)
        g.add_legend()
        plt.suptitle("Filtered Error Dashboard by Hs ranges", y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved filtered error dashboard to {output_path}")

    # =========================================================================
    # #15: Improvement Waterfall Chart
    # =========================================================================
    def plot_improvement_waterfall(self, deltas: pd.DataFrame, output_path: Path):
        """
        Waterfall of cumulative metric deltas. Expects columns: feature/step, delta (positive/negative).
        """
        required = {'step', 'delta'}
        if deltas is None or deltas.empty or not required.issubset(deltas.columns):
            self.logger.warning("Improvement waterfall skipped: missing step/delta.")
            return

        deltas = deltas.copy()
        deltas['cumulative'] = deltas['delta'].cumsum()

        plt.figure(figsize=(12, 6))
        colors = deltas['delta'].apply(lambda x: '#2ca02c' if x <= 0 else '#d62728')
        plt.bar(deltas['step'], deltas['delta'], color=colors)
        plt.plot(deltas['step'], deltas['cumulative'], color='black', marker='o', linestyle='--', label='Cumulative')
        plt.xticks(rotation=30, ha='right')
        plt.ylabel("Delta (lower is better)")
        plt.title("Improvement Waterfall (Cumulative Deltas)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved improvement waterfall to {output_path}")

    # =========================================================================
    # #16: Cluster Evolution Across Rounds (Counts)
    # =========================================================================
    def plot_cluster_evolution(self, df: pd.DataFrame, output_path: Path):
        """
        Line plot of cluster counts over rounds.
        Expects columns: round, cluster_id.
        """
        required = {'round', 'cluster_id'}
        if df is None or df.empty or not required.issubset(df.columns):
            self.logger.warning("Cluster evolution skipped: missing round/cluster_id.")
            return
        counts = df.groupby(['round', 'cluster_id']).size().reset_index(name='count')
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=counts, x='round', y='count', hue='cluster_id', marker='o')
        plt.title("Cluster Evolution Across Rounds")
        plt.xlabel("Round")
        plt.ylabel("Count")
        plt.legend(title="Cluster")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved cluster evolution to {output_path}")

    def plot_cluster_evolution_overlay(self, df: pd.DataFrame, output_path: Path):
        """
        Scatter overlay of clusters with optional positions (or hs/angle) to visualize movement.
        """
        if df.empty:
            self.logger.warning("Cluster evolution overlay skipped: empty dataframe.")
            return

        has_positions = {'pos_x', 'pos_y'}.issubset(df.columns)
        if not has_positions and not {'true_angle', 'pred_angle'}.issubset(df.columns):
            self.logger.warning("Cluster evolution overlay skipped: no positional proxies.")
            return

        plt.figure(figsize=(10, 6))
        if has_positions:
            x_col, y_col = 'pos_x', 'pos_y'
        else:
            x_col, y_col = 'true_angle', 'pred_angle'

        if 'cluster_id' not in df.columns:
            df['cluster_id'] = 'cluster'

        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue='cluster_id',
            style='round' if 'round' in df.columns else None,
            alpha=0.7,
            s=30,
        )
        plt.title("Cluster Evolution Overlay")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved cluster evolution overlay to {output_path}")

    # =========================================================================
    # #17: Boundary Gradient Visualization
    # =========================================================================
    def plot_boundary_gradient(self, df: pd.DataFrame, output_path: Path):
        """
        Gradient of abs_error near angle boundary (0/360). Expects true_angle, abs_error.
        """
        required = {'true_angle', 'abs_error'}
        if not required.issubset(df.columns):
            self.logger.warning("Boundary gradient skipped: missing required columns.")
            return
        subset = df[(df['true_angle'] <= 30) | (df['true_angle'] >= 330)].copy()
        if subset.empty:
            self.logger.warning("Boundary gradient skipped: no samples near boundary.")
            return
        subset = subset.sort_values('true_angle')
        subset['gradient'] = np.gradient(subset['abs_error'])
        plt.figure(figsize=(10, 5))
        plt.plot(subset['true_angle'], subset['gradient'], color='#ff7f0e')
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.xlabel("True Angle (deg)")
        plt.ylabel("Gradient of Abs Error")
        plt.title("Boundary Gradient of Error near 0/360")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved boundary gradient to {output_path}")

    # =========================================================================
    # #18: High-Error Region Zoom
    # =========================================================================
    def plot_high_error_zoom(self, df: pd.DataFrame, output_path: Path, hs_col: str = "Hs_ft", top_n: int = 50):
        """
        Zoomed view of top-N highest errors, scatter over Hs vs Angle colored by abs_error.
        """
        required = {hs_col, "true_angle", "abs_error", "pred_angle"}
        if not required.issubset(df.columns):
            self.logger.warning("High-error zoom skipped: missing required columns.")
            return
        if df.empty:
            self.logger.warning("High-error zoom skipped: empty dataframe.")
            return

        top = df.nlargest(top_n, "abs_error").copy()
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(top[hs_col], top["true_angle"], c=top["abs_error"], cmap="inferno", s=40, alpha=0.8)
        plt.colorbar(sc, label="Abs Error (deg)")
        plt.xlabel("Hs (ft)")
        plt.ylabel("True Angle (deg)")
        plt.title(f"Top {top_n} Highest Errors (Zoom)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved high-error zoom to {output_path}")

    def plot_high_error_zoom_facets(self, df: pd.DataFrame, output_path: Path, hs_col: str = "Hs_ft", top_n: int = 100):
        """
        Faceted zoom of highest-error samples by hs_bin and angle_bin for drill-down.
        """
        required = {hs_col, "true_angle", "abs_error"}
        if not required.issubset(df.columns) or df.empty:
            self.logger.warning("High-error zoom facets skipped: missing required columns.")
            return

        top = df.nlargest(top_n, "abs_error").copy()
        if "hs_bin" not in top.columns:
            try:
                top["hs_bin"] = pd.qcut(top[hs_col], q=min(4, len(top)), duplicates="drop")
            except Exception:
                top["hs_bin"] = "all"
        if "angle_bin" not in top.columns:
            top["angle_bin"] = pd.cut(top["true_angle"], bins=8, labels=False, include_lowest=True)

        g = sns.FacetGrid(top, col="hs_bin", row="angle_bin", margin_titles=True, sharex=False, sharey=False, height=2)
        g.map_dataframe(sns.scatterplot, x=hs_col, y="abs_error", hue="true_angle", palette="viridis", s=12, alpha=0.8)
        g.add_legend()
        plt.suptitle("High-Error Zoom (faceted by hs/angle bins)", y=1.02, fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved high-error zoom facets to {output_path}")

    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    def _save_and_close(self, path: Path):
        """Helper to save and close plot."""
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
