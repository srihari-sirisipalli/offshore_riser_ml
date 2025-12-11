import matplotlib
matplotlib.use("Agg")  # Ensure non-interactive backend for thread safety
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Callable, Tuple
from joblib import Parallel, delayed
from utils import constants

class RFEVisualizer:
    """
    Generates RFE-specific visualizations as defined in Specification Part 9.
    
    Covers:
    1. LOFO Feature Impact (Bar charts, Waterfalls).
    2. Baseline vs Dropped Comparison (Scatter overlays, CDFs).
    3. Error Heatmaps.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Style settings
        sns.set_theme(style="whitegrid")
        vis_config = config.get('visualization', {})
        self.parallel = vis_config.get('parallel_plots', False)
        self.n_jobs = config.get('execution', {}).get('n_jobs', -1)
        self.colors = {
            'baseline': '#1f77b4',  # Blue
            'dropped': '#d62728',   # Red
            'improvement': '#2ca02c', # Green
            'degradation': '#ff7f0e'  # Orange
        }

    def _run_tasks(self, tasks: List[Tuple[str, Callable[[], None]]]) -> None:
        """
        Execute plotting tasks in parallel (threaded) or sequentially depending on config.
        """
        if not tasks:
            return

        def _safe_task(name: str, fn: Callable[[], None]) -> str:
            try:
                plt.switch_backend('Agg')
                fn()
                return f"Success:{name}"
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"Failed plot '{name}': {exc}")
                return f"Failed:{name}"

        if self.parallel and len(tasks) > 1:
            results = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(_safe_task)(name, fn) for name, fn in tasks
            )
        else:
            results = [_safe_task(name, fn) for name, fn in tasks]

        failures = [r for r in results if r.startswith("Failed")]
        if failures:
            self.logger.warning(f"{len(failures)} visualization task(s) failed: {failures}")

    def visualize_lofo_impact(self, round_dir: Path, lofo_results: List[Dict]):
        """
        Generates plots summarizing the impact of removing each feature.
        Output: 05_Feature_ImportanceRanking/feature_evaluation_plots/
        """
        output_dir = round_dir / constants.ROUND_FEATURE_EVALUATION_DIR / "feature_evaluation_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not lofo_results:
            return

        df = pd.DataFrame(lofo_results)
        tasks: List[Tuple[str, Callable[[], None]]] = []

        def _task_lofo_bar():
            # Negative Delta = Error Decreased (Good) -> Feature was harmful.
            # Positive Delta = Error Increased (Bad) -> Feature was helpful.
            df_sorted = df.sort_values('delta_cmae')

            plt.figure(figsize=(12, 8))

            colors = df_sorted['delta_cmae'].apply(
                lambda x: self.colors['improvement'] if x < 0 else self.colors['degradation']
            )

            sns.barplot(x='delta_cmae', y='feature', data=df_sorted, hue='feature', palette=colors.tolist(), legend=False)

            plt.axvline(0, color='black', linewidth=1)
            plt.title("LOFO Impact: Change in CMAE when feature is removed\n(Negative = Improvement)")
            plt.xlabel("Delta CMAE (degrees)")
            plt.ylabel("Feature")

            self._save_and_close(output_dir / "lofo_comparison_bar.png")

        tasks.append(("lofo_bar", _task_lofo_bar))

        val_metrics = ['val_cmae', 'val_crmse', 'val_max_error']
        test_metrics = ['test_cmae', 'test_crmse', 'test_max_error']

        available_val = [m for m in val_metrics if m in df.columns]
        available_test = [m for m in test_metrics if m in df.columns]

        if available_val:
            def _task_val_heatmap():
                plt.figure(figsize=(10, len(df) * 0.4 + 2))

                heatmap_data = df.set_index('feature')[available_val]
                if len(heatmap_data) > 1:
                    range_ = heatmap_data.max() - heatmap_data.min()
                    range_[range_ == 0] = 1
                    normalized_data = (heatmap_data - heatmap_data.min()) / range_
                else:
                    normalized_data = heatmap_data

                sns.heatmap(normalized_data, annot=heatmap_data, fmt=".3f", cmap="viridis_r")
                plt.title("LOFO Metrics Heatmap (Validation Set)")
                self._save_and_close(output_dir / "lofo_val_metrics_heatmap.png")

            tasks.append(("lofo_val_heatmap", _task_val_heatmap))

        if available_test:
            def _task_test_heatmap():
                plt.figure(figsize=(10, len(df) * 0.4 + 2))

                heatmap_data = df.set_index('feature')[available_test]
                if len(heatmap_data) > 1:
                    range_ = heatmap_data.max() - heatmap_data.min()
                    range_[range_ == 0] = 1
                    normalized_data = (heatmap_data - heatmap_data.min()) / range_
                else:
                    normalized_data = heatmap_data

                sns.heatmap(normalized_data, annot=heatmap_data, fmt=".3f", cmap="viridis_r")
                plt.title("LOFO Metrics Heatmap (Test Set)")
                self._save_and_close(output_dir / "lofo_test_metrics_heatmap.png")

            tasks.append(("lofo_test_heatmap", _task_test_heatmap))

        self._run_tasks(tasks)

    def visualize_comparison(self,
                             round_dir: Path,
                             baseline_preds: pd.DataFrame,
                             dropped_preds: pd.DataFrame,
                             feature_name: str,
                             baseline_metrics: dict = None,
                             dropped_metrics: dict = None):
        """
        Generates comprehensive comparison plots comparing Baseline vs Best Dropped Model.
        Output: 07_ModelComparison_BaseVsReduced/comparison_plots/
        """
        output_dir = round_dir / constants.ROUND_COMPARISON_DIR / "comparison_plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Align dataframes by index just in case
        common_idx = baseline_preds.index.intersection(dropped_preds.index)
        base = baseline_preds.loc[common_idx]
        drop = dropped_preds.loc[common_idx]

        tasks: List[Tuple[str, Callable[[], None]]] = []

        if baseline_metrics and dropped_metrics:
            tasks.append((
                "metrics_comparison",
                lambda: self._plot_comprehensive_metrics_comparison(
                    baseline_metrics, dropped_metrics, feature_name, output_dir
                ),
            ))

        tasks.append((
            "cdf_overlay",
            lambda: self._plot_cdf_overlay(base['abs_error'], drop['abs_error'], feature_name, output_dir),
        ))

        tasks.append((
            "scatter_overlay",
            lambda: self._plot_scatter_overlay(base, drop, feature_name, output_dir),
        ))

        tasks.append((
            "residual_distribution_overlay",
            lambda: self._plot_residual_dist_overlay(base['error'], drop['error'], feature_name, output_dir),
        ))

        self._run_tasks(tasks)

    def _plot_cdf_overlay(self, err_base, err_drop, fname, output_dir):
        """Plots cumulative distribution function of absolute errors."""
        plt.figure(figsize=(10, 6))
        
        sns.ecdfplot(data=err_base, label='Baseline (All Features)', color=self.colors['baseline'], linewidth=2)
        sns.ecdfplot(data=err_drop, label=f'Dropped ({fname})', color=self.colors['dropped'], linewidth=2, linestyle='--')
        
        plt.title(f"Error CDF Comparison: Baseline vs Drop '{fname}'")
        plt.xlabel("Absolute Error (degrees)")
        plt.ylabel("Proportion")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Zoom in to 0-20 degrees if possible, as that's the interesting part
        plt.xlim(0, 20) 
        
        self._save_and_close(output_dir / "before_after_error_cdf.png")

    def _plot_scatter_overlay(self, base_df, drop_df, fname, output_dir):
        """Scatter plot of True vs Predicted for both models with visibility tweaks."""
        plt.figure(figsize=(12, 12))

        plt.scatter(
            base_df['true_angle'], base_df['pred_angle'],
            alpha=0.4, color='grey', s=35, marker='o',
            edgecolors='black', linewidths=0.3, label='Baseline', zorder=1
        )

        plt.scatter(
            drop_df['true_angle'], drop_df['pred_angle'],
            alpha=0.5, color=self.colors['dropped'], s=30, marker='x',
            linewidths=1.5, label=f'Dropped ({fname})', zorder=2
        )

        plt.plot([0, 360], [0, 360], 'k--', linewidth=2, label='Perfect', zorder=0, alpha=0.7)

        angles = np.array([0, 360])
        plt.fill_between(angles, angles - 5, angles + 5, color='green', alpha=0.08, label='+/-5 deg', zorder=0)
        plt.fill_between(angles, angles - 10, angles + 10, color='yellow', alpha=0.05, label='+/-10 deg', zorder=0)

        plt.xlabel("True Angle (deg)")
        plt.ylabel("Predicted Angle (deg)")
        plt.title(f"Prediction Scatter Overlay")
        plt.xlim([0, 360])
        plt.ylim([0, 360])
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle='--', axis='both')
        plt.gca().set_axisbelow(True)
        plt.tight_layout()

        self._save_and_close(output_dir / "angle_scatter_overlay.png")

    def _plot_residual_dist_overlay(self, err_base, err_drop, fname, output_dir):
        """KDE plot of signed residuals."""
        plt.figure(figsize=(10, 6))
        
        sns.kdeplot(err_base, fill=True, label='Baseline', color=self.colors['baseline'], alpha=0.2)
        sns.kdeplot(err_drop, fill=True, label=f'Dropped ({fname})', color=self.colors['dropped'], alpha=0.2)
        
        plt.axvline(0, color='black', linestyle=':')
        plt.title(f"Residual Distribution Overlay")
        plt.xlabel("Signed Error (deg)")
        plt.xlim(-20, 20) # Focus on the peak
        plt.legend()
        
        self._save_and_close(output_dir / "residual_distribution_overlay.png")

    def _plot_comprehensive_metrics_comparison(self, baseline_metrics, dropped_metrics, feature_name, output_dir):
        """
        Creates comprehensive bar charts comparing baseline vs dropped model metrics.
        Shows Val and Test metrics side-by-side with delta annotations.
        """
        # Define metrics to compare (lower is better for errors, higher is better for accuracy)
        error_metrics = ['cmae', 'crmse', 'max_error']
        accuracy_metrics = ['accuracy_at_0deg', 'accuracy_at_5deg', 'accuracy_at_10deg']

        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))

        # --- SUBPLOT 1: Error Metrics ---
        ax1 = axes[0]
        self._plot_metrics_grouped_bars(
            ax1, baseline_metrics, dropped_metrics,
            error_metrics, feature_name,
            title="Error Metrics Comparison (Lower is Better)",
            ylabel="Error (degrees)",
            lower_is_better=True
        )

        # --- SUBPLOT 2: Accuracy Metrics ---
        ax2 = axes[1]
        self._plot_metrics_grouped_bars(
            ax2, baseline_metrics, dropped_metrics,
            accuracy_metrics, feature_name,
            title="Accuracy Metrics Comparison (Higher is Better)",
            ylabel="Accuracy (%)",
            lower_is_better=False
        )

        plt.tight_layout()
        self._save_and_close(output_dir / "comprehensive_metrics_comparison.png")

    def _plot_metrics_grouped_bars(self, ax, baseline_metrics, dropped_metrics, metric_names, feature_name, title, ylabel, lower_is_better):
        """Helper to plot grouped bar charts with delta annotations."""
        x_labels = []
        baseline_vals_list = []
        baseline_tests_list = []
        dropped_vals_list = []
        dropped_tests_list = []

        # Extract data
        for metric in metric_names:
            # Format metric name for display
            display_name = metric.replace('_', ' ').replace('accuracy at ', '@').upper()
            x_labels.append(display_name)

            # Baseline
            baseline_val = baseline_metrics.get(f'val_{metric}', 0)
            baseline_test = baseline_metrics.get(f'test_{metric}', 0)
            baseline_vals_list.append(baseline_val)
            baseline_tests_list.append(baseline_test)

            # Dropped
            dropped_val = dropped_metrics.get(f'val_{metric}', 0)
            dropped_test = dropped_metrics.get(f'test_{metric}', 0)
            dropped_vals_list.append(dropped_val)
            dropped_tests_list.append(dropped_test)

        x = np.arange(len(x_labels))
        width = 0.2

        # Create bars
        bars1 = ax.bar(x - 1.5*width, baseline_vals_list, width, label='Baseline Val', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, baseline_tests_list, width, label='Baseline Test', color='#1f77b4', alpha=0.5, hatch='//')
        bars3 = ax.bar(x + 0.5*width, dropped_vals_list, width, label='Dropped Val', color='#d62728', alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, dropped_tests_list, width, label='Dropped Test', color='#d62728', alpha=0.5, hatch='//')

        # Add delta annotations
        for i in range(len(x_labels)):
            # Val delta
            delta_val = dropped_vals_list[i] - baseline_vals_list[i]
            self._add_delta_annotation(ax, i - width, max(baseline_vals_list[i], dropped_vals_list[i]), delta_val, lower_is_better)

            # Test delta
            delta_test = dropped_tests_list[i] - baseline_tests_list[i]
            self._add_delta_annotation(ax, i + width, max(baseline_tests_list[i], dropped_tests_list[i]), delta_test, lower_is_better)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title}\n(Dropped Feature: '{feature_name}')", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0, ha='center')
        ax.legend(loc='upper left', framealpha=0.95)
        ax.grid(axis='both', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    def _add_delta_annotation(self, ax, x_pos, y_pos, delta, lower_is_better):
        """Add delta annotation with color coding."""
        if abs(delta) < 0.001:
            return  # Skip negligible deltas

        # Determine if delta is an improvement
        is_improvement = (delta < 0 and lower_is_better) or (delta > 0 and not lower_is_better)
        color = 'green' if is_improvement else 'red'
        sign = '+' if delta > 0 else ''

        # Add annotation
        ax.annotate(f'{sign}{delta:.2f}',
                   xy=(x_pos, y_pos),
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   color=color,
                   weight='bold')

    def _save_and_close(self, path: Path):
        """Helper to save figure and clean memory."""
        try:
            plt.savefig(path, dpi=200, bbox_inches='tight')
        except Exception as e:
            self.logger.warning(f"Failed to save plot {path}: {e}")
        finally:
            plt.close('all')
