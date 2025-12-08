import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from contextlib import contextmanager # Added for FIX #97

class HyperparameterAnalyzer:
    """
    Analyzes HPO results with comprehensive visualizations:
    - Multi-metric analysis (CV, Val, Test)
    - 2D Heatmaps, 2D Contours, 3D Surfaces
    - Optimal region highlighting (configurable top %)
    - Multiple highlighting styles
    - Zoomed "optimal only" plots
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir: Path = None
        
        # Configurable optimal range percentage
        self.optimal_top_percent = config.get('hpo_analysis', {}).get('optimal_top_percent', 10)
        
        # Highlighting styles to generate
        self.highlighting_styles = config.get('hpo_analysis', {}).get('highlighting_styles', 
                                                                      ['box', 'stars', 'overlay'])
        
        # Metrics to analyze
        self.cv_metrics = ['cv_cmae_deg_mean', 'cv_crmse_deg_mean', 'cv_max_error_deg_mean']
        self.val_metrics = ['val_cmae_deg', 'val_crmse_deg', 'val_max_error_deg']
        self.test_metrics = ['test_cmae_deg', 'test_crmse_deg', 'test_max_error_deg']
        
        self.primary_metric = 'cv_cmae_deg_mean'

    @contextmanager
    def _plot_context(self):
        """
        FIX #97: Context manager to save and restore Matplotlib/Seaborn styles.
        Ensures global settings don't leak.
        """
        original_rcParams = plt.rcParams.copy()
        original_seaborn_style = sns.axes_style()
        try:
            yield
        finally:
            plt.rcParams.update(original_rcParams)
            sns.set_style(original_seaborn_style)
            plt.close('all') # Close all figures created within this context
            
    def analyze(self, run_id: str) -> None:
        """Execute full analysis on HPO results."""
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        hpo_dir = Path(base_dir) / "03_HYPERPARAMETER_OPTIMIZATION" / "results"
        results_file = hpo_dir / "all_configurations.xlsx"
        
        if not results_file.exists():
            self.logger.warning(f"HPO results file not found at {results_file}. Skipping analysis.")
            return

        self.logger.info("Starting Hyperparameter Analysis...")
        self.output_dir = Path(base_dir) / "05_HYPERPARAMETER_ANALYSIS"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        df = pd.read_excel(results_file)
        
        # Check available metrics
        available_cv = [m for m in self.cv_metrics if m in df.columns]
        available_val = [m for m in self.val_metrics if m in df.columns]
        available_test = [m for m in self.test_metrics if m in df.columns]
        
        if not available_cv:
            self.logger.error(f"No CV metrics found. Cannot proceed.")
            return
            
        self.logger.info(f"Found metrics - CV: {len(available_cv)}, Val: {len(available_val)}, Test: {len(available_test)}")

        df = self._preprocess_data(df)
        
        # Generate optimal ranges report (multi-sheet)
        self._generate_optimal_ranges_report_multisheet(df)
        
        # Generate summary report
        self._generate_summary_report(df, available_cv, available_val, available_test)
        
        # Visualizations per model
        models = df['model_name'].unique() if 'model_name' in df.columns else df['model'].unique()
        
        for model in models:
            model_col = 'model_name' if 'model_name' in df.columns else 'model'
            model_df = df[df[model_col] == model]
            
            self.logger.info(f"Generating visualizations for {model}...")
            
            # Get top configs for this model
            top_configs = self._get_top_configs(model_df, self.optimal_top_percent)
            
            # FIX #97: Wrap plotting in a context manager to prevent style leaks.
            with self._plot_context():
                # Generate plots for each metric type
                for metric_type, metrics in [('CV', available_cv), ('Val', available_val), ('Test', available_test)]:
                    for metric in metrics:
                        self._visualize_model_landscape(model_df, top_configs, model, metric, metric_type)
            
        self.logger.info(f"HPO Analysis complete. Artifacts in {self.output_dir}")

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and identify parameter types."""
        param_cols = [c for c in df.columns if c.startswith('param_')]
        
        for col in param_cols:
            # Convert None/NaN to string "None"
            df[col] = df[col].apply(lambda x: "None" if pd.isna(x) or x is None else x)
            
            # Try numeric conversion
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
                
        return df

    def _get_top_configs(self, df: pd.DataFrame, top_percent: float) -> pd.DataFrame:
        """Get top X% of configurations based on primary metric."""
        df_sorted = df.sort_values(self.primary_metric, ascending=True)
        top_n = max(1, int(len(df) * (top_percent / 100.0)))
        return df_sorted.head(top_n)

    def _generate_optimal_ranges_report_multisheet(self, df: pd.DataFrame) -> None:
        """Generate multi-sheet Excel with optimal ranges."""
        output_file = self.output_dir / "optimal_ranges.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet (all models combined)
            summary_data = []
            param_cols = [c for c in df.columns if c.startswith('param_')]
            
            top_overall = self._get_top_configs(df, self.optimal_top_percent)
            
            for col in param_cols:
                clean_name = col.replace('param_', '')
                unique_vals = top_overall[col].unique()
                
                row = {'Parameter': clean_name}
                
                if np.issubdtype(top_overall[col].dtype, np.number):
                    row['Type'] = 'Numeric'
                    row['Optimal Min'] = top_overall[col].min()
                    row['Optimal Max'] = top_overall[col].max()
                    row['Best Value'] = top_overall.iloc[0][col]
                else:
                    row['Type'] = 'Categorical'
                    row['Optimal Values'] = ", ".join(map(str, unique_vals))
                    row['Best Value'] = top_overall.iloc[0][col]
                    
                summary_data.append(row)
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Per-model sheets
            models = df['model_name'].unique() if 'model_name' in df.columns else df['model'].unique()
            
            for model in models:
                model_col = 'model_name' if 'model_name' in df.columns else 'model'
                model_df = df[df[model_col] == model]
                top_model = self._get_top_configs(model_df, self.optimal_top_percent)
                
                model_data = []
                for col in param_cols:
                    if col not in model_df.columns:
                        continue
                        
                    clean_name = col.replace('param_', '')
                    unique_vals = top_model[col].unique()
                    
                    row = {'Parameter': clean_name}
                    
                    if np.issubdtype(top_model[col].dtype, np.number):
                        row['Type'] = 'Numeric'
                        row['Optimal Min'] = top_model[col].min()
                        row['Optimal Max'] = top_model[col].max()
                        row['Best Value'] = top_model.iloc[0][col]
                    else:
                        row['Type'] = 'Categorical'
                        row['Optimal Values'] = ", ".join(map(str, unique_vals))
                        row['Best Value'] = top_model.iloc[0][col]
                        
                    model_data.append(row)
                
                sheet_name = model[:31]  # Excel sheet name limit
                pd.DataFrame(model_data).to_excel(writer, sheet_name=sheet_name, index=False)
        
        self.logger.info(f"Generated multi-sheet optimal ranges: {output_file}")

    def _generate_summary_report(self, df: pd.DataFrame, cv_metrics: List[str], 
                                 val_metrics: List[str], test_metrics: List[str]) -> None:
        """Generate cross-model comparison summary."""
        models = df['model_name'].unique() if 'model_name' in df.columns else df['model'].unique()
        model_col = 'model_name' if 'model_name' in df.columns else 'model'
        
        summary = []
        
        for model in models:
            model_df = df[df[model_col] == model]
            best_row = model_df.sort_values(self.primary_metric, ascending=True).iloc[0]
            
            row = {'Model': model, 'Total Configs': len(model_df)}
            
            for m in cv_metrics:
                if m in best_row:
                    row[f'Best {m}'] = best_row[m]
            
            for m in val_metrics:
                if m in best_row:
                    row[f'Best {m}'] = best_row[m]
                    
            for m in test_metrics:
                if m in best_row:
                    row[f'Best {m}'] = best_row[m]
            
            summary.append(row)
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_excel(self.output_dir / "summary_report.xlsx", index=False)
        self.logger.info("Generated summary report")

    def _visualize_model_landscape(self, df: pd.DataFrame, top_df: pd.DataFrame, 
                                   model: str, metric: str, metric_type: str):
        """Generate all visualizations for a model-metric combination."""
        param_cols = [c for c in df.columns if c.startswith('param_')]
        
        if len(param_cols) < 2:
            return

        # Create organized directory structure
        viz_dir = self.output_dir / "visualizations" / model / metric_type / metric
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        import itertools
        pairs = list(itertools.combinations(param_cols, 2))
        
        for p1, p2 in pairs[:10]:  # Limit to avoid explosion
            self._plot_parameter_pair(df, top_df, p1, p2, metric, viz_dir)

    def _plot_parameter_pair(self, df: pd.DataFrame, top_df: pd.DataFrame,
                             p1: str, p2: str, z_col: str, output_dir: Path):
        """Generate all plot types for a parameter pair."""
        is_num_1 = self._is_numeric_param(df[p1])
        is_num_2 = self._is_numeric_param(df[p2])
        
        name1 = p1.replace('param_', '')
        name2 = p2.replace('param_', '')

        # FIX #63: Sanitize parameter names for safe use in filenames.
        # Replace common invalid filename characters with underscores.
        name1_sanitized = "".join([c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in name1])
        name2_sanitized = "".join([c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in name2])
        
        base_name = f"{name1_sanitized}_vs_{name2_sanitized}"
        
        # Always generate heatmap
        for style in self.highlighting_styles:
            heatmap_dir = output_dir / "heatmap" / style
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            self._plot_heatmap(df, top_df, p1, p2, z_col, 
                             heatmap_dir / f"{base_name}.png", style)
        
        # For numeric pairs, generate contour and 3D
        if is_num_1 and is_num_2:
            # 2D Contour plots
            for style in self.highlighting_styles:
                contour_dir = output_dir / "contour" / style
                contour_dir.mkdir(parents=True, exist_ok=True)
                
                # Full range contour
                self._plot_contour_2d(df, top_df, p1, p2, z_col,
                                     contour_dir / f"{base_name}_full.png", 
                                     style, zoom=False)
                
                # Optimal only contour
                self._plot_contour_2d(df, top_df, p1, p2, z_col,
                                     contour_dir / f"{base_name}_optimal.png", 
                                     style, zoom=True)
            
            # 3D Surface plots
            for style in self.highlighting_styles:
                surface_dir = output_dir / "3d_surface" / style
                surface_dir.mkdir(parents=True, exist_ok=True)
                
                # Full range 3D
                self._plot_3d_surface(df, top_df, p1, p2, z_col,
                                     surface_dir / f"{base_name}_full.png",
                                     style, zoom=False)
                
                # Optimal only 3D
                self._plot_3d_surface(df, top_df, p1, p2, z_col,
                                     surface_dir / f"{base_name}_optimal.png",
                                     style, zoom=True)

    def _is_numeric_param(self, series: pd.Series) -> bool:
        """Check if parameter is truly numeric."""
        if np.issubdtype(series.dtype, np.number):
            return True
        
        # Try converting all unique values
        try:
            unique_vals = series.unique()
            for val in unique_vals:
                if val == "None":
                    return False
                if isinstance(val, str) and not val.replace('.', '').replace('-', '').isdigit():
                    return False
            return True
        except:
            return False

    def _plot_heatmap(self, df: pd.DataFrame, top_df: pd.DataFrame,
                     x_col: str, y_col: str, z_col: str, output_path: Path, style: str):
        """Generate heatmap with highlighting."""
        # FIX #28: Warn if duplicate (x,y) combinations are found.
        if df.duplicated(subset=[x_col, y_col]).any():
            self.logger.warning(
                f"Duplicate (x,y) combinations found for {x_col} and {y_col}. "
                "The pivot table will use `aggfunc='mean'` to aggregate, "
                "which might hide repeated configurations."
            )
        try:
            pivot = df.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc='mean')
        except Exception as e:
            self.logger.warning(f"Pivot failed for {x_col} vs {y_col}: {e}")
            return
            
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(pivot, annot=True, fmt=".2f", cmap='viridis_r', 
                        cbar_kws={'label': z_col})
        
        # Apply highlighting
        if style == 'box':
            self._add_heatmap_box_highlight(ax, pivot, top_df, x_col, y_col)
        elif style == 'stars':
            self._add_heatmap_star_highlight(ax, pivot, top_df, x_col, y_col)
        elif style == 'overlay':
            self._add_heatmap_overlay_highlight(ax, pivot, top_df, x_col, y_col)
        
        clean_x = x_col.replace('param_', '')
        clean_y = y_col.replace('param_', '')
        plt.xlabel(clean_x)
        plt.ylabel(clean_y)
        plt.title(f"Heatmap [{style}]: {clean_x} vs {clean_y}\n{z_col}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

    def _add_heatmap_box_highlight(self, ax, pivot, top_df, x_col, y_col):
        """Add bold box around optimal cells."""
        opt_x_vals = top_df[x_col].unique()
        opt_y_vals = top_df[y_col].unique()
        
        x_labels = [str(x) for x in pivot.columns]
        y_labels = [str(y) for y in pivot.index]
        
        for i, y_val in enumerate(pivot.index):
            for j, x_val in enumerate(pivot.columns):
                if str(x_val) in [str(v) for v in opt_x_vals] and \
                   str(y_val) in [str(v) for v in opt_y_vals]:
                    rect = Rectangle((j, i), 1, 1, fill=False, 
                                   edgecolor='red', linewidth=3)
                    ax.add_patch(rect)

    def _add_heatmap_star_highlight(self, ax, pivot, top_df, x_col, y_col):
        """Add star markers to optimal cells."""
        opt_x_vals = top_df[x_col].unique()
        opt_y_vals = top_df[y_col].unique()
        
        for i, y_val in enumerate(pivot.index):
            for j, x_val in enumerate(pivot.columns):
                if str(x_val) in [str(v) for v in opt_x_vals] and \
                   str(y_val) in [str(v) for v in opt_y_vals]:
                    ax.text(j + 0.5, i + 0.5, '★', color='gold', 
                           fontsize=20, ha='center', va='center')

    def _add_heatmap_overlay_highlight(self, ax, pivot, top_df, x_col, y_col):
        """Add semi-transparent overlay to optimal cells."""
        opt_x_vals = top_df[x_col].unique()
        opt_y_vals = top_df[y_col].unique()
        
        for i, y_val in enumerate(pivot.index):
            for j, x_val in enumerate(pivot.columns):
                if str(x_val) in [str(v) for v in opt_x_vals] and \
                   str(y_val) in [str(v) for v in opt_y_vals]:
                    rect = Rectangle((j, i), 1, 1, fill=True, 
                                   facecolor='lime', alpha=0.3, edgecolor='none')
                    ax.add_patch(rect)

    def _plot_contour_2d(self, df: pd.DataFrame, top_df: pd.DataFrame,
                         x_col: str, y_col: str, z_col: str, 
                         output_path: Path, style: str, zoom: bool = False):
        """Generate 2D contour plot with optional zoom to optimal range."""
        if zoom:
            plot_df = top_df.copy()
            title_suffix = "(Optimal Range Only)"
        else:
            plot_df = df.copy()
            title_suffix = "(Full Range)"
        
        x = plot_df[x_col].values
        y = plot_df[y_col].values
        z = plot_df[z_col].values

        # FIXED: Check for variance and close figure on early return
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            return 
        
        try:
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
        except Exception as e:
            self.logger.warning(f"Contour interpolation failed: {e}")
            return

        plt.figure(figsize=(12, 10))
        
        # Contour plot
        try:
            cp = plt.contourf(xi, yi, zi, levels=20, cmap='viridis_r')
            plt.colorbar(cp, label=z_col)
            plt.contour(xi, yi, zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        except Exception as e:
            self.logger.warning(f"Contour plotting failed: {e}")
            plt.close()
            return

        # Scatter all points
        plt.scatter(x, y, c='white', s=30, alpha=0.6, edgecolors='black', linewidths=1)
        
        # Highlight optimal region
        if not zoom and style == 'box':
            opt_x_min, opt_x_max = top_df[x_col].min(), top_df[x_col].max()
            opt_y_min, opt_y_max = top_df[y_col].min(), top_df[y_col].max()
            
            width = opt_x_max - opt_x_min if opt_x_max > opt_x_min else (x.max()-x.min())*0.05
            height = opt_y_max - opt_y_min if opt_y_max > opt_y_min else (y.max()-y.min())*0.05

            # FIX #85: Ensure width and height are not zero for the rectangle.
            if width == 0:
                width = 0.1 # A small default width
            if height == 0:
                height = 0.1 # A small default height
            
            rect = Rectangle((opt_x_min, opt_y_min), width, height, 
                           linewidth=3, edgecolor='red', facecolor='none', 
                           label='Optimal Region')
            plt.gca().add_patch(rect)
            plt.legend()
        elif not zoom and style == 'overlay':
            # Shade optimal region
            opt_x_min, opt_x_max = top_df[x_col].min(), top_df[x_col].max()
            opt_y_min, opt_y_max = top_df[y_col].min(), top_df[y_col].max()
            
            plt.axvspan(opt_x_min, opt_x_max, alpha=0.2, color='lime')
            plt.axhspan(opt_y_min, opt_y_max, alpha=0.2, color='lime')
        elif not zoom and style == 'stars':
            # Mark optimal points
            opt_x = top_df[x_col].values
            opt_y = top_df[y_col].values
            plt.scatter(opt_x, opt_y, marker='*', s=200, c='gold', 
                       edgecolors='red', linewidths=2, label='Optimal Configs')
            plt.legend()
        
        clean_x = x_col.replace('param_', '')
        clean_y = y_col.replace('param_', '')
        plt.xlabel(clean_x)
        plt.ylabel(clean_y)
        plt.title(f"2D Contour [{style}]: {clean_x} vs {clean_y} {title_suffix}\n{z_col}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

    def _plot_3d_surface(self, df: pd.DataFrame, top_df: pd.DataFrame,
                         x_col: str, y_col: str, z_col: str,
                         output_path: Path, style: str, zoom: bool = False):
        """Generate 3D surface plot with optional zoom."""
        if zoom:
            plot_df = top_df.copy()
            title_suffix = "(Optimal Range Only)"
        else:
            plot_df = df.copy()
            title_suffix = "(Full Range)"
        
        x = plot_df[x_col].values
        y = plot_df[y_col].values
        z = plot_df[z_col].values
        
        fig = plt.figure(figsize=(14, 12))
        # FIXED: Check for variance and close figure on early return (FIX #4, #35)
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            plt.close(fig) 
            return 

        ax = fig.add_subplot(111, projection='3d')

        try:
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
        except Exception as e:
            self.logger.warning(f"3D interpolation failed: {e}")
            plt.close(fig)
            return
            
        XI, YI = np.meshgrid(xi, yi)
        
        # Surface
        surf = ax.plot_surface(XI, YI, zi, cmap='viridis_r', 
                              edgecolor='none', alpha=0.7)
        
        # Highlighting
        if not zoom:
            is_optimal = df.index.isin(top_df.index)
            
            # Non-optimal points
            ax.scatter(x[~is_optimal], y[~is_optimal], z[~is_optimal], 
                      c='red', marker='o', s=20, alpha=0.4, label='Other Configs')
            
            # Optimal points
            if style == 'stars':
                ax.scatter(x[is_optimal], y[is_optimal], z[is_optimal], 
                          c='lime', marker='*', s=100, alpha=1.0, 
                          edgecolors='black', linewidths=1, label='Optimal Configs')
            else:
                ax.scatter(x[is_optimal], y[is_optimal], z[is_optimal], 
                          c='lime', marker='o', s=50, alpha=1.0, label='Optimal Configs')
            
            # Floor projection
            if style in ['box', 'overlay']:
                opt_x_min, opt_x_max = top_df[x_col].min(), top_df[x_col].max()
                opt_y_min, opt_y_max = top_df[y_col].min(), top_df[y_col].max()
                z_floor = z.min()
                
                verts = [
                    [(opt_x_min, opt_y_min, z_floor), (opt_x_max, opt_y_min, z_floor), 
                     (opt_x_max, opt_y_max, z_floor), (opt_x_min, opt_y_max, z_floor)]
                ]
                
                poly = Poly3DCollection(verts, alpha=0.3, facecolor='lime', edgecolor='darkgreen')
                ax.add_collection3d(poly)
        else:
            # Zoomed view - all points are optimal
            ax.scatter(x, y, z, c='lime', marker='o', s=40, alpha=0.8)
        
        clean_x = x_col.replace('param_', '')
        clean_y = y_col.replace('param_', '')
        
        ax.set_xlabel(clean_x)
        ax.set_ylabel(clean_y)
        ax.set_zlabel(z_col)
        ax.set_title(f"3D Surface [{style}]: {clean_x} vs {clean_y} {title_suffix}\n{z_col}")
        
        fig.colorbar(surf, shrink=0.5, aspect=5, label=z_col)
        ax.legend(loc='upper left')
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)