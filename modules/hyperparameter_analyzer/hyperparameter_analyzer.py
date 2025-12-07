import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
# Import required for 3D plotting
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class HyperparameterAnalyzer:
    """
    Analyzes HPO results to visualize parameter landscapes and identify optimal ranges.
    Supports:
    - 2D Heatmaps (Categorical/Mixed)
    - 2D Contour Plots with Optimal Region Highlighting (Numeric)
    - 3D Surface Plots with Optimal Floor Projection (Numeric)
    - Optimal Range Reporting
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir: Path = None
        
        # List of potential CV metrics to look for and analyze (low is better)
        self.potential_metric_cols = [
            'cv_cmae_deg_mean', # Expected primary metric name
            'cv_mean_cmae',     # Old/alternative name (caused the error)
            'cv_crmse_deg_mean',
            'cv_max_error_deg_mean' # Analyzing for Max Error as requested
        ]
        self.primary_metric = None 
        
    def analyze(self, run_id: str) -> None:
        """
        Execute full analysis on HPO results.
        """
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        hpo_dir = Path(base_dir) / "04_HYPERPARAMETER_SEARCH"
        results_file = hpo_dir / "all_config_results.xlsx"
        
        if not results_file.exists():
            self.logger.warning(f"HPO results file not found at {results_file}. Skipping analysis.")
            return

        self.logger.info("Starting Hyperparameter Landscape Analysis...")
        self.output_dir = hpo_dir / "ANALYSIS"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load and Preprocess Data
        df = pd.read_excel(results_file)
        
        # Determine the primary metric and other available CV metrics
        available_metrics = [m for m in self.potential_metric_cols if m in df.columns]
        
        if not available_metrics:
            self.logger.error(f"No expected metric columns found in HPO results. Looked for: {self.potential_metric_cols}. Available: {list(df.columns)}")
            return
            
        # Use the first metric found as the primary metric for the optimal range report
        self.primary_metric = available_metrics[0] 
        self.logger.info(f"Using '{self.primary_metric}' as the primary metric for the optimal range report.")
        self.logger.info(f"Found metrics for analysis: {available_metrics}")

        df = self._preprocess_data(df)
        
        # 2. Identify Optimal Ranges (based on primary metric)
        top_configs = self._generate_optimal_ranges_report(df, self.primary_metric)
        
        # 3. Generate Visualizations per Model (iterate over all available metrics)
        models = df['model_name'].unique() if 'model_name' in df.columns else df['model'].unique()
        for model in models:
            model_col = 'model_name' if 'model_name' in df.columns else 'model'
            model_df = df[df[model_col] == model]
            self._visualize_model_landscape(model_df, model, top_configs, available_metrics)
            
        self.logger.info(f"HPO Analysis complete. Artifacts in {self.output_dir}")

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data, handle None/NaN, identify param columns."""
        # Convert 'param_' columns
        param_cols = [c for c in df.columns if c.startswith('param_')]
        
        for col in param_cols:
            # Convert None/NaN to string "None" for visualization
            df[col] = df[col].apply(lambda x: "None" if pd.isna(x) or x is None else x)
            
            # Attempt to convert to numeric if possible (e.g. mixed types)
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Keep as object/string if conversion fails
                pass
                
        return df

    def _generate_optimal_ranges_report(self, df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
        """
        Filter Top 10% of configurations based on metric_col and report min/max/values for parameters.
        Returns the top_df for highlighting logic.
        """
        # Sort by primary metric (CMAE) - Low is good
        df_sorted = df.sort_values(metric_col, ascending=True)
        
        # Select top 10%
        top_n = max(1, int(len(df) * 0.10))
        top_df = df_sorted.head(top_n)
        
        report_data = []
        param_cols = [c for c in df.columns if c.startswith('param_')]
        
        for col in param_cols:
            clean_name = col.replace('param_', '')
            unique_vals = top_df[col].unique()
            
            row = {'Parameter': clean_name}
            
            if np.issubdtype(top_df[col].dtype, np.number):
                row['Type'] = 'Numeric'
                row['Optimal Min'] = top_df[col].min()
                row['Optimal Max'] = top_df[col].max()
                row['Best Value'] = top_df.iloc[0][col]
            else:
                row['Type'] = 'Categorical'
                row['Optimal Values'] = ", ".join(map(str, unique_vals))
                row['Best Value'] = top_df.iloc[0][col]
                
            report_data.append(row)
            
        pd.DataFrame(report_data).to_excel(self.output_dir / "optimal_parameter_ranges.xlsx", index=False)
        self.logger.info(f"Generated Optimal Ranges Report based on {metric_col}.")
        return top_df

    def _visualize_model_landscape(self, df: pd.DataFrame, model_name: str, top_df: pd.DataFrame, metrics_to_plot: List[str]):
        """Generate 2D/3D plots for a specific model, iterating over all metrics."""
        param_cols = [c for c in df.columns if c.startswith('param_')]
        
        if len(param_cols) < 2:
            self.logger.info(f"Model {model_name} has < 2 parameters. Skipping surface plots.")
            return

        # Iterate pairs of parameters
        import itertools
        pairs = list(itertools.combinations(param_cols, 2))
        
        # Limit pairs to avoid explosion
        for p1, p2 in pairs[:5]: 
            for metric in metrics_to_plot: # Iterate over requested metrics
                self._plot_pair_analysis(df, top_df, model_name, p1, p2, metric)

    def _plot_pair_analysis(self, df: pd.DataFrame, top_df: pd.DataFrame, model: str, p1: str, p2: str, z_col: str):
        """Determine best plot type for a pair and generate it."""
        is_num_1 = np.issubdtype(df[p1].dtype, np.number)
        is_num_2 = np.issubdtype(df[p2].dtype, np.number)
        
        name1 = p1.replace('param_', '')
        name2 = p2.replace('param_', '')
        
        # Include model and metric name in save file for separation
        save_base = self.output_dir / f"{model}_{z_col}_{name1}_vs_{name2}"
        
        # 1. Plots for Numeric Pairs (Contour and 3D Surface)
        if is_num_1 and is_num_2:
            self._plot_contour_highlight(df, top_df, p1, p2, z_col, f"{save_base}_contour.png")
            self._plot_3d_surface(df, top_df, p1, p2, z_col, f"{save_base}_3d_surface.png")
            
        # 2. Heatmap (Robust for Categorical/Mixed/Numeric)
        self._plot_heatmap(df, p1, p2, z_col, f"{save_base}_heatmap.png")

    def _plot_contour_highlight(self, df: pd.DataFrame, top_df: pd.DataFrame, 
                               x_col: str, y_col: str, z_col: str, output_path: str):
        """
        2D Contour plot with highlighted optimal region.
        """
        x = df[x_col]
        y = df[y_col]
        z = df[z_col]
        
        # Grid for contour
        try:
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
        except Exception as e:
            self.logger.warning(f"Contour interpolation failed for {z_col}: {e}")
            return

        plt.figure(figsize=(10, 8))
        
        # Contour
        try:
            # _r so Blue=Low Error (Good) for minimization metrics
            cp = plt.contourf(xi, yi, zi, levels=20, cmap='viridis_r') 
            plt.colorbar(cp, label=z_col)
        except Exception as e:
             self.logger.warning(f"Contour plotting failed (possibly constant data) for {z_col}: {e}")
             plt.close()
             return

        # Scatter all points
        plt.scatter(x, y, c='black', s=10, alpha=0.3, label='Configurations')
        
        # Highlight Optimal Box (based on the globally optimal range from the primary metric)
        opt_x_min, opt_x_max = top_df[x_col].min(), top_df[x_col].max()
        opt_y_min, opt_y_max = top_df[y_col].min(), top_df[y_col].max()
        
        width = opt_x_max - opt_x_min
        height = opt_y_max - opt_y_min
        
        # Handle zero-width (single best value)
        width = width if width > 0 else (x.max()-x.min())*0.05
        height = height if height > 0 else (y.max()-y.min())*0.05
        
        rect = Rectangle((opt_x_min, opt_y_min), width, height, 
                         linewidth=2, edgecolor='red', facecolor='none', label='Optimal Region (Top 10%)')
        plt.gca().add_patch(rect)
        
        # Highlight ticks
        plt.xticks(rotation=45)
        
        clean_x = x_col.replace('param_', '')
        clean_y = y_col.replace('param_', '')
        plt.xlabel(clean_x)
        plt.ylabel(clean_y)
        plt.title(f"Contour: {clean_x} vs {clean_y} (Metric: {z_col})")
        plt.legend(loc='upper right')
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Generated Contour plot: {Path(output_path).name}")
        
    def _plot_3d_surface(self, df: pd.DataFrame, top_df: pd.DataFrame, x_col: str, y_col: str, z_col: str, output_path: str):
        """
        3D Surface plot with optimal region projected on the floor and highlighted points.
        """
        x = df[x_col].values
        y = df[y_col].values
        z = df[z_col].values
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d') 

        # Create grid for smooth surface plot
        try:
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
        except Exception as e:
            self.logger.warning(f"3D Surface interpolation failed for {z_col}: {e}")
            plt.close(fig)
            return
            
        XI, YI = np.meshgrid(xi, yi)
        
        # 1. Plot the Surface (Semi-transparent)
        surf = ax.plot_surface(XI, YI, zi, cmap='viridis_r', edgecolor='none', alpha=0.6)
        
        # 2. Scatter Points: Highlight Top 10% in Green, others in Red
        # Identify which points in df are in top_df (by index or value match)
        # Assuming unique values for simplicity, but index matching is safer
        is_optimal = df.index.isin(top_df.index)
        
        # Plot Non-Optimal (Red, small)
        ax.scatter(x[~is_optimal], y[~is_optimal], z[~is_optimal], 
                   c='red', marker='o', s=10, alpha=0.3, label='Standard Configs')
        
        # Plot Optimal (Green, larger)
        ax.scatter(x[is_optimal], y[is_optimal], z[is_optimal], 
                   c='lime', marker='*', s=50, alpha=1.0, label='Optimal Configs (Top 10%)')

        # 3. Project Optimal Area on Floor (Shadow Box)
        opt_x_min, opt_x_max = top_df[x_col].min(), top_df[x_col].max()
        opt_y_min, opt_y_max = top_df[y_col].min(), top_df[y_col].max()
        z_floor = z.min()
        
        # Define vertices of the rectangle on the floor
        verts = [
            [(opt_x_min, opt_y_min, z_floor), (opt_x_max, opt_y_min, z_floor), 
             (opt_x_max, opt_y_max, z_floor), (opt_x_min, opt_y_max, z_floor)]
        ]
        
        # Add the semi-transparent polygon on the floor
        poly = Poly3DCollection(verts, alpha=0.4, facecolor='green', edgecolor='black')
        ax.add_collection3d(poly)

        # Labels & Legend
        clean_x = x_col.replace('param_', '')
        clean_y = y_col.replace('param_', '')
        
        ax.set_xlabel(clean_x)
        ax.set_ylabel(clean_y)
        ax.set_zlabel(z_col)
        ax.set_title(f"3D Surface: {clean_x} vs {clean_y}\n(Floor shadow indicates optimal range)")
        
        fig.colorbar(surf, shrink=0.5, aspect=5, label=z_col)
        ax.legend(loc='upper left')
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Generated 3D Surface plot: {Path(output_path).name}")

    def _plot_heatmap(self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str, output_path: str):
        """
        Pivot Heatmap handling categorical/text/None values.
        """
        try:
            # Pivot table to get the mean metric value for each combination
            pivot = df.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc='mean')
        except Exception as e:
            self.logger.warning(f"Pivot failed for heatmap {x_col} vs {y_col} (Metric: {z_col}): {e}")
            return
            
        plt.figure(figsize=(10, 8))
        # cmap='viridis_r' uses dark colors for low error (good)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap='viridis_r', cbar_kws={'label': z_col})
        
        clean_x = x_col.replace('param_', '')
        clean_y = y_col.replace('param_', '')
        plt.xlabel(clean_x)
        plt.ylabel(clean_y)
        plt.title(f"Performance Heatmap: {clean_x} vs {clean_y} (Metric: {z_col})")
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Generated Heatmap plot: {Path(output_path).name}")