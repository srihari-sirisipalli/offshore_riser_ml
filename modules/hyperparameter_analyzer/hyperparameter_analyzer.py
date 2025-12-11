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
from contextlib import contextmanager
from modules.base.base_engine import BaseEngine
from utils.file_io import save_dataframe
from utils import constants

class HyperparameterAnalyzer(BaseEngine):
    """
    Analyzes HPO results with comprehensive visualizations.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config, logger)
        self.optimal_top_percent = config.get('hpo_analysis', {}).get('optimal_top_percent', 10)
        self.highlighting_styles = config.get('hpo_analysis', {}).get('highlighting_styles', ['box', 'stars', 'overlay'])
        self.cv_metrics = ['cv_cmae_deg_mean', 'cv_crmse_deg_mean', 'cv_max_error_deg_mean']
        self.val_metrics = ['val_cmae_deg', 'val_crmse_deg', 'val_max_error_deg']
        self.test_metrics = ['test_cmae_deg', 'test_crmse_deg', 'test_max_error_deg']
        self.primary_metric = 'cv_cmae_deg_mean'

    def _get_engine_directory_name(self) -> str:
        return constants.HPO_ANALYSIS_DIR

    @contextmanager
    def _plot_context(self):
        original_rcParams = plt.rcParams.copy()
        original_seaborn_style = sns.axes_style()
        try:
            yield
        finally:
            plt.rcParams.update(original_rcParams)
            sns.set_style(original_seaborn_style)
            plt.close('all')
            
    def execute(self, hpo_results_file: Path) -> None:
        """Execute full analysis on HPO results for a specific round."""
        if not hpo_results_file.exists():
            self.logger.warning(f"HPO results file not found at {hpo_results_file}. Skipping analysis.")
            return

        self.logger.info(f"Starting Hyperparameter Analysis on {hpo_results_file.name}...")
        
        # Parquet-first; fall back to Excel for legacy files
        df = pd.read_parquet(hpo_results_file) if hpo_results_file.suffix.lower() == ".parquet" else pd.read_excel(hpo_results_file)
        
        available_cv = [m for m in self.cv_metrics if m in df.columns]
        available_val = [m for m in self.val_metrics if m in df.columns]
        available_test = [m for m in self.test_metrics if m in df.columns]
        
        if not available_cv:
            self.logger.error(f"No CV metrics found in {hpo_results_file}. Cannot proceed with HPO analysis.")
            return
            
        self.logger.info(f"Found metrics - CV: {len(available_cv)}, Val: {len(available_val)}, Test: {len(available_test)}")

        df = self._preprocess_data(df)
        
        self._generate_optimal_ranges_report_multisheet(df)
        self._generate_summary_report(df, available_cv, available_val, available_test)
        
        models = df['model_name'].unique() if 'model_name' in df.columns else df['model'].unique()
        
        for model in models:
            model_col = 'model_name' if 'model_name' in df.columns else 'model'
            model_df = df[df[model_col] == model]
            
            self.logger.info(f"Generating visualizations for {model}...")
            
            top_configs = self._get_top_configs(model_df, self.optimal_top_percent)
            
            with self._plot_context():
                for metric_type, metrics in [('CV', available_cv), ('Val', available_val), ('Test', available_test)]:
                    for metric in metrics:
                        self._visualize_model_landscape(model_df, top_configs, model, metric, metric_type)
            
        self.logger.info(f"HPO Analysis complete. Artifacts in {self.output_dir}")

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        param_cols = [c for c in df.columns if c.startswith('param_')]
        for col in param_cols:
            df[col] = df[col].apply(lambda x: "None" if pd.isna(x) or x is None else x)
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
        return df

    def _get_top_configs(self, df: pd.DataFrame, top_percent: float) -> pd.DataFrame:
        df_sorted = df.sort_values(self.primary_metric, ascending=True)
        top_n = max(1, int(len(df) * (top_percent / 100.0)))
        return df_sorted.head(top_n)

    def _generate_optimal_ranges_report_multisheet(self, df: pd.DataFrame) -> None:
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        output_file = self.output_dir / "optimal_ranges.parquet"
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
        summary_df = pd.DataFrame(summary_data)
        save_dataframe(summary_df, output_file, excel_copy=excel_copy, index=False)
        
        models = df['model_name'].unique() if 'model_name' in df.columns else df['model'].unique()
        for model in models:
            model_col = 'model_name' if 'model_name' in df.columns else 'model'
            model_df = df[df[model_col] == model]
            top_model = self._get_top_configs(model_df, self.optimal_top_percent)
            model_data = []
            for col in param_cols:
                if col not in model_df.columns: continue
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
            sheet_name = model[:31]
            model_df_out = pd.DataFrame(model_data)
            save_dataframe(model_df_out, self.output_dir / f"optimal_ranges_{sheet_name}.parquet", excel_copy=excel_copy, index=False)
        self.logger.info(f"Generated optimal ranges artifacts (Parquet-first) in {self.output_dir}")

    def _generate_summary_report(self, df: pd.DataFrame, cv_metrics: List[str], 
                                 val_metrics: List[str], test_metrics: List[str]) -> None:
        models = df['model_name'].unique() if 'model_name' in df.columns else df['model'].unique()
        model_col = 'model_name' if 'model_name' in df.columns else 'model'
        summary = []
        for model in models:
            model_df = df[df[model_col] == model]
            best_row = model_df.sort_values(self.primary_metric, ascending=True).iloc[0]
            row = {'Model': model, 'Total Configs': len(model_df)}
            for m in cv_metrics:
                if m in best_row: row[f'Best {m}'] = best_row[m]
            for m in val_metrics:
                if m in best_row: row[f'Best {m}'] = best_row[m]
            for m in test_metrics:
                if m in best_row: row[f'Best {m}'] = best_row[m]
            summary.append(row)
        summary_df = pd.DataFrame(summary)
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        save_dataframe(summary_df, self.output_dir / "summary_report.parquet", excel_copy=excel_copy, index=False)
        self.logger.info("Generated summary report (Parquet-first)")

    def _visualize_model_landscape(self, df: pd.DataFrame, top_df: pd.DataFrame, 
                                   model: str, metric: str, metric_type: str):
        param_cols = [c for c in df.columns if c.startswith('param_')]
        if len(param_cols) < 2: return
        viz_dir = self.output_dir / "visualizations" / model / metric_type / metric
        viz_dir.mkdir(parents=True, exist_ok=True)
        import itertools
        pairs = list(itertools.combinations(param_cols, 2))
        for p1, p2 in pairs[:10]:
            self._plot_parameter_pair(df, top_df, p1, p2, metric, viz_dir)

    def _plot_parameter_pair(self, df: pd.DataFrame, top_df: pd.DataFrame,
                             p1: str, p2: str, z_col: str, output_dir: Path):
        is_num_1 = self._is_numeric_param(df[p1])
        is_num_2 = self._is_numeric_param(df[p2])
        name1, name2 = p1.replace('param_', ''), p2.replace('param_', '')
        name1_sanitized = "".join([c if c.isalnum() else '_' for c in name1])
        name2_sanitized = "".join([c if c.isalnum() else '_' for c in name2])
        base_name = f"{name1_sanitized}_vs_{name2_sanitized}"
        for style in self.highlighting_styles:
            heatmap_dir = output_dir / "heatmap" / style
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            self._plot_heatmap(df, top_df, p1, p2, z_col, heatmap_dir / f"{base_name}.png", style)
        if is_num_1 and is_num_2:
            for style in self.highlighting_styles:
                contour_dir = output_dir / "contour" / style
                contour_dir.mkdir(parents=True, exist_ok=True)
                self._plot_contour_2d(df, top_df, p1, p2, z_col, contour_dir / f"{base_name}_full.png", style, zoom=False)
                self._plot_contour_2d(df, top_df, p1, p2, z_col, contour_dir / f"{base_name}_optimal.png", style, zoom=True)
            for style in self.highlighting_styles:
                surface_dir = output_dir / "3d_surface" / style
                surface_dir.mkdir(parents=True, exist_ok=True)
                self._plot_3d_surface(df, top_df, p1, p2, z_col, surface_dir / f"{base_name}_full.png", style, zoom=False)
                self._plot_3d_surface(df, top_df, p1, p2, z_col, surface_dir / f"{base_name}_optimal.png", style, zoom=True)

    def _is_numeric_param(self, series: pd.Series) -> bool:
        if np.issubdtype(series.dtype, np.number): return True
        try:
            unique_vals = series.unique()
            for val in unique_vals:
                if val == "None": return False
                if isinstance(val, str) and not val.replace('.', '').replace('-', '').isdigit(): return False
            return True
        except: return False

    def _plot_heatmap(self, df: pd.DataFrame, top_df: pd.DataFrame, x_col: str, y_col: str, z_col: str, output_path: Path, style: str):
        agg_df = df.groupby([x_col, y_col], as_index=False)[z_col].mean()
        try:
            pivot = agg_df.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc='mean')
        except Exception as e:
            self.logger.warning(f"Pivot failed for {x_col} vs {y_col}: {e}")
            return
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(pivot, annot=True, fmt=".2f", cmap='viridis_r', cbar_kws={'label': z_col})
        if style == 'box': self._add_heatmap_box_highlight(ax, pivot, top_df, x_col, y_col)
        elif style == 'stars': self._add_heatmap_star_highlight(ax, pivot, top_df, x_col, y_col)
        elif style == 'overlay': self._add_heatmap_overlay_highlight(ax, pivot, top_df, x_col, y_col)
        clean_x, clean_y = x_col.replace('param_', ''), y_col.replace('param_', '')
        plt.xlabel(clean_x); plt.ylabel(clean_y)
        plt.title(f"Heatmap [{style}]: {clean_x} vs {clean_y}\n{z_col}")
        plt.tight_layout(); plt.savefig(output_path, dpi=200, bbox_inches='tight'); plt.close()

    def _add_heatmap_box_highlight(self, ax, pivot, top_df, x_col, y_col):
        opt_x_vals, opt_y_vals = top_df[x_col].unique(), top_df[y_col].unique()
        for i, y_val in enumerate(pivot.index):
            for j, x_val in enumerate(pivot.columns):
                if str(x_val) in [str(v) for v in opt_x_vals] and str(y_val) in [str(v) for v in opt_y_vals]:
                    ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', linewidth=3))

    def _add_heatmap_star_highlight(self, ax, pivot, top_df, x_col, y_col):
        opt_x_vals, opt_y_vals = top_df[x_col].unique(), top_df[y_col].unique()
        for i, y_val in enumerate(pivot.index):
            for j, x_val in enumerate(pivot.columns):
                if str(x_val) in [str(v) for v in opt_x_vals] and str(y_val) in [str(v) for v in opt_y_vals]:
                    # Use matplotlib scatter with star marker instead of Unicode text
                    ax.scatter(j + 0.5, i + 0.5, marker='*', s=200, color='gold', edgecolors='red', linewidths=1.5, zorder=10)

    def _add_heatmap_overlay_highlight(self, ax, pivot, top_df, x_col, y_col):
        opt_x_vals, opt_y_vals = top_df[x_col].unique(), top_df[y_col].unique()
        for i, y_val in enumerate(pivot.index):
            for j, x_val in enumerate(pivot.columns):
                if str(x_val) in [str(v) for v in opt_x_vals] and str(y_val) in [str(v) for v in opt_y_vals]:
                    ax.add_patch(Rectangle((j, i), 1, 1, fill=True, facecolor='lime', alpha=0.3, edgecolor='none'))

    def _plot_contour_2d(self, df: pd.DataFrame, top_df: pd.DataFrame, x_col: str, y_col: str, z_col: str, output_path: Path, style: str, zoom: bool = False):
        plot_df = top_df.copy() if zoom else df.copy()
        title_suffix = "(Optimal Range Only)" if zoom else "(Full Range)"
        x, y, z = plot_df[x_col].values, plot_df[y_col].values, plot_df[z_col].values
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2: return 
        try:
            xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
            zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
        except Exception as e:
            self.logger.warning(f"Contour interpolation failed: {e}"); return
        plt.figure(figsize=(12, 10))
        try:
            cp = plt.contourf(xi, yi, zi, levels=20, cmap='viridis_r')
            plt.colorbar(cp, label=z_col)
            plt.contour(xi, yi, zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        except Exception as e:
            self.logger.warning(f"Contour plotting failed: {e}"); plt.close(); return
        plt.scatter(x, y, c='white', s=30, alpha=0.6, edgecolors='black', linewidths=1)
        if not zoom and style == 'box':
            opt_x_min, opt_x_max = top_df[x_col].min(), top_df[x_col].max()
            opt_y_min, opt_y_max = top_df[y_col].min(), top_df[y_col].max()
            width = opt_x_max - opt_x_min if opt_x_max > opt_x_min else (x.max()-x.min())*0.05
            height = opt_y_max - opt_y_min if opt_y_max > opt_y_min else (y.max()-y.min())*0.05
            if width == 0: width = 0.1
            if height == 0: height = 0.1
            rect = Rectangle((opt_x_min, opt_y_min), width, height, linewidth=3, edgecolor='red', facecolor='none', label='Optimal Region')
            plt.gca().add_patch(rect); plt.legend()
        elif not zoom and style == 'overlay':
            opt_x_min, opt_x_max = top_df[x_col].min(), top_df[x_col].max()
            opt_y_min, opt_y_max = top_df[y_col].min(), top_df[y_col].max()
            plt.axvspan(opt_x_min, opt_x_max, alpha=0.2, color='lime')
            plt.axhspan(opt_y_min, opt_y_max, alpha=0.2, color='lime')
        elif not zoom and style == 'stars':
            opt_x, opt_y = top_df[x_col].values, top_df[y_col].values
            plt.scatter(opt_x, opt_y, marker='*', s=200, c='gold', edgecolors='red', linewidths=2, label='Optimal Configs'); plt.legend()
        clean_x = x_col.replace('param_', '').replace('_', ' ').title()
        clean_y = y_col.replace('param_', '').replace('_', ' ').title()
        plt.xlabel(clean_x, fontsize=12)
        plt.ylabel(clean_y, fontsize=12)
        plt.title(
            f"Hyperparameter Contour Map: {clean_x} vs {clean_y} {title_suffix}\n{z_col}",
            fontsize=13,
            fontweight='bold'
        )
        plt.tight_layout(); plt.savefig(output_path, dpi=200, bbox_inches='tight'); plt.close()

    def _plot_3d_surface(self, df: pd.DataFrame, top_df: pd.DataFrame, x_col: str, y_col: str, z_col: str, output_path: Path, style: str, zoom: bool = False):
        plot_df = top_df.copy() if zoom else df.copy()
        title_suffix = "(Optimal Range Only)" if zoom else "(Full Range)"
        x, y, z = plot_df[x_col].values, plot_df[y_col].values, plot_df[z_col].values
        fig = plt.figure(figsize=(14, 12))
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            plt.close(fig); return
        ax = fig.add_subplot(111, projection='3d')
        try:
            xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
            zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
        except Exception as e:
            self.logger.warning(f"3D interpolation failed: {e}"); plt.close(fig); return
        XI, YI = np.meshgrid(xi, yi)
        surf = ax.plot_surface(XI, YI, zi, cmap='viridis_r', edgecolor='none', alpha=0.7)
        if not zoom:
            is_optimal = df.index.isin(top_df.index)
            ax.scatter(x[~is_optimal], y[~is_optimal], z[~is_optimal], c='red', marker='o', s=20, alpha=0.4, label='Other Configs')
            if style == 'stars': ax.scatter(x[is_optimal], y[is_optimal], z[is_optimal], c='lime', marker='*', s=100, alpha=1.0, edgecolors='black', linewidths=1, label='Optimal Configs')
            else: ax.scatter(x[is_optimal], y[is_optimal], z[is_optimal], c='lime', marker='o', s=50, alpha=1.0, label='Optimal Configs')
            if style in ['box', 'overlay']:
                opt_x_min, opt_x_max = top_df[x_col].min(), top_df[x_col].max()
                opt_y_min, opt_y_max = top_df[y_col].min(), top_df[y_col].max()
                z_floor = z.min()
                verts = [[(opt_x_min, opt_y_min, z_floor), (opt_x_max, opt_y_min, z_floor), (opt_x_max, opt_y_max, z_floor), (opt_x_min, opt_y_max, z_floor)]]
                poly = Poly3DCollection(verts, alpha=0.3, facecolor='lime', edgecolor='darkgreen')
                ax.add_collection3d(poly)
        else: ax.scatter(x, y, z, c='lime', marker='o', s=40, alpha=0.8)
        clean_x, clean_y = x_col.replace('param_', ''), y_col.replace('param_', '')
        ax.set_xlabel(clean_x); ax.set_ylabel(clean_y); ax.set_zlabel(z_col)
        ax.set_title(f"3D Surface [{style}]: {clean_x} vs {clean_y} {title_suffix}\n{z_col}")
        fig.colorbar(surf, shrink=0.5, aspect=5, label=z_col); ax.legend(loc='upper left')
        plt.savefig(output_path, dpi=200, bbox_inches='tight'); plt.close(fig)
