import pandas as pd
import numpy as np
import logging
import os
import json
from pathlib import Path
from scipy import stats
from typing import Optional, Dict, Any
from modules.base.base_engine import BaseEngine
from utils.error_handling import handle_engine_errors
from utils.file_io import save_dataframe
from modules.error_analysis_engine.safety_analysis import safety_threshold_summary
from utils import constants

class ErrorAnalysisEngine(BaseEngine):
    """
    Performs deep-dive analysis on model errors.
    Identifies specific failure modes, statistical outliers, and feature correlations.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config, logger)
        self.ea_config = config.get('error_analysis', {})
        
    def _get_engine_directory_name(self) -> str:
        return constants.ERROR_ANALYSIS_ENGINE_DIR

    @handle_engine_errors("Error Analysis")
    def execute(self, 
                predictions: pd.DataFrame, 
                features: pd.DataFrame, 
                split_name: str) -> Dict[str, Any]:
        """
        Execute full error analysis suite.
        
        Parameters:
            predictions: DataFrame with 'abs_error', 'error', 'index'.
            features: DataFrame containing input features (must have matching index).
            split_name: 'val' or 'test'. Used for logging.
        """
        if not self.ea_config.get('enabled', True):
            self.logger.info("Error Analysis disabled in config.")
            return {}

        self.logger.info(f"Starting Error Analysis for {split_name} set...")

        output_dir = self.output_dir / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        
        # 2. Threshold Analysis
        self._analyze_thresholds(predictions, output_dir, excel_copy)
        
        # 3. Statistical Outlier Detection
        self._detect_outliers(predictions, output_dir, excel_copy)
        
        # 4. Feature Correlations
        if features is not None and not features.empty:
            analysis_df = pd.concat([predictions.set_index('row_index'), features], axis=1, join='inner')
            self._analyze_correlations(analysis_df, output_dir, excel_copy)
            self._generate_prediction_explanations(analysis_df, output_dir, excel_copy)
        else:
            self.logger.warning("Features DataFrame missing or empty. Skipping correlation analysis.")
            
        # 5. Bias Analysis
        self._analyze_bias(predictions, output_dir, excel_copy)
        # 6. Safety Threshold Analysis
        self._analyze_safety_thresholds(predictions, output_dir, excel_copy)
        # 7. Extreme Conditions (Hs/angle) Analysis
        self._analyze_extremes(predictions, output_dir, excel_copy)
        
        self.logger.info(f"Error Analysis complete for {split_name}.")
        return {"status": "complete", "output_dir": str(output_dir)}

    def _analyze_thresholds(self, df: pd.DataFrame, output_dir: Path, excel_copy: bool) -> None:
        """Identify samples exceeding specific error thresholds."""
        if df.empty:
            self.logger.warning("Empty predictions dataframe provided for threshold analysis. Skipping.")
            return

        thresholds = self.ea_config.get('error_thresholds', [5, 10, 20])
        
        summary = []
        for t in thresholds:
            high_error_df = df[df['abs_error'] > t].copy()
            count = len(high_error_df)
            pct = (count / len(df)) * 100
            
            summary.append({
                'threshold': t,
                'count': count,
                'percentage': pct
            })
            
            if count > 0:
                save_path = output_dir / f"samples_error_gt_{t}deg.parquet"
                save_dataframe(high_error_df.sort_values('abs_error', ascending=False), save_path, excel_copy=excel_copy, index=False)
        
        save_dataframe(pd.DataFrame(summary), output_dir / "threshold_summary.parquet", excel_copy=excel_copy, index=False)

    def _detect_outliers(self, df: pd.DataFrame, output_dir: Path, excel_copy: bool) -> None:
        """Detect statistical outliers using Z-score or IQR."""
        method = self.ea_config.get('outlier_detection', '3sigma')
        errors = df['abs_error']
        
        if method == '3sigma':
            if errors.std() == 0:
                self.logger.info("Absolute errors have zero variance. No outliers detected by Z-score method.")
                return
            z_scores = np.abs(stats.zscore(errors))
            outliers = df[z_scores > 3].copy()
        elif method == 'iqr':
            Q1 = errors.quantile(0.25)
            Q3 = errors.quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[errors > (Q3 + 1.5 * IQR)].copy()
        else:
            self.logger.warning(f"Unknown outlier detection method: {method}")
            return
            
        if not outliers.empty:
            save_path = output_dir / f"statistical_outliers_{method}.parquet"
            save_dataframe(outliers.sort_values('abs_error', ascending=False), save_path, excel_copy=excel_copy, index=False)
            self.logger.info(f"Detected {len(outliers)} statistical outliers using {method}.")

    def _analyze_correlations(self, df: pd.DataFrame, output_dir: Path, excel_copy: bool) -> None:
        """Check correlation between absolute error and input features."""
        if not self.ea_config.get('correlation_analysis', True):
            return

        ignore_cols = ['true_angle', 'pred_angle', 'error', 'row_index', 'true_sin', 'true_cos', 'pred_sin', 'pred_cos']
        target_col = 'abs_error'
        
        if target_col not in df.columns:
            return

        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
        
        if target_col not in numeric_df.columns:
            return

        correlations = numeric_df.corrwith(numeric_df[target_col])
        correlations = correlations.dropna().sort_values(ascending=False)
        correlations = correlations.drop(labels=[target_col] + [c for c in ignore_cols if c in correlations.index], errors='ignore')
        
        corr_df = correlations.reset_index()
        corr_df.columns = ['Feature', 'Correlation_with_AbsError']
        save_dataframe(corr_df, output_dir / "error_feature_correlations.parquet", excel_copy=excel_copy, index=False)

    def _generate_prediction_explanations(self, df: pd.DataFrame, output_dir: Path, excel_copy: bool) -> None:
        """
        Provide lightweight prediction explanations without SHAP.

        Uses correlations to rank features most associated with predictions
        and absolute error as a reporting-only guide.
        """
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        drop_cols = ['true_angle', 'pred_angle', 'error', 'abs_error', 'true_sin', 'true_cos', 'pred_sin', 'pred_cos']
        feature_cols = [c for c in numeric_df.columns if c not in drop_cols]
        if not feature_cols:
            return

        explanations = []
        pred_angle_series = numeric_df['pred_angle'] if 'pred_angle' in numeric_df else None
        abs_error_series = numeric_df['abs_error'] if 'abs_error' in numeric_df else None
        
        for col in feature_cols:
            series = numeric_df[col]
            # Ensure we have a Series, not DataFrame (in case of duplicate columns)
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            std_val = series.std()
            if pd.isna(std_val) or float(std_val) == 0.0:
                continue
            corr_pred = series.corr(pred_angle_series) if pred_angle_series is not None else np.nan
            corr_err = series.corr(abs_error_series) if abs_error_series is not None else np.nan
            explanations.append({
                "feature": col,
                "corr_with_pred_angle": corr_pred,
                "corr_with_abs_error": corr_err,
            })

        if not explanations:
            return

        exp_df = pd.DataFrame(explanations).sort_values(by="corr_with_abs_error", key=lambda s: s.abs(), ascending=False)
        save_dataframe(exp_df, output_dir / "prediction_explanations.parquet", excel_copy=excel_copy, index=False)

        top_k = exp_df.head(5).to_dict(orient="records")
        summary_path = output_dir / "prediction_explanations_summary.json"
        summary_path.write_text(json.dumps({"top_features_by_error": top_k}, indent=2))

    def _analyze_bias(self, df: pd.DataFrame, output_dir: Path, excel_copy: bool) -> None:
        """Analyze systematic bias (signed error)."""
        mean_bias = df['error'].mean()
        
        df['quadrant'] = pd.cut(df['true_angle'], bins=[0, 90, 180, 270, 360], 
                                labels=['Q1', 'Q2', 'Q3', 'Q4'], include_lowest=True)
        quad_bias = df.groupby('quadrant', observed=False)['error'].agg(['mean', 'std', 'count']).reset_index()
        
        save_dataframe(quad_bias, output_dir / "bias_analysis_by_quadrant.parquet", excel_copy=excel_copy, index=False)

    def _analyze_safety_thresholds(self, df: pd.DataFrame, output_dir: Path, excel_copy: bool) -> None:
        """Categorize errors into safety tiers and save summary."""
        if df.empty or 'abs_error' not in df.columns:
            return
        summary = safety_threshold_summary(df['abs_error'])
        save_dataframe(summary, output_dir / "safety_threshold_analysis.parquet", excel_copy=excel_copy, index=False)

    def _analyze_extremes(self, df: pd.DataFrame, output_dir: Path, excel_copy: bool) -> None:
        """Analyze performance at extreme sea states and angle boundaries."""
        hs_col = self.config.get('data', {}).get('hs_column', 'Hs_ft')
        if hs_col not in df.columns or df.empty:
            self.logger.info("Skipping extreme condition analysis: hs column missing or empty dataframe.")
            return

        hs = df[hs_col]
        low_thresh = np.nanpercentile(hs, 10)
        high_thresh = np.nanpercentile(hs, 90)

        def summarize(mask, label):
            subset = df[mask]
            if subset.empty:
                return {"segment": label, "n": 0}
            return {
                "segment": label,
                "n": int(len(subset)),
                "cmae": float(subset['abs_error'].mean()),
                "median_abs_error": float(subset['abs_error'].median()),
                "max_abs_error": float(subset['abs_error'].max()),
            }

        records = [
            summarize(hs <= low_thresh, f"low_hs_<=p10_{low_thresh:.2f}"),
            summarize(hs >= high_thresh, f"high_hs_>=p90_{high_thresh:.2f}"),
            summarize((df['true_angle'] <= 10) | (df['true_angle'] >= 350), "angle_boundary_0_360"),
        ]

        save_dataframe(pd.DataFrame(records), output_dir / "extreme_conditions_analysis.parquet", excel_copy=excel_copy, index=False)
