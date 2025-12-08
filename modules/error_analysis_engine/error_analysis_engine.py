import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from scipy import stats
from typing import Optional, Dict, Any

class ErrorAnalysisEngine:
    """
    Performs deep-dive analysis on model errors.
    Identifies specific failure modes, statistical outliers, and feature correlations.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.ea_config = config.get('error_analysis', {})
        
    def analyze(self, 
                predictions: pd.DataFrame, 
                features: pd.DataFrame, 
                split_name: str, 
                run_id: str) -> Dict[str, Any]:
        """
        Execute full error analysis suite.
        
        Parameters:
            predictions: DataFrame with 'abs_error', 'error', 'index'.
            features: DataFrame containing input features (must have matching index).
            split_name: 'val' or 'test'.
            run_id: Run identifier.
        """
        if not self.ea_config.get('enabled', True):
            self.logger.info("Error Analysis disabled in config.")
            return {}

        self.logger.info(f"Starting Error Analysis for {split_name} set...")
        
        # 1. Setup Directory
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "10_ERROR_ANALYSIS"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Threshold Analysis
        self._analyze_thresholds(predictions, output_dir)
        
        # 3. Statistical Outlier Detection
        self._detect_outliers(predictions, output_dir)
        
        # 4. Feature Correlations
        # Merge predictions with features based on index
        # Ensure we only analyze numeric features
        if features is not None and not features.empty:
            # Align indices
            analysis_df = pd.concat([predictions.set_index('row_index'), features], axis=1, join='inner')
            self._analyze_correlations(analysis_df, output_dir)
        else:
            self.logger.warning("Features DataFrame missing or empty. Skipping correlation analysis.")
            
        # 5. Bias Analysis
        self._analyze_bias(predictions, output_dir)
        
        self.logger.info(f"Error Analysis complete for {split_name}.")
        return {"status": "complete", "output_dir": str(output_dir)}

    def _analyze_thresholds(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Identify samples exceeding specific error thresholds."""
        # FIX #95: Early return if dataframe is empty.
        if df.empty:
            self.logger.warning("Empty predictions dataframe provided for threshold analysis. Skipping.")
            return

        thresholds = self.ea_config.get('error_thresholds', [5, 10, 20])
        
        summary = []
        for t in thresholds:
            high_error_df = df[df['abs_error'] > t].copy()
            count = len(high_error_df)
            pct = (count / len(df)) * 100 # len(df) is guaranteed > 0 here
            
            summary.append({
                'threshold': t,
                'count': count,
                'percentage': pct
            })
            
            if count > 0:
                # Save specific bad rows
                save_path = output_dir / f"samples_error_gt_{t}deg.xlsx"
                high_error_df.sort_values('abs_error', ascending=False).to_excel(save_path, index=False)
        
        # Save summary
        pd.DataFrame(summary).to_excel(output_dir / "threshold_summary.xlsx", index=False)

    def _detect_outliers(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Detect statistical outliers using Z-score or IQR."""
        method = self.ea_config.get('outlier_detection', '3sigma')
        errors = df['abs_error']
        
        if method == '3sigma':
            # FIX #23: Handle zero variance in errors to prevent RuntimeWarning/error in stats.zscore.
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
            save_path = output_dir / f"statistical_outliers_{method}.xlsx"
            outliers.sort_values('abs_error', ascending=False).to_excel(save_path, index=False)
            self.logger.info(f"Detected {len(outliers)} statistical outliers using {method}.")

    def _analyze_correlations(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Check correlation between absolute error and input features.
        Helps identify conditions (e.g., high Hs) that lead to errors.
        """
        if not self.ea_config.get('correlation_analysis', True):
            return

        # Drop non-feature columns
        ignore_cols = ['true_angle', 'pred_angle', 'error', 'row_index', 'true_sin', 'true_cos', 'pred_sin', 'pred_cos']
        target_col = 'abs_error'
        
        if target_col not in df.columns:
            return

        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Compute correlation with abs_error
        correlations = numeric_df.corrwith(numeric_df[target_col])
        
        # FIX #79: Drop NaN values from correlations (e.g., if a feature is constant).
        correlations = correlations.dropna().sort_values(ascending=False)
        
        # Filter out self-correlation and ignored columns
        correlations = correlations.drop(labels=[target_col] + [c for c in ignore_cols if c in correlations.index], errors='ignore')
        
        # Save results
        corr_df = correlations.reset_index()
        corr_df.columns = ['Feature', 'Correlation_with_AbsError']
        corr_df.to_excel(output_dir / "error_feature_correlations.xlsx", index=False)

    def _analyze_bias(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Analyze systematic bias (signed error)."""
        # Overall bias
        mean_bias = df['error'].mean()
        
        # Bias by Angle Quadrant (0-90, 90-180, etc.)
        df['quadrant'] = pd.cut(df['true_angle'], bins=[0, 90, 180, 270, 360], 
                                labels=['Q1', 'Q2', 'Q3', 'Q4'], include_lowest=True)
        quad_bias = df.groupby('quadrant', observed=False)['error'].agg(['mean', 'std', 'count']).reset_index()
        
        quad_bias.to_excel(output_dir / "bias_analysis_by_quadrant.xlsx", index=False)