import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from modules.base.base_engine import BaseEngine
from utils.error_handling import handle_engine_errors
from utils.file_io import save_dataframe, read_dataframe
from modules.evaluation_engine.stat_tests import compare_rounds
from modules.evaluation_engine.cv_analysis import cv_fold_consistency, overfitting_gaps
from utils import constants

from utils.circular_metrics import compute_cmae, compute_crmse

class EvaluationEngine(BaseEngine):
    """
    Computes comprehensive performance metrics for model predictions.
    Includes circular metrics, accuracy bands, component metrics, and extreme sample identification.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config, logger)
        eval_cfg = config.get("evaluation", {})
        self.bootstrap_samples = eval_cfg.get("bootstrap_samples", 0)
        self.bootstrap_alpha = eval_cfg.get("bootstrap_alpha", 0.05)
        
    def _get_engine_directory_name(self) -> str:
        return constants.EVALUATION_DIR

    @handle_engine_errors("Evaluation")
    def execute(self, predictions: pd.DataFrame, split_name: str, run_id: str) -> Dict[str, float]:
        """
        Compute metrics and save evaluation reports.
        
        Parameters:
            predictions: DataFrame containing true/pred values and errors.
            split_name: 'val' or 'test'.
            run_id: Run identifier.
            
        Returns:
            dict: Dictionary of computed metrics.
        """
        self.logger.info(f"Starting Evaluation for {split_name} set...")
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        
        # 1. Compute All Metrics
        metrics = self.compute_metrics(predictions)
        
        # 2. Identify Extremes (Best/Worst samples)
        # FIX #17: Implemented _identify_extremes method below to prevent AttributeError
        best_df, worst_df = self._identify_extremes(predictions)
        
        # 3. Save Artifacts
        # Save Metrics Table
        metrics_df = pd.DataFrame([metrics])
        metrics_path = self.output_dir / f"metrics_{split_name}.parquet"
        save_dataframe(metrics_df, metrics_path, excel_copy=excel_copy, index=False)

        # 3a. Naive baseline comparison (circular mean/median predictors)
        naive_df = self._compute_naive_baselines(predictions)
        if not naive_df.empty:
            save_dataframe(
                naive_df,
                self.output_dir / f"naive_baseline_comparison_{split_name}.parquet",
                excel_copy=excel_copy,
                index=False,
            )

        # Industry baseline comparison (configurable)
        self._compute_industry_baseline(predictions, split_name, metrics)
        
        # Save Extremes (useful for debugging outlier cases)
        best_path = self.output_dir / f"best_10_samples_{split_name}.parquet"
        worst_path = self.output_dir / f"worst_10_samples_{split_name}.parquet"
        save_dataframe(best_df, best_path, excel_copy=excel_copy, index=False)
        save_dataframe(worst_df, worst_path, excel_copy=excel_copy, index=False)

        # Optional bootstrap confidence intervals
        if self.bootstrap_samples and self.bootstrap_samples > 0:
            boot_df = self._bootstrap_metrics(predictions, split_name)
            if boot_df is not None and not boot_df.empty:
                save_dataframe(
                    boot_df,
                    self.output_dir / f"bootstrap_metrics_{split_name}.parquet",
                    excel_copy=excel_copy,
                    index=False,
                )

        cmae_val = metrics.get('cmae', 0.0)
        self.logger.info(f"Evaluation complete. {split_name} CMAE: {cmae_val:.4f}°")
        return metrics

    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate full suite of metrics: Circular, Linear, Bands, Components.
        """
        if df.empty:
            self.logger.warning("Empty predictions dataframe provided for evaluation.")
            return {}

        # Extract columns
        true_angle = df['true_angle'].values
        pred_angle = df['pred_angle'].values
        abs_error = df['abs_error'].values
        error = df['error'].values
        
        true_sin = df['true_sin'].values
        pred_sin = df['pred_sin'].values
        true_cos = df['true_cos'].values
        pred_cos = df['pred_cos'].values
        
        metrics = {}
        
        # --- 1. Circular Metrics ---
        metrics['cmae'] = compute_cmae(true_angle, pred_angle)
        metrics['crmse'] = compute_crmse(true_angle, pred_angle)
        
        # --- 2. Linear Error Statistics ---
        metrics['max_error'] = np.max(abs_error)
        metrics['mean_error'] = np.mean(error)  # Bias (Signed)
        metrics['std_error'] = np.std(error)
        metrics['median_abs_error'] = np.median(abs_error)
        
        # --- 3. Accuracy Bands ---
        # Returns keys like 'accuracy_at_5deg' scaled to 0-100%
        metrics.update(self._compute_accuracy_bands(abs_error))
        
        # --- 4. Percentiles ---
        # FIX #77: Use nanpercentile to handle potential NaNs safely
        for p in [50, 75, 90, 95, 99]:
            metrics[f'percentile_{p}'] = np.nanpercentile(abs_error, p)
            
        # --- 5. Component Metrics (Sin/Cos) ---
        metrics['mae_sin'] = np.mean(np.abs(true_sin - pred_sin))
        metrics['rmse_sin'] = np.sqrt(np.mean((true_sin - pred_sin)**2))
        metrics['mae_cos'] = np.mean(np.abs(true_cos - pred_cos))
        metrics['rmse_cos'] = np.sqrt(np.mean((true_cos - pred_cos)**2))
        
        # Metadata
        metrics['n_samples'] = len(df)
        
        return metrics

    def _compute_accuracy_bands(self, abs_error: np.ndarray) -> Dict[str, float]:
        """Calculate percentage of predictions within specific error tolerances."""
        bands = [0, 5, 10, 20]
        results = {}
        total = len(abs_error)
        
        # FIX #21: Prevent Division by Zero
        if total == 0:
            for b in bands:
                results[f'accuracy_at_{b}deg'] = 0.0
            return results

        for b in bands:
            if b == 0:
                count = np.sum(abs_error == 0)
            else:
                count = np.sum(abs_error <= b)
            
            # FIX #1: Changed multiplier from 10 to 100 for correct percentage
            results[f'accuracy_at_{b}deg'] = (count / total) * 100
            
        return results

    def _identify_extremes(self, df: pd.DataFrame, n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify best and worst predictions based on absolute error.
        Fixes Critical Bug #17 (Method previously missing).
        """
        if df.empty or 'abs_error' not in df.columns:
            return pd.DataFrame(), pd.DataFrame()
            
        # Sort by absolute error
        sorted_df = df.sort_values('abs_error', ascending=True)
        
        # Best = Lowest error (Top of sorted)
        best_df = sorted_df.head(n).copy()
        
        # Worst = Highest error (Bottom of sorted, reversed)
        worst_df = sorted_df.tail(n).iloc[::-1].copy()
        
        return best_df, worst_df

    def _compute_naive_baselines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute simple naive predictors (circular mean and median) for comparison.
        """
        required = {'true_angle'}
        if df.empty or not required.issubset(df.columns):
            return pd.DataFrame()

        true_angle = df['true_angle'].values

        # Circular mean using sin/cos components
        true_sin = np.sin(np.radians(true_angle))
        true_cos = np.cos(np.radians(true_angle))
        mean_angle = np.degrees(np.arctan2(true_sin.mean(), true_cos.mean())) % 360

        # Circular median approximation: median of angles mapped to unit circle then atan2 of medians
        median_sin = np.median(true_sin)
        median_cos = np.median(true_cos)
        median_angle = np.degrees(np.arctan2(median_sin, median_cos)) % 360

        def circular_abs_error(targets, pred_angle):
            diff = np.abs(((targets - pred_angle + 180) % 360) - 180)
            return diff

        rows = []
        for label, pred_ang in [("circular_mean", mean_angle), ("circular_median", median_angle)]:
            errs = circular_abs_error(true_angle, pred_ang)
            rows.append({
                "predictor": label,
                "pred_angle": pred_ang,
                "cmae": float(np.mean(errs)),
                "median_abs_error": float(np.median(errs)),
                "max_error": float(np.max(errs)),
            })

        return pd.DataFrame(rows)

    def _bootstrap_metrics(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """
        Bootstrap confidence intervals for key metrics (CMAE, CRMSE, median_abs_error).
        """
        if df.empty:
            self.logger.warning("Bootstrap skipped: empty predictions.")
            return pd.DataFrame()

        n = len(df)
        rng = np.random.default_rng(self.config.get("execution", {}).get("seed"))
        metrics_list = []

        for _ in range(self.bootstrap_samples):
            sample_idx = rng.integers(0, n, size=n)
            sample = df.iloc[sample_idx]
            errs = sample['abs_error'].values
            preds_angle = sample['pred_angle'].values
            true_angle = sample['true_angle'].values
            metrics_list.append({
                "cmae": compute_cmae(true_angle, preds_angle),
                "crmse": compute_crmse(true_angle, preds_angle),
                "median_abs_error": float(np.median(errs)),
            })

        boot_df = pd.DataFrame(metrics_list)
        alpha = self.bootstrap_alpha
        summary = []
        for col in boot_df.columns:
            lower = boot_df[col].quantile(alpha / 2)
            upper = boot_df[col].quantile(1 - alpha / 2)
            summary.append({
                "split": split_name,
                "metric": col,
                "bootstrap_samples": self.bootstrap_samples,
                "alpha": alpha,
                "mean": float(boot_df[col].mean()),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
            })

        return pd.DataFrame(summary)

    def summarize_cv_and_overfitting(
        self,
        cv_scores: Dict[str, np.ndarray],
        train_scores: Dict[str, float],
        val_scores: Dict[str, float],
        output_dir: Path,
        excel_copy: bool = False,
    ) -> None:
        """Persist CV fold consistency and overfitting gap summaries."""
        cv_df = cv_fold_consistency(cv_scores)
        if not cv_df.empty:
            save_dataframe(cv_df, output_dir / "cv_fold_consistency.parquet", excel_copy=excel_copy, index=False)

        gap_df = overfitting_gaps(train_scores, val_scores)
        if not gap_df.empty:
            save_dataframe(gap_df, output_dir / "overfitting_gaps.parquet", excel_copy=excel_copy, index=False)

    def compare_splits(
        self,
        baseline_errors: pd.Series,
        candidate_errors: pd.Series,
        alpha: float = 0.05,
        output_path: Path = None,
        excel_copy: bool = False,
    ) -> Dict[str, Any]:
        """
        Compare two error distributions for statistical significance.
        """
        result = compare_rounds(baseline_errors.values, candidate_errors.values, alpha=alpha)
        if output_path:
            save_dataframe(pd.DataFrame([result]), output_path, excel_copy=excel_copy, index=False)
        return result

    def _compute_metrics_flexible(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute metrics with graceful degradation if some columns are missing.
        """
        metrics: Dict[str, float] = {}
        cols = set(df.columns)
        # Full metric suite if all columns available
        required_full = {'true_angle', 'pred_angle', 'abs_error', 'true_sin', 'true_cos', 'pred_sin', 'pred_cos'}
        if required_full.issubset(cols):
            return self.compute_metrics(df)

        if 'abs_error' in cols:
            metrics['cmae'] = float(df['abs_error'].mean())
            metrics['median_abs_error'] = float(df['abs_error'].median())
            metrics['max_error'] = float(df['abs_error'].max())

        if {'true_angle', 'pred_angle'}.issubset(cols):
            true_angle = df['true_angle'].values
            pred_angle = df['pred_angle'].values
            metrics['crmse'] = compute_crmse(true_angle, pred_angle)

        if metrics and 'n_samples' not in metrics:
            metrics['n_samples'] = len(df)

        return metrics

    def _compute_industry_baseline(self, predictions: pd.DataFrame, split_name: str, model_metrics: Dict[str, float]) -> None:
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        
        industry_baseline_cfg = self.config.get("evaluation", {}).get("industry_baseline")

        if industry_baseline_cfg is None:
            self.logger.info("Industry baseline comparison skipped (no config provided in 'evaluation.industry_baseline').")
            return

        cfg = industry_baseline_cfg
        label = cfg.get("label", "industry_baseline")
        rows = []

        # Option 1: use provided metrics directly
        provided_metrics = cfg.get("metrics")
        if isinstance(provided_metrics, dict):
            for metric_name, value in provided_metrics.items():
                rows.append({
                    "model": label,
                    "metric": metric_name,
                    "value": value,
                    "delta_vs_model": value - model_metrics.get(metric_name, np.nan),
                    "source": "provided_metrics"
                })

        # Option 2: load baseline predictions and compute metrics
        elif cfg.get("predictions_path"):
            try:
                baseline_df = read_dataframe(Path(cfg["predictions_path"]))
                baseline_metrics = self._compute_metrics_flexible(baseline_df)
                for metric_name, value in baseline_metrics.items():
                    rows.append({
                        "model": label,
                        "metric": metric_name,
                        "value": value,
                        "delta_vs_model": value - model_metrics.get(metric_name, np.nan),
                        "source": "predictions_path"
                    })
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"Industry baseline metrics could not be computed: {exc}")

        # Option 3: fall back to naive baseline
        else:
            naive_df = self._compute_naive_baselines(predictions)
            if not naive_df.empty:
                for _, row in naive_df.iterrows():
                    rows.append({
                        "model": row.get("predictor", "naive"),
                        "metric": "cmae",
                        "value": row.get("cmae"),
                        "delta_vs_model": row.get("cmae") - model_metrics.get("cmae", np.nan),
                        "source": "naive_baseline"
                    })

        if not rows:
            self.logger.info("Industry baseline comparison skipped (no config or inputs).")
            return

        out_df = pd.DataFrame(rows)
        save_dataframe(
            out_df,
            self.output_dir / f"industry_baseline_comparison_{split_name}.parquet",
            excel_copy=excel_copy,
            index=False,
        )
