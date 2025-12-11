import pandas as pd
import numpy as np
import logging
import json
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm  # Progress indicators

from modules.training_engine import TrainingEngine
from modules.evaluation_engine import EvaluationEngine
from utils.file_io import save_dataframe
from utils import constants
from utils.circular_metrics import reconstruct_angle, wrap_angle
from sklearn.inspection import permutation_importance

class FeatureEvaluator:
    """
    Performs Leave-One-Feature-Out (LOFO) evaluation.
    
    Logic:
    1. Train a Baseline Model using ALL active features.
    2. For each feature f in active_features:
       a. Create a view of data excluding f.
       b. Train a model using the SAME hyperparameters as baseline.
       c. Evaluate on Validation set.
       d. Calculate Impact = Metric(without f) - Metric(Baseline).
    
    This identifies which feature is 'safest' to remove (i.e., removing it causes 
    the least error increase, or even an error decrease).
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.train_engine = TrainingEngine(config, logger)
        self.eval_engine = EvaluationEngine(config, logger)
        fi_cfg = config.get("feature_importance", {})
        self.enable_perm_importance = fi_cfg.get("enable_permutation_importance", False)
        self.perm_repeats = fi_cfg.get("permutation_n_repeats", 5)
        self.enable_pdp = fi_cfg.get("enable_pdp", False)
        self.pdp_features = fi_cfg.get("pdp_features")  # optional list
        self.pdp_grid_points = fi_cfg.get("pdp_grid_points", 10)

    def evaluate_features(self, 
                          round_dir: Path, 
                          active_features: List[str], 
                          hyperparams: Dict[str, Any],
                          train_df: pd.DataFrame, 
                          val_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Executes the full LOFO loop.
        
        Args:
            round_dir: Path to the current ROUND_XXX folder.
            active_features: List of features currently in use.
            hyperparams: Optimal hyperparameters found in HPO step.
            train_df: Full training dataframe.
            val_df: Full validation dataframe.
            
        Returns:
            List of dictionaries containing metrics for each dropped feature.
        """
        self.logger.info(f"Starting LOFO Evaluation for {len(active_features)} features...")
        
        # 1. Train Baseline (Reference Point)
        # We need this to calculate "Impact" (Delta).
        self.logger.info("Training Baseline Model (All Active Features)...")
        baseline_model_config = {'model': self.config['models']['native'][0], 'params': hyperparams}
        
        # Filter DFs to only active features + targets
        # Note: TrainingEngine handles dropping non-feature cols defined in config, 
        # but here we must explicitly restrict to 'active_features'
        train_base = self._subset_df(train_df, active_features)
        
        baseline_model = self.train_engine.train(train_base, baseline_model_config, "baseline")

        # Predict on Val to get Baseline Metrics
        # (We bypass PredictionEngine to keep it lightweight, or use internal predict logic)
        # For speed in LOFO, we do direct prediction if possible, but let's stick to standard flow
        # to ensure scaling/wrapping is correct.
        X_val_base = self._subset_df(val_df, active_features).drop(columns=self._get_target_cols())
        preds_base = baseline_model.predict(X_val_base)

        # We need to format preds into a DF for EvaluationEngine
        df_preds_base = self._format_predictions(val_df, preds_base)
        baseline_metrics = self.eval_engine.compute_metrics(df_preds_base)

        # Optional: permutation importance (reporting only)
        if self.enable_perm_importance:
            try:
                self._compute_permutation_importance(
                    baseline_model,
                    X_val_base,
                    val_df,
                    round_dir / constants.ROUND_FEATURE_EVALUATION_DIR,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"Permutation importance skipped due to error: {exc}")

        # Optional: partial dependence (reporting only)
        if self.enable_pdp:
            try:
                self._compute_pdp(
                    baseline_model,
                    X_val_base,
                    val_df,
                    round_dir / constants.ROUND_FEATURE_EVALUATION_DIR,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"PDP skipped due to error: {exc}")

        self.logger.info(f"Baseline Validation CMAE: {baseline_metrics['cmae']:.4f}")

        # CRITICAL: Delete baseline model immediately after getting metrics
        # This prevents the most significant memory leak (models are large)
        del baseline_model, X_val_base, preds_base, df_preds_base, train_base
        gc.collect()

        # 2. LOFO Loop
        results = []
        output_dir = round_dir / constants.ROUND_FEATURE_EVALUATION_DIR / "minus_feature_models"
        output_dir.mkdir(parents=True, exist_ok=True)
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)

        # Progress bar for LOFO analysis
        with tqdm(total=len(active_features), desc="LOFO Analysis", unit="feature") as pbar:
            for i, feature_to_drop in enumerate(active_features):
                # A. Create Subset (Minus one feature)
                # Use list comprehension to create new list excluding feature_to_drop
                subset_features = [f for f in active_features if f != feature_to_drop]

                # Log periodically
                if i % 5 == 0:
                    self.logger.info(f"Evaluating drop of feature: {feature_to_drop} ({i+1}/{len(active_features)})")

                # B. Train Model
                # We assume TrainingEngine handles the 'y' columns automatically if they are in the DF
                train_subset = self._subset_df(train_df, subset_features)

                # Use a generic run_id so we don't spam logs/artifacts too much?
                # Actually, TrainingEngine saves artifacts. We might want to disable saving for LOFO
                # to save disk space, or save them into specific subfolders.
                # We temporarily override config to prevent massive artifact generation per feature if needed.
                # For strict reproducibility, we let it save but direct it to specific folder?
                # Ideally TrainingEngine supports a flag 'save_artifacts=False'.
                # Assuming it does (or we accept the I/O cost).

                model = self.train_engine.train(train_subset, baseline_model_config, f"lofo_{feature_to_drop}")

                # C. Evaluate
                X_val_subset = self._subset_df(val_df, subset_features).drop(columns=self._get_target_cols())
                preds = model.predict(X_val_subset)

                df_preds = self._format_predictions(val_df, preds)
                metrics = self.eval_engine.compute_metrics(df_preds)

                # D. Calculate Deltas
                # Delta = New - Baseline.
                # If Delta < 0 (Negative), Error decreased -> Feature was harmful (Good to drop).
                # If Delta > 0 (Positive), Error increased -> Feature was important (Bad to drop).
                delta_cmae = metrics['cmae'] - baseline_metrics['cmae']

                result_entry = {
                    'feature': feature_to_drop,
                    'val_cmae': metrics['cmae'],
                    'delta_cmae': delta_cmae,
                    'val_crmse': metrics['crmse'],
                    'val_accuracy_5deg': metrics['accuracy_at_5deg']
                }
                results.append(result_entry)

                # E. Save Individual Result
                # (Optional: Save lightweight JSON per feature for resume capability)
                feature_dir = output_dir / feature_to_drop
                feature_dir.mkdir(exist_ok=True)
                with open(feature_dir / "metrics.json", 'w') as f:
                    json.dump(result_entry, f, indent=2)

                # ENHANCED Cleanup: Delete ALL large objects immediately
                # This is critical for memory management with 100+ features
                del model, train_subset, X_val_subset, preds, df_preds, metrics
                gc.collect()

                # Force more aggressive cleanup every 10 features
                if (i + 1) % 10 == 0:
                    self.logger.debug(f"Aggressive garbage collection after {i+1} features")
                    gc.collect()
                    gc.collect()  # Double collection to catch circular references

                pbar.update(1)  # Update progress bar

        # 3. Save Aggregate Report
        df_results = pd.DataFrame(results)
        # Rank: 1 = Best feature to drop (Lowest resulting CMAE)
        df_results['rank'] = df_results['val_cmae'].rank(ascending=True)
        df_results = df_results.sort_values('val_cmae')
        
        save_path = round_dir / constants.ROUND_FEATURE_EVALUATION_DIR / "feature_impact_all_features.parquet"
        save_dataframe(df_results, save_path, excel_copy=excel_copy, index=False)
        
        return results

    def _get_target_cols(self) -> List[str]:
        return [
            self.config['data']['target_sin'],
            self.config['data']['target_cos']
        ]

    def _compute_permutation_importance(
        self,
        model,
        X_val: pd.DataFrame,
        val_df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """
        Compute permutation importance on the validation set using circular MAE
        as the scoring metric. Reporting-only (does not affect decisions).
        """
        if X_val.empty:
            self.logger.warning("Permutation importance skipped: empty validation features.")
            return

        # Use true angle derived from sin/cos targets
        true_angle = reconstruct_angle(
            val_df[self.config['data']['target_sin']].values,
            val_df[self.config['data']['target_cos']].values,
        )

        def cmae_scorer(estimator, X, y_true):
            preds = estimator.predict(X)
            pred_angle = reconstruct_angle(preds[:, 0], preds[:, 1])
            errors = np.abs(wrap_angle(y_true - pred_angle))
            return -np.mean(errors)

        result = permutation_importance(
            model,
            X_val,
            true_angle,
            n_repeats=self.perm_repeats,
            scoring=cmae_scorer,
            random_state=self.config.get("execution", {}).get("seed"),
        )

        imp_df = pd.DataFrame({
            "feature": X_val.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "n_repeats": self.perm_repeats,
        }).sort_values("importance_mean", ascending=False)

        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_dataframe(imp_df, output_dir / "permutation_importance.parquet", excel_copy=excel_copy, index=False)

    def _compute_pdp(
        self,
        model,
        X_val: pd.DataFrame,
        val_df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """
        Compute 1D partial dependence of mean absolute error vs feature value.
        Reporting-only; does not affect feature selection decisions.
        """
        if X_val.empty:
            self.logger.warning("PDP skipped: empty validation features.")
            return

        features = self.pdp_features or list(X_val.columns)
        features = [f for f in features if f in X_val.columns]
        if not features:
            self.logger.warning("PDP skipped: no valid features specified.")
            return

        true_angle = reconstruct_angle(
            val_df[self.config['data']['target_sin']].values,
            val_df[self.config['data']['target_cos']].values,
        )

        records = []
        grid_points = max(2, self.pdp_grid_points)
        for feat in features:
            col = X_val[feat]
            if col.dtype.kind not in {'i', 'u', 'f'}:
                self.logger.debug(f"PDP skipping non-numeric feature {feat}")
                continue

            lower, upper = np.nanpercentile(col, [5, 95])
            grid = np.linspace(lower, upper, grid_points)

            for val in grid:
                X_mod = X_val.copy()
                X_mod[feat] = val
                preds = model.predict(X_mod)
                pred_angle = reconstruct_angle(preds[:, 0], preds[:, 1])
                errors = np.abs(wrap_angle(true_angle - pred_angle))
                records.append({
                    "feature": feat,
                    "grid_value": float(val),
                    "mean_abs_error": float(np.mean(errors)),
                    "n_samples": len(errors),
                })

        if not records:
            self.logger.warning("PDP produced no records.")
            return

        pdp_df = pd.DataFrame(records)
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_dataframe(pdp_df, output_dir / "partial_dependence.parquet", excel_copy=excel_copy, index=False)

    def _subset_df(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Creates a dataframe view with only selected features + targets.
        Includes targets so TrainingEngine can split X/y.

        Note: Returns a copy to avoid SettingWithCopyWarning, but this is
        necessary for data integrity. The copy is immediately used and then
        deleted, so memory impact is temporary.
        """
        targets = self._get_target_cols()
        # Ensure we don't duplicate targets if they happen to be in 'features' list (unlikely)
        cols = list(set(features + targets))
        # Use copy() here - it's necessary and the object is short-lived
        return df[cols].copy()

    def _format_predictions(self, val_df_original: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
        """
        Reconstructs the dataframe structure expected by EvaluationEngine.
        EvaluationEngine expects: true_angle, pred_angle, etc.
        """
        from utils.circular_metrics import reconstruct_angle, wrap_angle
        
        t_sin = self.config['data']['target_sin']
        t_cos = self.config['data']['target_cos']
        
        true_sin = val_df_original[t_sin].values
        true_cos = val_df_original[t_cos].values
        
        pred_sin = preds[:, 0]
        pred_cos = preds[:, 1]
        
        true_angle = reconstruct_angle(true_sin, true_cos)
        pred_angle = reconstruct_angle(pred_sin, pred_cos)
        error = wrap_angle(true_angle - pred_angle)
        
        return pd.DataFrame({
            'true_angle': true_angle,
            'pred_angle': pred_angle,
            'abs_error': np.abs(error),
            'error': error,
            'true_sin': true_sin, # Required for component metrics
            'true_cos': true_cos,
            'pred_sin': pred_sin,
            'pred_cos': pred_cos
        })
