import pandas as pd
import numpy as np
import logging
import json
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple

from modules.training_engine import TrainingEngine
from modules.evaluation_engine import EvaluationEngine

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
        
        self.logger.info(f"Baseline Validation CMAE: {baseline_metrics['cmae']:.4f}")

        # 2. LOFO Loop
        results = []
        output_dir = round_dir / "04_FEATURE_EVALUATION" / "minus_feature_models"
        output_dir.mkdir(parents=True, exist_ok=True)

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

            # Cleanup
            del model
            gc.collect()

        # 3. Save Aggregate Report
        df_results = pd.DataFrame(results)
        # Rank: 1 = Best feature to drop (Lowest resulting CMAE)
        df_results['rank'] = df_results['val_cmae'].rank(ascending=True)
        df_results = df_results.sort_values('val_cmae')
        
        save_path = round_dir / "04_FEATURE_EVALUATION" / "feature_impact_all_features.xlsx"
        df_results.to_excel(save_path, index=False)
        
        return results

    def _get_target_cols(self) -> List[str]:
        return [
            self.config['data']['target_sin'],
            self.config['data']['target_cos']
        ]

    def _subset_df(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Creates a dataframe view with only selected features + targets.
        Includes targets so TrainingEngine can split X/y.
        """
        targets = self._get_target_cols()
        # Ensure we don't duplicate targets if they happen to be in 'features' list (unlikely)
        cols = list(set(features + targets))
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