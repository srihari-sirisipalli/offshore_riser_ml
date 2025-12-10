import pandas as pd
import numpy as np
import logging
import shutil
import json
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from joblib import Parallel, delayed

from modules.hpo_search_engine import HPOSearchEngine
from modules.training_engine import TrainingEngine
from modules.evaluation_engine import EvaluationEngine
from modules.visualization.rfe_visualizer import RFEVisualizer
from modules.rfe.comparison_engine import ComparisonEngine
from modules.hyperparameter_analyzer.hyperparameter_analyzer import HyperparameterAnalyzer
from modules.error_analysis_engine.error_analysis_engine import ErrorAnalysisEngine
from modules.diagnostics_engine.diagnostics_engine import DiagnosticsEngine
from modules.global_error_tracking.evolution_tracker import EvolutionTracker

class RFEController:
    """
    Orchestrates the Recursive Feature Elimination (RFE) Pipeline.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.run_id = Path(config.get('outputs', {}).get('base_results_dir', 'results')).name
        self.base_dir = Path(config.get('outputs', {}).get('base_results_dir', 'results'))
        
        # Configuration Shortcuts
        self.rfe_config = config.get('iterative', {})
        self.min_features = self.rfe_config.get('min_features', 10)
        self.max_rounds = self.rfe_config.get('max_rounds', 50)
        self.stopping_criteria_config = self.rfe_config.get('stopping_criteria', 'combined')
        
        # Engine Initialization
        self.hpo_engine = HPOSearchEngine(self.config, self.logger)
        self.training_engine = TrainingEngine(self.config, self.logger)
        self.evaluation_engine = EvaluationEngine(self.config, self.logger)
        self.visualizer = RFEVisualizer(self.config, self.logger)
        self.comparison_engine = ComparisonEngine(self.config, self.logger)
        self.hpo_analyzer = HyperparameterAnalyzer(self.config, self.logger)
        self.error_analyzer = ErrorAnalysisEngine(self.config, self.logger)
        self.diagnostics_engine = DiagnosticsEngine(self.config, self.logger)
        self.evolution_tracker = EvolutionTracker(self.config, self.logger)
        
        # State Tracking
        self.current_round = 0
        self.active_features: List[str] = []
        self.best_params_history: List[Dict] = []
        self.rounds_history: List[Dict] = []

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Main execution loop for Circular RFE.
        """
        self.logger.info("Initializing Circular RFE Pipeline...")
        
        if not self.active_features:
             self.active_features = self._initialize_features(train_df)
        
        self.logger.info(f"Starting with {len(self.active_features)} features.")

        while True:
            round_dir = self.base_dir / f"ROUND_{self.current_round:03d}"

            if (round_dir / "_ROUND_COMPLETE.flag").exists():
                self.logger.info(f"Round {self.current_round} already complete. Skipping/Resuming...")

                # Load the flag to check stopping status
                with open(round_dir / "_ROUND_COMPLETE.flag", 'r') as f:
                    flag_data = json.load(f)

                stopping_status = flag_data.get('stopping_status', '')

                # If this round had a stopping condition, stop here
                if stopping_status and stopping_status != "CONTINUE":
                    self.logger.info(f"Previous run stopped at Round {self.current_round}: {stopping_status}")
                    self.logger.info("RFE Pipeline Completed Successfully (Resumed).")
                    return

                self._load_round_state(round_dir)
                self.current_round += 1
                continue

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"STARTING ROUND {self.current_round} | Features: {len(self.active_features)}")
            self.logger.info(f"{'='*60}")

            self._setup_round_directory(round_dir)
            self._save_round_datasets(round_dir, train_df, val_df, test_df)

            # Phase 1 & 2: HPO and Analysis
            best_config = self._execute_grid_search_phase(round_dir, train_df, val_df, test_df)
            self.best_params_history.append(best_config)

            # Phase 3: Baseline Training and Evaluation
            baseline_metrics, baseline_preds = self._train_baseline_phase(round_dir, best_config, train_df, val_df, test_df)

            # Phase 4: LOFO Feature Evaluation
            lofo_results = self._execute_lofo_phase(round_dir, best_config, train_df, val_df, test_df, baseline_metrics)
            
            self.visualizer.visualize_lofo_impact(round_dir, lofo_results)

            feature_to_drop, drop_reason = self._select_feature_to_drop(lofo_results)
            
            dropped_model_info = next((r for r in lofo_results if r['feature'] == feature_to_drop), None)

            comparison_summary = {}
            if dropped_model_info:
                self.logger.info("  >> Phase 5: Error Analysis on Dropped Feature Model...")
                # Run error analysis on the dropped feature model's predictions
                dropped_error_dir = round_dir / "05_DROPPED_FEATURE_RESULTS" / "error_analysis"
                filtered_test = self._filter_to_specific_features(test_df, [f for f in self.active_features if f != feature_to_drop])

                # We need to retrain on test to get test predictions for the dropped model
                dropped_model = self._train_model_internal(
                    self._filter_to_specific_features(train_df, [f for f in self.active_features if f != feature_to_drop]),
                    best_config,
                    f"dropped_{feature_to_drop}"
                )
                dropped_test_preds = self._make_predictions(dropped_model, filtered_test, "test")

                self.error_analyzer.analyze(
                    predictions=dropped_test_preds,
                    features=filtered_test,
                    split_name="test",
                    output_dir=dropped_error_dir
                )

                # Save dropped model predictions
                dropped_preds_dir = round_dir / "05_DROPPED_FEATURE_RESULTS"
                dropped_preds_dir.mkdir(parents=True, exist_ok=True)
                dropped_model_info['val_predictions'].to_excel(dropped_preds_dir / "dropped_predictions_val.xlsx", index=False)
                dropped_test_preds.to_excel(dropped_preds_dir / "dropped_predictions_test.xlsx", index=False)

                # Generate Full Diagnostic Plots for dropped model
                self.logger.info("  Generating diagnostic plots for dropped feature model...")
                dropped_diag_config = self.config.copy()
                dropped_diag_config['outputs'] = self.config['outputs'].copy()
                dropped_diag_config['outputs']['base_results_dir'] = str(dropped_preds_dir / "diagnostics")

                temp_dropped_diag = DiagnosticsEngine(dropped_diag_config, self.logger)
                temp_dropped_diag.generate_all(dropped_model_info['val_predictions'], "val_dropped", "dropped")
                temp_dropped_diag.generate_all(dropped_test_preds, "test_dropped", "dropped")

                self.logger.info("  >> Phase 6: Comparing Baseline vs. Dropped Model...")
                comparison_summary = self.comparison_engine.compare(
                    round_dir,
                    baseline_metrics,
                    dropped_model_info['metrics'],
                    feature_to_drop
                )
                self.visualizer.visualize_comparison(
                    round_dir,
                    baseline_preds,
                    dropped_model_info['val_predictions'],
                    feature_to_drop,
                    baseline_metrics,
                    dropped_model_info['metrics']
                )
            else:
                self.logger.warning(f"Could not find results for dropped feature '{feature_to_drop}'. Skipping comparison.")

            round_summary = {
                'round': self.current_round,
                'n_features': len(self.active_features),
                'dropped_feature': feature_to_drop,
                'metrics': baseline_metrics,
                'hyperparameters': best_config,
                'comparison': comparison_summary
            }
            self.rounds_history.append(round_summary)

            # Phase 7: Update Global Evolution Tracker
            self.logger.info("  >> Phase 7: Updating Global Evolution Tracker...")
            self.evolution_tracker.update_tracker(round_summary)

            should_stop, stop_reason = self._check_stopping_criteria(round_summary)

            self._finalize_round(round_dir, feature_to_drop, stop_reason)

            if should_stop:
                self.logger.info(f"Stopping Criteria Met: {stop_reason}")
                break

            if feature_to_drop in self.active_features:
                self.active_features.remove(feature_to_drop)
            self.current_round += 1
            
            gc.collect()

        self.logger.info("RFE Pipeline Completed Successfully.")

    def _initialize_features(self, df: pd.DataFrame) -> List[str]:
        targets = [
            self.config['data']['target_sin'],
            self.config['data']['target_cos']
        ]
        meta = ['angle_deg', 'angle_bin', 'hs_bin', 'combined_bin', 'row_index']
        exclude = set(targets + meta + self.config['data'].get('drop_columns', []))

        # Conditionally exclude Hs column based on config
        if not self.config['data'].get('include_hs_in_features', False):
            hs_col = self.config['data']['hs_column']
            exclude.add(hs_col)

        return [c for c in df.columns if c not in exclude]

    def _setup_round_directory(self, round_dir: Path):
        subdirs = [
            "00_DATASETS", "01_GRID_SEARCH", "02_HYPERPARAMETER_ANALYSIS",
            "03_BASE_MODEL_RESULTS", "04_FEATURE_EVALUATION", 
            "05_DROPPED_FEATURE_RESULTS", "06_COMPARISON"
        ]
        for d in subdirs:
            (round_dir / d).mkdir(parents=True, exist_ok=True)

    def _save_round_datasets(self, round_dir: Path, train, val, test):
        features_path = round_dir / "00_DATASETS" / "feature_list.json"
        with open(features_path, 'w') as f:
            json.dump(self.active_features, f, indent=2)

    def _execute_grid_search_phase(self, round_dir: Path, train_df, val_df, test_df) -> Dict:
        """
        Execute full HPO search and analysis for the current round.
        """
        self.logger.info("  >> Phase 1: Hyperparameter Optimization")

        # 1. Create round-specific config
        round_config = self.config.copy()
        round_config['outputs'] = self.config['outputs'].copy()
        round_config['outputs']['base_results_dir'] = str(round_dir / "01_GRID_SEARCH")

        # 2. Filter datasets to only include active features (plus targets and metadata)
        filtered_train = self._filter_to_active_features(train_df)
        filtered_val = self._filter_to_active_features(val_df)
        filtered_test = self._filter_to_active_features(test_df)

        # 3. Run HPO Search
        round_hpo_engine = HPOSearchEngine(round_config, self.logger)
        best_config = round_hpo_engine.execute(filtered_train, filtered_val, filtered_test, f"round_{self.current_round:03d}")

        # 4. Run Hyperparameter Analysis (optional)
        run_hpo_analysis = self.rfe_config.get('run_hpo_analysis', True)

        if run_hpo_analysis:
            self.logger.info("  >> Phase 2: Hyperparameter Analysis")
            hpo_results_file = round_dir / "01_GRID_SEARCH" / "03_HYPERPARAMETER_OPTIMIZATION" / "results" / "all_configurations.xlsx"
            analysis_output_dir = round_dir / "02_HYPERPARAMETER_ANALYSIS"

            if hpo_results_file.exists():
                self.hpo_analyzer.analyze(hpo_results_file, analysis_output_dir)
            else:
                self.logger.warning(f"HPO results file not found: {hpo_results_file}")
        else:
            self.logger.info("  >> Hyperparameter Analysis skipped (disabled in config)")

        return best_config

    def _train_baseline_phase(self, round_dir, model_config, train, val, test) -> Tuple[Dict, pd.DataFrame]:
        """
        Train baseline model with all current active features and evaluate.
        """
        self.logger.info("  >> Phase 3: Baseline Model Training & Evaluation")

        # 1. Filter datasets to only include active features
        filtered_train = self._filter_to_active_features(train)
        filtered_val = self._filter_to_active_features(val)
        filtered_test = self._filter_to_active_features(test)

        # 2. Train model
        baseline_model = self._train_model_internal(filtered_train, model_config, "baseline")

        # 3. Make predictions on val and test
        val_predictions = self._make_predictions(baseline_model, filtered_val, "val")
        test_predictions = self._make_predictions(baseline_model, filtered_test, "test")

        # 4. Evaluate
        val_metrics = self.evaluation_engine.compute_metrics(val_predictions)
        test_metrics = self.evaluation_engine.compute_metrics(test_predictions)

        # 5. Save predictions and metrics
        output_dir = round_dir / "03_BASE_MODEL_RESULTS"
        output_dir.mkdir(parents=True, exist_ok=True)

        val_predictions.to_excel(output_dir / "baseline_predictions_val.xlsx", index=False)
        test_predictions.to_excel(output_dir / "baseline_predictions_test.xlsx", index=False)
        pd.DataFrame([val_metrics]).to_excel(output_dir / "baseline_metrics_val.xlsx", index=False)
        pd.DataFrame([test_metrics]).to_excel(output_dir / "baseline_metrics_test.xlsx", index=False)

        # 6. Run Error Analysis on test set
        error_analysis_dir = output_dir / "error_analysis_test"
        self.error_analyzer.analyze(
            predictions=test_predictions,
            features=filtered_test,
            split_name="test",
            output_dir=error_analysis_dir
        )

        # 7. Generate Full Diagnostic Plots for baseline model
        self.logger.info("  Generating diagnostic plots for baseline model...")
        diag_config_backup = self.config.copy()
        diag_config_backup['outputs'] = self.config['outputs'].copy()
        diag_config_backup['outputs']['base_results_dir'] = str(output_dir / "diagnostics")

        # Create a temporary DiagnosticsEngine with custom output dir
        temp_diag_engine = DiagnosticsEngine(diag_config_backup, self.logger)
        temp_diag_engine.generate_all(val_predictions, "val_baseline", "baseline")
        temp_diag_engine.generate_all(test_predictions, "test_baseline", "baseline")

        # Combine metrics for return
        combined_metrics = {**{f'val_{k}': v for k, v in val_metrics.items()},
                           **{f'test_{k}': v for k, v in test_metrics.items()}}

        self.logger.info(f"  Baseline Model - Val CMAE: {val_metrics['cmae']:.4f}°, Test CMAE: {test_metrics['cmae']:.4f}°")

        return combined_metrics, val_predictions

    def _execute_lofo_phase(self, round_dir, model_config, train, val, test, baseline_metrics) -> List[Dict]:
        """
        Leave-One-Feature-Out evaluation - train and evaluate model with each feature removed.
        Uses parallelization for speed. Evaluates on BOTH val and test sets.
        """
        self.logger.info("  >> Phase 4: LOFO Feature Evaluation (Parallelized)")

        output_dir = round_dir / "04_FEATURE_EVALUATION"
        output_dir.mkdir(parents=True, exist_ok=True)

        baseline_val_cmae = baseline_metrics.get('val_cmae', baseline_metrics.get('cmae', 0))
        baseline_test_cmae = baseline_metrics.get('test_cmae', 0)

        # Define the evaluation function for a single feature
        def evaluate_single_feature(i, feature_to_drop):
            # 1. Create temporary feature list without this feature
            temp_features = [f for f in self.active_features if f != feature_to_drop]

            # 2. Filter datasets
            filtered_train = self._filter_to_specific_features(train, temp_features)
            filtered_val = self._filter_to_specific_features(val, temp_features)
            filtered_test = self._filter_to_specific_features(test, temp_features)

            # 3. Train model
            lofo_model = self._train_model_internal(filtered_train, model_config, f"lofo_{feature_to_drop}")

            # 4. Make predictions on BOTH val and test
            val_predictions = self._make_predictions(lofo_model, filtered_val, "val")
            test_predictions = self._make_predictions(lofo_model, filtered_test, "test")

            # 5. Evaluate on both sets
            val_metrics = self.evaluation_engine.compute_metrics(val_predictions)
            test_metrics = self.evaluation_engine.compute_metrics(test_predictions)

            # 6. Calculate deltas
            delta_val_cmae = val_metrics['cmae'] - baseline_val_cmae
            delta_test_cmae = test_metrics['cmae'] - baseline_test_cmae

            # 7. Return comprehensive results
            return {
                "feature": feature_to_drop,
                "val_predictions": val_predictions,
                "test_predictions": test_predictions,
                # Validation metrics
                "val_cmae": val_metrics['cmae'],
                "val_crmse": val_metrics.get('crmse', 0),
                "val_max_error": val_metrics.get('max_error', 0),
                "val_accuracy_at_0deg": val_metrics.get('accuracy_at_0deg', 0),
                "val_accuracy_at_5deg": val_metrics.get('accuracy_at_5deg', 0),
                "val_accuracy_at_10deg": val_metrics.get('accuracy_at_10deg', 0),
                "delta_val_cmae": delta_val_cmae,
                "delta_cmae": delta_val_cmae,  # Alias for visualizer compatibility
                # Test metrics
                "test_cmae": test_metrics['cmae'],
                "test_crmse": test_metrics.get('crmse', 0),
                "test_max_error": test_metrics.get('max_error', 0),
                "test_accuracy_at_0deg": test_metrics.get('accuracy_at_0deg', 0),
                "test_accuracy_at_5deg": test_metrics.get('accuracy_at_5deg', 0),
                "test_accuracy_at_10deg": test_metrics.get('accuracy_at_10deg', 0),
                "delta_test_cmae": delta_test_cmae,
                # Store full metrics for later use
                "metrics": {**{f'val_{k}': v for k, v in val_metrics.items()},
                           **{f'test_{k}': v for k, v in test_metrics.items()}}
            }

        # Get number of parallel jobs from config
        n_jobs = self.config['execution'].get('n_jobs', -1)
        self.logger.info(f"     Running LOFO on {len(self.active_features)} features with n_jobs={n_jobs}...")

        # Run LOFO evaluations in parallel
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(evaluate_single_feature)(i, feature)
            for i, feature in enumerate(self.active_features)
        )

        # Save comprehensive LOFO results summary
        lofo_summary = pd.DataFrame([{
            'feature': r['feature'],
            # Validation metrics
            'val_cmae': r['val_cmae'],
            'val_crmse': r['val_crmse'],
            'val_max_error': r['val_max_error'],
            'val_accuracy_at_0deg': r['val_accuracy_at_0deg'],
            'val_accuracy_at_5deg': r['val_accuracy_at_5deg'],
            'val_accuracy_at_10deg': r['val_accuracy_at_10deg'],
            'delta_val_cmae': r['delta_val_cmae'],
            # Test metrics
            'test_cmae': r['test_cmae'],
            'test_crmse': r['test_crmse'],
            'test_max_error': r['test_max_error'],
            'test_accuracy_at_0deg': r['test_accuracy_at_0deg'],
            'test_accuracy_at_5deg': r['test_accuracy_at_5deg'],
            'test_accuracy_at_10deg': r['test_accuracy_at_10deg'],
            'delta_test_cmae': r['delta_test_cmae']
        } for r in results])

        lofo_summary.to_excel(output_dir / "lofo_summary.xlsx", index=False)
        self.logger.info(f"  LOFO Evaluation complete. Evaluated {len(results)} features on val and test sets.")

        return results

    def _select_feature_to_drop(self, lofo_results: List[Dict]) -> Tuple[str, str]:
        if not lofo_results:
             return self.active_features[-1], "Only one feature remaining"
        sorted_res = sorted(lofo_results, key=lambda x: x['val_cmae'])
        best_drop = sorted_res[0]
        self.logger.info(f"  Best feature to drop: '{best_drop['feature']}' (CMAE: {best_drop['val_cmae']:.4f})")
        return best_drop['feature'], "Lowest Validation CMAE"

    def _check_stopping_criteria(self, round_summary: Dict) -> Tuple[bool, str]:
        if (round_summary['n_features'] - 1) <= self.min_features:
            return True, "Minimum Feature Count Reached"
        # Check if we've completed max_rounds (current_round + 1 since we start at 0)
        if (self.current_round + 1) >= self.max_rounds:
            return True, "Maximum Rounds Reached"
        if len(self.rounds_history) > 1:
            # Use validation CMAE for stopping criteria
            prev_metrics = self.rounds_history[-2]['metrics']
            curr_metrics = round_summary['metrics']

            # Handle both direct 'cmae' and prefixed 'val_cmae' keys
            prev_cmae = prev_metrics.get('val_cmae', prev_metrics.get('cmae', 0))
            curr_cmae = curr_metrics.get('val_cmae', curr_metrics.get('cmae', 0))

            if curr_cmae > prev_cmae * 1.1:
                return True, "Performance Degradation > 10%"
        return False, ""

    def _finalize_round(self, round_dir: Path, dropped_feature: str, stop_reason: str):
        flag_data = {
            "timestamp": datetime.now().isoformat(),
            "round": self.current_round,
            "dropped_feature": dropped_feature,
            "features_remaining": len(self.active_features) - 1,
            "stopping_status": stop_reason if stop_reason else "CONTINUE"
        }
        with open(round_dir / "_ROUND_COMPLETE.flag", 'w') as f:
            json.dump(flag_data, f, indent=2)
        self.logger.info(f"  >> Round {self.current_round} Complete. Dropped: {dropped_feature}")

    def _load_round_state(self, round_dir: Path):
        try:
            with open(round_dir / "_ROUND_COMPLETE.flag", 'r') as f:
                flag = json.load(f)
            with open(round_dir / "00_DATASETS" / "feature_list.json", 'r') as f:
                features_at_start = json.load(f)
            dropped = flag.get('dropped_feature')
            self.active_features = [f for f in features_at_start if f != dropped]
            self.logger.info(f"  Loaded state from Round {flag['round']}. Next features: {len(self.active_features)}")
        except Exception as e:
            self.logger.error(f"Failed to load resume state from {round_dir}: {e}")
            raise RuntimeError("Resume failed. Corrupted state.")

    # --- Helper Methods ---

    def _filter_to_active_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe to include only active features plus targets and metadata columns.
        """
        targets = [
            self.config['data']['target_sin'],
            self.config['data']['target_cos']
        ]
        metadata = ['angle_deg', 'angle_bin', 'hs_bin', 'combined_bin', 'row_index']

        # Keep: active features + targets + metadata columns that exist in df
        keep_cols = self.active_features + targets + [m for m in metadata if m in df.columns]

        # Also keep Hs if it's configured to be included
        if self.config['data'].get('include_hs_in_features', False):
            hs_col = self.config['data']['hs_column']
            if hs_col not in keep_cols and hs_col in df.columns:
                keep_cols.append(hs_col)

        return df[[c for c in keep_cols if c in df.columns]].copy()

    def _filter_to_specific_features(self, df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        """
        Filter dataframe to include only specified features plus targets and metadata columns.
        """
        targets = [
            self.config['data']['target_sin'],
            self.config['data']['target_cos']
        ]
        metadata = ['angle_deg', 'angle_bin', 'hs_bin', 'combined_bin', 'row_index']

        # Keep: specified features + targets + metadata columns that exist in df
        keep_cols = feature_list + targets + [m for m in metadata if m in df.columns]

        # Also keep Hs if it's configured to be included
        if self.config['data'].get('include_hs_in_features', False):
            hs_col = self.config['data']['hs_column']
            if hs_col not in keep_cols and hs_col in df.columns:
                keep_cols.append(hs_col)

        return df[[c for c in keep_cols if c in df.columns]].copy()

    def _train_model_internal(self, train_df: pd.DataFrame, model_config: Dict, model_name: str):
        """
        Internal method to train a model on the given training data.
        """
        from modules.model_factory import ModelFactory

        # Prepare data
        drop_cols = self.config['data']['drop_columns'] + [
            self.config['data']['target_sin'],
            self.config['data']['target_cos'],
            'angle_deg', 'angle_bin', 'hs_bin', 'combined_bin'
        ]

        X = train_df.drop(columns=drop_cols, errors='ignore')
        y = train_df[[self.config['data']['target_sin'], self.config['data']['target_cos']]]

        # Create and train model
        model = ModelFactory.create(model_config.get('model'), model_config.get('params', {}))
        model.fit(X, y)

        return model

    def _make_predictions(self, model, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """
        Make predictions using the trained model and format as expected by evaluation engine.
        """
        from utils.circular_metrics import reconstruct_angle

        # Prepare data
        drop_cols = self.config['data']['drop_columns'] + [
            self.config['data']['target_sin'],
            self.config['data']['target_cos'],
            'angle_deg', 'angle_bin', 'hs_bin', 'combined_bin'
        ]

        X = df.drop(columns=drop_cols, errors='ignore')
        y_true = df[[self.config['data']['target_sin'], self.config['data']['target_cos']]].values

        # Predict
        y_pred = model.predict(X)

        # Reconstruct angles
        true_angle = reconstruct_angle(y_true[:, 0], y_true[:, 1])
        pred_angle = reconstruct_angle(y_pred[:, 0], y_pred[:, 1])

        # Calculate circular error
        raw_diff = np.abs(true_angle - pred_angle)
        abs_error = np.where(raw_diff > 180, 360 - raw_diff, raw_diff)
        error = pred_angle - true_angle

        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'row_index': df.index if 'row_index' not in df.columns else df['row_index'].values,
            'true_sin': y_true[:, 0],
            'true_cos': y_true[:, 1],
            'pred_sin': y_pred[:, 0],
            'pred_cos': y_pred[:, 1],
            'true_angle': true_angle,
            'pred_angle': pred_angle,
            'error': error,
            'abs_error': abs_error
        })

        return predictions_df