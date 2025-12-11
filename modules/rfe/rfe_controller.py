import pandas as pd
import numpy as np
import logging
import shutil
import json
import gc
import hashlib
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
from modules.error_analysis_engine.safety_analysis import safety_threshold_summary
from modules.diagnostics_engine.diagnostics_engine import DiagnosticsEngine
from modules.global_error_tracking.evolution_tracker import EvolutionTracker
from modules.data_integrity import DataIntegrityTracker
from utils.file_io import save_dataframe
from utils.results_layout import ResultsLayoutManager
from utils import constants

class RFEController:
    """
    Orchestrates the Recursive Feature Elimination (RFE) Pipeline.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.run_id = Path(config.get('outputs', {}).get('base_results_dir', 'results')).name
        self.base_dir = Path(config.get('outputs', {}).get('base_results_dir', 'results'))
        self.excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        self.results_layout = ResultsLayoutManager(self.base_dir, excel_copy=self.excel_copy, logger=self.logger)
        self.results_layout.ensure_base_structure()
        
        # Configuration Shortcuts
        self.rfe_config = config.get('iterative', {})
        self.min_features = self.rfe_config.get('min_features', 10)
        self.max_rounds = self.rfe_config.get('max_rounds', 50)
        self.stopping_criteria_config = self.rfe_config.get('stopping_criteria', 'combined')
        
        # Engine Initialization
        eval_cfg = self.config.copy()
        eval_cfg['outputs'] = self.config.get('outputs', {}).copy()
        # Compute-only; avoid creating a top-level PerformanceMetrics folder
        eval_cfg['outputs']['skip_dir_creation'] = True
        self.evaluation_engine = EvaluationEngine(eval_cfg, self.logger)
        self.visualizer = RFEVisualizer(self.config, self.logger)
        self.comparison_engine = ComparisonEngine(self.config, self.logger)
        hpo_an_cfg = self.config.copy()
        hpo_an_cfg['outputs'] = self.config.get('outputs', {}).copy()
        hpo_an_cfg['outputs']['skip_dir_creation'] = True
        self.hpo_analyzer = HyperparameterAnalyzer(hpo_an_cfg, self.logger)
        self.evolution_tracker = EvolutionTracker(self.config, self.logger)
        self.data_integrity_tracker = DataIntegrityTracker(self.config, self.logger)
        vis_cfg = self.config.get("visualization", {})
        self.run_advanced_suite = vis_cfg.get("run_advanced_suite", False)
        self.run_dashboard = vis_cfg.get("run_dashboard", False)
        
        # State Tracking
        self.current_round = 0
        self.active_features: List[str] = []
        self.best_params_history: List[Dict] = []
        self.rounds_history: List[Dict] = []
        self.prev_val_errors: Optional[np.ndarray] = None
        self.prev_test_errors: Optional[np.ndarray] = None
        self.significance_history: List[Dict[str, Any]] = []
        self.safety_history: List[Dict[str, Any]] = []
        self.cv_consistency_history: List[Dict[str, Any]] = []

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
            self.data_integrity_tracker.track_round(self.current_round, train_df, val_df, test_df)

            # Phase 1 & 2: HPO and Analysis
            best_config = self._execute_grid_search_phase(round_dir, train_df, val_df, test_df)
            self.best_params_history.append(best_config)

            # Phase 3: Baseline Training and Evaluation
            baseline_metrics, baseline_val_preds, baseline_test_preds = self._train_baseline_phase(round_dir, best_config, train_df, val_df, test_df)
            self._run_significance_analysis(round_dir, baseline_val_preds, baseline_test_preds)
            self._summarize_safety(round_dir, baseline_val_preds, baseline_test_preds)
            # CV consistency / overfitting gap persistence (train vs val metrics if available)
            train_scores = {k.replace('train_', ''): v for k, v in baseline_metrics.items() if k.startswith('train_')}
            val_scores_clean = {k.replace('val_', ''): v for k, v in baseline_metrics.items() if k.startswith('val_')}
            cv_scores = {
                "cmae": np.array([baseline_metrics.get('val_cmae', np.nan)]),
                "crmse": np.array([baseline_metrics.get('val_crmse', np.nan)]),
            }
            self.evaluation_engine.summarize_cv_and_overfitting(
                cv_scores=cv_scores,
                train_scores=train_scores,
                val_scores=val_scores_clean,
                output_dir=round_dir / constants.ROUND_BASE_MODEL_RESULTS_DIR,
                excel_copy=self.config.get("outputs", {}).get("save_excel_copy", False),
            )

            # Phase 4: LOFO Feature Evaluation
            lofo_results = self._execute_lofo_phase(round_dir, best_config, train_df, val_df, test_df, baseline_metrics)
            
            self.visualizer.visualize_lofo_impact(round_dir, lofo_results)

            feature_to_drop, drop_reason = self._select_feature_to_drop(lofo_results)
            
            dropped_model_info = next((r for r in lofo_results if r['feature'] == feature_to_drop), None)

            comparison_summary = {}
            if dropped_model_info:
                self.logger.info("  >> Phase 5: Error Analysis on Dropped Feature Model...")
                # Run error analysis on the dropped feature model's predictions
                dropped_error_dir = round_dir / constants.ROUND_DROPPED_FEATURE_RESULTS_DIR / "error_analysis"
                filtered_test = self._filter_to_specific_features(test_df, [f for f in self.active_features if f != feature_to_drop])

                # We need to retrain on test to get test predictions for the dropped model
                dropped_model = self._train_model_internal(
                    self._filter_to_specific_features(train_df, [f for f in self.active_features if f != feature_to_drop]),
                    best_config,
                    f"dropped_{feature_to_drop}"
                )
                dropped_test_preds = self._make_predictions(dropped_model, filtered_test, "test")

                temp_ea_config = self.config.copy()
                temp_ea_config['outputs'] = self.config['outputs'].copy()
                temp_ea_config['outputs']['base_results_dir'] = str(round_dir / constants.ROUND_DROPPED_FEATURE_RESULTS_DIR)
                temp_error_analyzer = ErrorAnalysisEngine(temp_ea_config, self.logger)
                temp_error_analyzer.execute(
                    predictions=dropped_test_preds,
                    features=filtered_test,
                    split_name="test"
                )

                # Save dropped model predictions
                dropped_preds_dir = round_dir / constants.ROUND_DROPPED_FEATURE_RESULTS_DIR
                dropped_preds_dir.mkdir(parents=True, exist_ok=True)
                excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
                save_dataframe(dropped_model_info['val_predictions'], dropped_preds_dir / "dropped_predictions_val.parquet", excel_copy=excel_copy, index=False)
                save_dataframe(dropped_test_preds, dropped_preds_dir / "dropped_predictions_test.parquet", excel_copy=excel_copy, index=False)

                # Generate Full Diagnostic Plots for dropped model
                self.logger.info("  Generating diagnostic plots for dropped feature model...")
                dropped_diag_config = self.config.copy()
                dropped_diag_config['outputs'] = self.config['outputs'].copy()
                # Place diagnostics directly under the dropped model directory
                dropped_diag_config['outputs']['base_results_dir'] = str(dropped_preds_dir)

                temp_dropped_diag = DiagnosticsEngine(dropped_diag_config, self.logger)
                temp_dropped_diag.execute(dropped_model_info['val_predictions'], "val_dropped", "dropped")
                temp_dropped_diag.execute(dropped_test_preds, "test_dropped", "dropped")

                self.logger.info("  >> Phase 6: Comparing Baseline vs. Dropped Model...")
                comparison_summary = self.comparison_engine.compare(
                    round_dir,
                    baseline_metrics,
                    dropped_model_info['metrics'],
                    feature_to_drop
                )
                self._compare_baseline_vs_dropped(round_dir, baseline_val_preds, baseline_test_preds, dropped_model_info)
                self.visualizer.visualize_comparison(
                    round_dir,
                    baseline_val_preds,
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

        self.data_integrity_tracker.finalize()
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
        structure_flag = round_dir / "_ROUND_STRUCTURE_READY.flag"
        temp_dir = round_dir.with_name(round_dir.name + ".tmp_build")

        # Clean up any stale temp directory from a previous interrupted build
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

        # If structure already marked ready, just ensure subdirs exist and return
        if structure_flag.exists():
            self.results_layout.ensure_round_structure(round_dir)
            return

        # If round_dir exists but no flag (partial build), complete it and mark ready
        if round_dir.exists():
            created = self.results_layout.ensure_round_structure(round_dir)
            self._write_structure_flag(structure_flag, created)
            return

        # Fresh build: create in temp, then atomic rename to final
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            created = self.results_layout.ensure_round_structure(temp_dir)
            self._write_structure_flag(temp_dir / "_ROUND_STRUCTURE_READY.flag", created)
            temp_dir.replace(round_dir)
        except Exception as e:
            # Cleanup temp on failure
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to set up round directory atomically: {e}")

    def _write_structure_flag(self, flag_path: Path, subdirs: List[str]) -> None:
        """Write a small marker file indicating directory structure is complete."""
        temp_flag = flag_path.with_suffix(".tmp")
        payload = {
            "timestamp": datetime.now().isoformat(),
            "subdirectories": subdirs
        }
        with open(temp_flag, "w") as f:
            json.dump(payload, f, indent=2)
        temp_flag.replace(flag_path)

    def _save_round_datasets(self, round_dir: Path, train, val, test):
        """
        Save feature list with checksum for validation and atomicity.
        """
        datasets_dir = round_dir / constants.ROUND_DATASETS_DIR
        datasets_dir.mkdir(parents=True, exist_ok=True)
        features_path = datasets_dir / "feature_list.json"
        checksum_path = datasets_dir / "feature_list_checksum.txt"

        # Create feature list data with metadata
        feature_data = {
            "features": self.active_features,
            "count": len(self.active_features),
            "round": self.current_round,
            "timestamp": datetime.now().isoformat()
        }

        # Calculate checksum
        checksum = self._calculate_feature_checksum(self.active_features)

        # Atomic write: write to temp file first, then rename
        temp_features_path = features_path.with_suffix('.tmp')
        temp_checksum_path = checksum_path.with_suffix('.tmp')

        try:
            datasets_dir.mkdir(parents=True, exist_ok=True)
            # Write feature list
            with open(temp_features_path, 'w') as f:
                json.dump(feature_data, f, indent=2)

            # Write checksum
            with open(temp_checksum_path, 'w') as f:
                f.write(checksum)

            # Atomic rename (on most filesystems this is atomic)
            temp_features_path.replace(features_path)
            temp_checksum_path.replace(checksum_path)

            self.logger.info(f"  Feature list saved with checksum: {checksum[:8]}...")

        except Exception as e:
            # Cleanup temp files if write failed
            if temp_features_path.exists():
                temp_features_path.unlink()
            if temp_checksum_path.exists():
                temp_checksum_path.unlink()
            raise RuntimeError(f"Failed to save feature list atomically: {e}")

    def _execute_grid_search_phase(self, round_dir: Path, train_df, val_df, test_df) -> Dict:
        """
        Execute full HPO search and analysis for the current round.
        """
        self.logger.info("  >> Phase 1: Hyperparameter Optimization")

        # 1. Create round-specific config
        round_config = self.config.copy()
        round_config['outputs'] = self.config['outputs'].copy()
        round_config['outputs']['base_results_dir'] = str(round_dir / constants.ROUND_GRID_SEARCH_DIR)

        # 2. Filter datasets to only include active features (plus targets and metadata)
        filtered_train = self._filter_to_active_features(train_df)
        filtered_val = self._filter_to_active_features(val_df)
        filtered_test = self._filter_to_active_features(test_df)

        # 3. Run HPO Search
        round_hpo_engine = HPOSearchEngine(round_config, self.logger)
        best_config = round_hpo_engine.execute(filtered_train, filtered_val, filtered_test, f"round_{self.current_round:03d}")

        # Record CV consistency for best config if available
        self._record_cv_consistency(round_dir, best_config)

        # 4. Run Hyperparameter Analysis (optional)
        run_hpo_analysis = self.rfe_config.get('run_hpo_analysis', True)

        if run_hpo_analysis:
            self.logger.info("  >> Phase 2: Hyperparameter Analysis")
            hpo_results_file = (
                round_dir
                / constants.ROUND_GRID_SEARCH_DIR
                / constants.HPO_OPTIMIZATION_DIR
                / "results"
                / "all_configurations.parquet"
            )
            analysis_output_dir = round_dir / constants.ROUND_HPO_ANALYSIS_DIR

            if hpo_results_file.exists():
                # Use a fresh analyzer scoped to this round to avoid top-level artifacts
                round_an_cfg = self.config.copy()
                round_an_cfg['outputs'] = self.config.get('outputs', {}).copy()
                round_an_cfg['outputs']['base_results_dir'] = str(analysis_output_dir)
                round_an_cfg['outputs']['skip_dir_creation'] = False
                round_hpo_analyzer = HyperparameterAnalyzer(round_an_cfg, self.logger)
                round_hpo_analyzer.execute(hpo_results_file)
            else:
                self.logger.warning(f"HPO results file not found: {hpo_results_file}")
        else:
            self.logger.info("  >> Hyperparameter Analysis skipped (disabled in config)")

        return best_config

    def _train_baseline_phase(self, round_dir, model_config, train, val, test) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
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
        output_dir = round_dir / constants.ROUND_BASE_MODEL_RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        val_pred_path = output_dir / "baseline_predictions_val.parquet"
        test_pred_path = output_dir / "baseline_predictions_test.parquet"
        val_metrics_path = output_dir / "baseline_metrics_val.parquet"
        test_metrics_path = output_dir / "baseline_metrics_test.parquet"

        save_dataframe(val_predictions, val_pred_path, excel_copy=excel_copy, index=False)
        save_dataframe(test_predictions, test_pred_path, excel_copy=excel_copy, index=False)
        save_dataframe(pd.DataFrame([val_metrics]), val_metrics_path, excel_copy=excel_copy, index=False)
        save_dataframe(pd.DataFrame([test_metrics]), test_metrics_path, excel_copy=excel_copy, index=False)

        # 6. Run Error Analysis on test set
        temp_ea_config = self.config.copy()
        temp_ea_config['outputs'] = self.config['outputs'].copy()
        temp_ea_config['outputs']['base_results_dir'] = str(output_dir)
        temp_error_analyzer = ErrorAnalysisEngine(temp_ea_config, self.logger)
        temp_error_analyzer.execute(
            predictions=test_predictions,
            features=filtered_test,
            split_name="test"
        )

        # 7. Generate Full Diagnostic Plots for baseline model
        self.logger.info("  Generating diagnostic plots for baseline model...")
        diag_config_backup = self.config.copy()
        diag_config_backup['outputs'] = self.config['outputs'].copy()
        diag_config_backup['outputs']['base_results_dir'] = str(output_dir)

        # Create a temporary DiagnosticsEngine with custom output dir
        temp_diag_engine = DiagnosticsEngine(diag_config_backup, self.logger)
        temp_diag_engine.execute(val_predictions, "val_baseline", "baseline")
        temp_diag_engine.execute(test_predictions, "test_baseline", "baseline")

        # 8. Advanced visuals (optional)
        if self.run_advanced_suite:
            try:
                from modules.visualization.advanced_viz import AdvancedVisualizer
                adv_viz = AdvancedVisualizer(self.config, self.logger)
                adv_output = output_dir / constants.ROUND_ADVANCED_VISUALIZATIONS_DIR
                adv_viz.run_default_suite(test_predictions, adv_output, split_name="test_baseline")
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"Advanced visualizations skipped: {exc}")

        # 9. Interactive dashboard (optional)
        if self.run_dashboard:
            try:
                from modules.visualization.interactive_dashboard import build_dashboard
                dash_path = output_dir / constants.ROUND_ADVANCED_VISUALIZATIONS_DIR / "interactive_dashboard_test_baseline.html"
                build_dashboard(test_predictions, dash_path, hs_col=self.config.get("data", {}).get("hs_column", "Hs_ft"))
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"Dashboard generation skipped: {exc}")

        # Mirror into standardized results layout (Complete Results Directory Organization)
        try:
            self.results_layout.mirror_baseline_outputs(round_dir)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"Results layout mirroring skipped: {exc}")

        # Combine metrics for return
        combined_metrics = {**{f'val_{k}': v for k, v in val_metrics.items()},
                           **{f'test_{k}': v for k, v in test_metrics.items()}}

        self.logger.info(f"  Baseline Model - Val CMAE: {val_metrics['cmae']:.4f}°, Test CMAE: {test_metrics['cmae']:.4f}°")

        return combined_metrics, val_predictions, test_predictions

    def _execute_lofo_phase(self, round_dir, model_config, train, val, test, baseline_metrics) -> List[Dict]:
        """
        Leave-One-Feature-Out evaluation - train and evaluate model with each feature removed.
        Uses parallelization for speed. Evaluates on BOTH val and test sets.
        """
        self.logger.info("  >> Phase 4: LOFO Feature Evaluation (Parallelized)")

        output_dir = round_dir / constants.ROUND_FEATURE_EVALUATION_DIR
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

        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        save_dataframe(lofo_summary, output_dir / "lofo_summary.parquet", excel_copy=excel_copy, index=False)
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
        """
        Finalize round with atomic feature list update.
        Ensures feature list is consistent with what will be used in next round.
        """
        # Calculate the feature list for the NEXT round (current - dropped)
        next_round_features = [f for f in self.active_features if f != dropped_feature]
        next_round_checksum = self._calculate_feature_checksum(next_round_features)

        flag_data = {
            "timestamp": datetime.now().isoformat(),
            "round": self.current_round,
            "dropped_feature": dropped_feature,
            "features_remaining": len(next_round_features),
            "stopping_status": stop_reason if stop_reason else "CONTINUE",
            "next_round_feature_checksum": next_round_checksum,
            "next_round_feature_count": len(next_round_features)
        }

        # Atomic write of flag file
        flag_path = round_dir / "_ROUND_COMPLETE.flag"
        temp_flag_path = flag_path.with_suffix('.tmp')

        try:
            with open(temp_flag_path, 'w') as f:
                json.dump(flag_data, f, indent=2)
            temp_flag_path.replace(flag_path)
            self.logger.info(f"  >> Round {self.current_round} Complete. Dropped: {dropped_feature}")
            self.logger.info(f"  >> Next round will use {len(next_round_features)} features (checksum: {next_round_checksum[:8]}...)")
        except Exception as e:
            if temp_flag_path.exists():
                temp_flag_path.unlink()
            raise RuntimeError(f"Failed to finalize round atomically: {e}")

    def _load_round_state(self, round_dir: Path):
        """
        Load round state with checksum validation to ensure consistency.
        """
        try:
            # Load completion flag
            with open(round_dir / "_ROUND_COMPLETE.flag", 'r') as f:
                flag = json.load(f)

            # Load feature list
            datasets_dir = round_dir / constants.ROUND_DATASETS_DIR
            legacy_dataset_dir = round_dir / "00_DATASETS"
            if not datasets_dir.exists() and legacy_dataset_dir.exists():
                datasets_dir = legacy_dataset_dir
            features_path = datasets_dir / "feature_list.json"
            checksum_path = datasets_dir / "feature_list_checksum.txt"

            with open(features_path, 'r') as f:
                feature_data = json.load(f)

            # Handle both old format (list) and new format (dict)
            if isinstance(feature_data, list):
                features_at_start = feature_data
                self.logger.warning("  Feature list in old format (no checksum validation possible)")
            else:
                features_at_start = feature_data['features']

                # Validate checksum if available
                if checksum_path.exists():
                    with open(checksum_path, 'r') as f:
                        saved_checksum = f.read().strip()

                    calculated_checksum = self._calculate_feature_checksum(features_at_start)

                    if saved_checksum != calculated_checksum:
                        raise RuntimeError(
                            f"Feature list checksum mismatch! "
                            f"Saved: {saved_checksum[:8]}..., Calculated: {calculated_checksum[:8]}... "
                            f"Feature list may be corrupted."
                        )
                    self.logger.info(f"  Feature list checksum validated: {saved_checksum[:8]}...")

            # Calculate next round features
            dropped = flag.get('dropped_feature')
            next_features = [f for f in features_at_start if f != dropped]

            # Validate against expected checksum from flag (if available)
            if 'next_round_feature_checksum' in flag:
                expected_checksum = flag['next_round_feature_checksum']
                actual_checksum = self._calculate_feature_checksum(next_features)

                if expected_checksum != actual_checksum:
                    raise RuntimeError(
                        f"Next round feature checksum mismatch! "
                        f"Expected: {expected_checksum[:8]}..., Actual: {actual_checksum[:8]}... "
                        f"Feature list may be corrupted."
                    )
                self.logger.info(f"  Next round feature checksum validated: {expected_checksum[:8]}...")

            # Validate feature count
            if 'next_round_feature_count' in flag:
                expected_count = flag['next_round_feature_count']
                if len(next_features) != expected_count:
                    raise RuntimeError(
                        f"Feature count mismatch! Expected: {expected_count}, Actual: {len(next_features)}"
                    )

            self.active_features = next_features
            self.logger.info(f"  Loaded state from Round {flag['round']}. Next features: {len(self.active_features)}")

        except Exception as e:
            self.logger.error(f"Failed to load resume state from {round_dir}: {e}")
            raise RuntimeError(f"Resume failed. {e}")

    # --- Helper Methods ---

    def _calculate_feature_checksum(self, features: List[str]) -> str:
        """
        Calculate SHA256 checksum of feature list for validation.
        Features are sorted before hashing to ensure consistent checksums.
        """
        # Sort features to ensure consistent checksum regardless of order
        sorted_features = sorted(features)
        feature_string = json.dumps(sorted_features, sort_keys=True)
        return hashlib.sha256(feature_string.encode('utf-8')).hexdigest()

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
        Includes validation that training features match expected active features.
        """
        from modules.model_factory import ModelFactory

        # Prepare data - drop all non-feature columns
        drop_cols = self.config['data']['drop_columns'] + [
            self.config['data']['target_sin'],
            self.config['data']['target_cos'],
            'angle_deg', 'angle_bin', 'hs_bin', 'combined_bin', 'row_index'
        ]

        X = train_df.drop(columns=drop_cols, errors='ignore')
        y = train_df[[self.config['data']['target_sin'], self.config['data']['target_cos']]]

        # Validate that the features being used for training match expectations
        actual_features = sorted(X.columns.tolist())
        expected_features = sorted([f for f in self.active_features if f in train_df.columns])

        if actual_features != expected_features:
            missing_features = set(expected_features) - set(actual_features)
            extra_features = set(actual_features) - set(expected_features)
            error_msg = f"Feature mismatch detected in training for '{model_name}':\n"
            if missing_features:
                error_msg += f"  Missing features: {missing_features}\n"
            if extra_features:
                error_msg += f"  Unexpected features: {extra_features}\n"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.logger.debug(f"  Training '{model_name}' with {len(actual_features)} features (checksum: {self._calculate_feature_checksum(actual_features)[:8]}...)")

        # Create and train model
        model = ModelFactory.create(model_config.get('model'), model_config.get('params', {}))
        model.fit(X, y)

        return model

    def _make_predictions(self, model, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """
        Make predictions using the trained model and format as expected by evaluation engine.
        """
        from utils.circular_metrics import reconstruct_angle
        hs_col = self.config['data'].get('hs_column')

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

        # Carry wave height info (meters + feet) for downstream diagnostics
        if hs_col and hs_col in df.columns:
            predictions_df[hs_col] = df[hs_col].values
            predictions_df[f"{hs_col}_ft"] = df[hs_col].values * 3.28084
            predictions_df["Hs_ft"] = predictions_df[f"{hs_col}_ft"]
        if 'hs_bin' in df.columns:
            predictions_df['hs_bin'] = df['hs_bin'].values

        return predictions_df

    def _record_cv_consistency(self, round_dir: Path, best_config: Dict[str, Any]) -> None:
        """
        Capture CV fold consistency for the best HPO config and roll up over rounds.
        """
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        config_id = best_config.get("config_id")
        model = best_config.get("model")
        if config_id is None or model is None:
            return
        src = (
            round_dir
            / constants.ROUND_GRID_SEARCH_DIR
            / constants.HPO_OPTIMIZATION_DIR
            / "results"
            / f"cv_consistency_{model}_{int(config_id):04d}.parquet"
        )
        if not src.exists():
            return
        try:
            df = pd.read_parquet(src)
        except Exception:
            return
        if df.empty:
            return
        df['round'] = self.current_round
        df['model'] = model
        df['config_id'] = config_id
        dest = round_dir / constants.ROUND_GRID_SEARCH_DIR / "cv_consistency_best.parquet"
        save_dataframe(df, dest, excel_copy=excel_copy, index=False)
        self.cv_consistency_history.extend(df.to_dict(orient="records"))
        save_dataframe(pd.DataFrame(self.cv_consistency_history), self.base_dir / "cv_fold_consistency_over_rounds.parquet", excel_copy=excel_copy, index=False)

    def _run_significance_analysis(self, round_dir: Path, val_preds: pd.DataFrame, test_preds: pd.DataFrame) -> None:
        """Compare current round errors to previous round using paired tests."""
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        rows = []
        for split_name, preds, prev_errors in [
            ("val", val_preds, self.prev_val_errors),
            ("test", test_preds, self.prev_test_errors),
        ]:
            if prev_errors is None:
                rows.append({
                    "round": self.current_round,
                    "split": split_name,
                    "p_value_ttest": np.nan,
                    "p_value_wilcoxon": np.nan,
                    "cohens_d": np.nan,
                    "significant": False
                })
                continue
            result = self.evaluation_engine.compare_splits(
                baseline_errors=pd.Series(prev_errors),
                candidate_errors=preds['abs_error'],
                alpha=0.05,
            )
            result.update({"round": self.current_round, "split": split_name})
            rows.append(result)

        sig_df = pd.DataFrame(rows)
        save_dataframe(sig_df, round_dir / constants.ROUND_BASE_MODEL_RESULTS_DIR / "significance_comparison.parquet", excel_copy=excel_copy, index=False)
        self.significance_history.extend(rows)
        save_dataframe(pd.DataFrame(self.significance_history), self.base_dir / "significance_over_rounds.parquet", excel_copy=excel_copy, index=False)
        # Persist cumulative statistical tests summary for transparency
        save_dataframe(pd.DataFrame(self.significance_history), self.base_dir / "statistical_tests_round_comparisons.parquet", excel_copy=excel_copy, index=False)

        # Update cache for next round
        self.prev_val_errors = val_preds['abs_error'].values
        self.prev_test_errors = test_preds['abs_error'].values

    def _compare_baseline_vs_dropped(self, round_dir: Path, baseline_val_preds: pd.DataFrame, baseline_test_preds: pd.DataFrame, dropped_info: Dict[str, Any]) -> None:
        """Significance and safety comparisons between baseline and dropped model predictions."""
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        summaries = []

        for split_name, base_df, drop_df in [
            ("val", baseline_val_preds, dropped_info.get("val_predictions")),
            ("test", baseline_test_preds, dropped_info.get("test_predictions")),
        ]:
            if base_df is None or drop_df is None or base_df.empty or drop_df.empty:
                continue

            sig = self.evaluation_engine.compare_splits(
                baseline_errors=base_df['abs_error'],
                candidate_errors=drop_df['abs_error'],
                alpha=0.05,
            )
            sig.update({"round": self.current_round, "split": split_name, "model": "baseline_vs_dropped"})

            safety_base = safety_threshold_summary(base_df['abs_error'])
            safety_base['model'] = 'baseline'
            safety_base['split'] = split_name
            safety_base['round'] = self.current_round

            safety_drop = safety_threshold_summary(drop_df['abs_error'])
            safety_drop['model'] = 'dropped'
            safety_drop['split'] = split_name
            safety_drop['round'] = self.current_round

            sig_df = pd.DataFrame([sig])
            safety_df = pd.concat([safety_base, safety_drop], ignore_index=True)

            save_dataframe(sig_df, round_dir / constants.ROUND_COMPARISON_DIR / f"significance_baseline_vs_dropped_{split_name}.parquet", excel_copy=excel_copy, index=False)
            save_dataframe(safety_df, round_dir / constants.ROUND_COMPARISON_DIR / f"safety_baseline_vs_dropped_{split_name}.parquet", excel_copy=excel_copy, index=False)

            summaries.append(sig)
            self.safety_history.extend(safety_df.to_dict(orient="records"))

        # Roll-up
        if summaries:
            save_dataframe(pd.DataFrame(summaries), self.base_dir / "significance_baseline_vs_dropped_over_rounds.parquet", excel_copy=excel_copy, index=False)
        if self.safety_history:
            save_dataframe(pd.DataFrame(self.safety_history), self.base_dir / "safety_threshold_summary_all_rounds.parquet", excel_copy=excel_copy, index=False)

    def _summarize_safety(self, round_dir: Path, val_preds: pd.DataFrame, test_preds: pd.DataFrame) -> None:
        """Summarize safety tiers for current round and roll up."""
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        summaries = []
        for split_name, preds in [("val", val_preds), ("test", test_preds)]:
            summary_df = safety_threshold_summary(preds['abs_error'])
            summary_df['split'] = split_name
            summary_df['round'] = self.current_round
            summaries.append(summary_df)

        all_summary = pd.concat(summaries, ignore_index=True)
        save_dataframe(all_summary, round_dir / constants.ROUND_BASE_MODEL_RESULTS_DIR / "safety_threshold_summary.parquet", excel_copy=excel_copy, index=False)
        self.safety_history.extend(all_summary.to_dict(orient="records"))
        save_dataframe(pd.DataFrame(self.safety_history), self.base_dir / "safety_threshold_summary_all_rounds.parquet", excel_copy=excel_copy, index=False)

        # Safety gates evaluation (configurable thresholds)
        gates_cfg = self.config.get("error_analysis", {}).get("safety_gates", {})
        
        if not gates_cfg: # If no safety_gates config, skip gate status
            self.logger.info("Safety gates disabled (no config provided in 'error_analysis.safety_gates').")
            return

        max_critical = gates_cfg.get("max_critical_pct", 25.0)
        max_warning = gates_cfg.get("max_warning_pct", 50.0)
        gate_rows = []
        for split_name in ["val", "test"]:
            split_df = all_summary[all_summary["split"] == split_name]
            crit_pct = float(split_df.loc[split_df["tier"] == "CRITICAL", "percentage"].sum() if not split_df.empty else 0.0)
            warn_pct = float(split_df.loc[split_df["tier"] == "WARNING", "percentage"].sum() if not split_df.empty else 0.0)
            status = "PASS"
            if crit_pct > max_critical:
                status = "FAIL"
            elif warn_pct > max_warning:
                status = "WARN"
            gate_rows.append({
                "round": self.current_round,
                "split": split_name,
                "critical_pct": crit_pct,
                "warning_pct": warn_pct,
                "max_critical_pct": max_critical,
                "max_warning_pct": max_warning,
                "status": status,
            })
        if gate_rows:
            gates_df = pd.DataFrame(gate_rows)
            save_dataframe(gates_df, round_dir / constants.ROUND_BASE_MODEL_RESULTS_DIR / "safety_gate_status.parquet", excel_copy=excel_copy, index=False)
            save_dataframe(gates_df, self.base_dir / "safety_gate_status_all_rounds.parquet", excel_copy=excel_copy, index=False)
