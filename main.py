import sys
import logging
import json
import pandas as pd
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import ALL Modules
from modules.config_manager import ConfigurationManager
from modules.logging_config import LoggingConfigurator
from modules.data_manager import DataManager
from modules.split_engine import SplitEngine
from modules.hpo_search_engine import HPOSearchEngine
from modules.hyperparameter_analyzer import HyperparameterAnalyzer
from modules.training_engine import TrainingEngine
from modules.prediction_engine import PredictionEngine
from modules.evaluation_engine import EvaluationEngine
from modules.diagnostics_engine import DiagnosticsEngine
from modules.error_analysis_engine import ErrorAnalysisEngine
from modules.bootstrapping_engine import BootstrappingEngine
from modules.ensembling_engine import EnsemblingEngine
from modules.stability_engine import StabilityEngine
from modules.reporting_engine import ReportingEngine
from modules.reproducibility_engine import ReproducibilityEngine
from utils.exceptions import RiserMLException


# =====================================================================
#                        PIPELINE MAIN FUNCTION
# =====================================================================
def main():
    try:
        print("\n" + "=" * 80)
        print("    OFFSHORE RISER ML PREDICTION SYSTEM - PIPELINE EXECUTION")
        print("=" * 80 + "\n")

        # ---------------------------------------------------------------
        # PHASE 1: FOUNDATION
        # ---------------------------------------------------------------
        config_manager = ConfigurationManager()
        config = config_manager.load_and_validate()
        run_id = config_manager.generate_run_id()

        logging_configurator = LoggingConfigurator(config)
        logging_configurator.setup()
        logger = logging_configurator.get_logger('pipeline')

        base_dir = config.get('outputs', {}).get('base_results_dir', 'results')
        results_dir = Path(base_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        config_manager.save_artifacts(str(results_dir))

        logger.info("=" * 60)
        logger.info("PHASE 1: FOUNDATION")
        logger.info("=" * 60)
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Results Directory: {results_dir.absolute()}")

        # ---------------------------------------------------------------
        # PHASE 2: DATA PREPARATION
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: DATA PREPARATION")
        logger.info("=" * 60)

        data_manager = DataManager(config, logger)
        validated_data = data_manager.execute(run_id)
        logger.info(f"[OK] Data validated: {len(validated_data)} samples")

        split_engine = SplitEngine(config, logger)
        train_df, val_df, test_df = split_engine.execute(validated_data, run_id)
        logger.info(f"[OK] Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # ---------------------------------------------------------------
        # PHASE 3: HYPERPARAMETER OPTIMIZATION (SEARCH)
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: HYPERPARAMETER OPTIMIZATION (SEARCH)")
        logger.info("=" * 60)

        hpo_enabled = config.get('hyperparameters', {}).get('enabled', False)

        if hpo_enabled:
            hpo_engine = HPOSearchEngine(config, logger)
            # New signature: includes validation + test sets
            best_config = hpo_engine.execute(train_df, val_df, test_df, run_id)
        else:
            best_config = {'model': config['models']['native'][0], 'params': {}}
            logger.info("HPO disabled. Using default model configuration.")

        # ---------------------------------------------------------------
        # PHASE 4: HYPERPARAMETER ANALYSIS (VISUALIZATION)
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4: HYPERPARAMETER ANALYSIS (VISUALIZATION)")
        logger.info("=" * 60)

        if hpo_enabled:
            hpo_analyzer = HyperparameterAnalyzer(config, logger)
            hpo_analyzer.analyze(run_id)
            logger.info("[OK] HPO analysis complete")
        else:
            logger.info("Skipping Phase 4 (HPO disabled)")

        # ---------------------------------------------------------------
        # PHASE 5: MODEL TRAINING
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 5: MODEL TRAINING")
        logger.info("=" * 60)

        training_engine = TrainingEngine(config, logger)
        final_model = training_engine.train(train_df, best_config, run_id)
        logger.info(f"[OK] Model trained: {best_config.get('model')}")

        # ---------------------------------------------------------------
        # PHASE 6: PREDICTION
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 6: PREDICTION")
        logger.info("=" * 60)

        prediction_engine = PredictionEngine(config, logger)
        preds_val = prediction_engine.predict(final_model, val_df, "val", run_id)
        preds_test = prediction_engine.predict(final_model, test_df, "test", run_id)
        logger.info("[OK] Predictions generated")

        # ---------------------------------------------------------------
        # PHASE 7: EVALUATION
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 7: EVALUATION")
        logger.info("=" * 60)

        eval_engine = EvaluationEngine(config, logger)
        metrics_val = eval_engine.evaluate(preds_val, "val", run_id)
        metrics_test = eval_engine.evaluate(preds_test, "test", run_id)

        logger.info(f"Validation CMAE:         {metrics_val['cmae']:.4f}")
        logger.info(f"Validation Accuracy@5°: {metrics_val['accuracy_at_5deg']:.2f}%")
        logger.info(f"Test CMAE:               {metrics_test['cmae']:.4f}")
        logger.info(f"Test Accuracy@5°:       {metrics_test['accuracy_at_5deg']:.2f}%")

        # ---------------------------------------------------------------
        # PHASE 8: DIAGNOSTICS
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 8: DIAGNOSTICS")
        logger.info("=" * 60)

        diag_engine = DiagnosticsEngine(config, logger)
        diag_engine.generate_all(preds_val, "val", run_id)
        diag_engine.generate_all(preds_test, "test", run_id)
        logger.info("[OK] Diagnostic plots generated")

        # ---------------------------------------------------------------
        # PHASE 9: ERROR ANALYSIS
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 9: ERROR ANALYSIS")
        logger.info("=" * 60)

        ea_engine = ErrorAnalysisEngine(config, logger)
        ea_engine.analyze(preds_val, val_df, "val", run_id)
        ea_engine.analyze(preds_test, test_df, "test", run_id)
        logger.info("[OK] Error analysis complete")

        # ---------------------------------------------------------------
        # PHASE 10: ADVANCED ANALYTICS
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 10: ADVANCED ANALYTICS")
        logger.info("=" * 60)

        boot_engine = BootstrappingEngine(config, logger)
        boot_engine.bootstrap(preds_test, "test", run_id)

        ens_engine = EnsemblingEngine(config, logger)
        ens_engine.ensemble([preds_test], [metrics_test], "test", run_id)

        stab_engine = StabilityEngine(config, logger)
        stab_engine.run_stability_analysis(validated_data, run_id)

        # ---------------------------------------------------------------
        # PHASE 11: REPORTING
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 11: REPORTING")
        logger.info("=" * 60)

        hpo_artifacts = _prepare_hpo_artifacts(results_dir, hpo_enabled, logger)

        report_data = {
            'run_metadata': {'id': run_id},
            'model_info': best_config,
            'metrics': {'val': metrics_val, 'test': metrics_test},
            'hpo_analysis': hpo_artifacts,
            'plots': _collect_diagnostic_plots(results_dir),
        }

        rep_engine = ReportingEngine(config, logger)
        report_path = rep_engine.generate_report(report_data, run_id)
        logger.info(f"[OK] PDF Report: {report_path}")

        # ---------------------------------------------------------------
        # PHASE 12: REPRODUCIBILITY
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 12: REPRODUCIBILITY PACKAGING")
        logger.info("=" * 60)

        repro_engine = ReproducibilityEngine(config, logger)
        pkg_path = repro_engine.package(run_id)
        logger.info(f"[OK] Reproducibility Package: {pkg_path}")

        # ---------------------------------------------------------------
        # COMPLETION
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("    PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"\n>> Final Report:      {report_path}")
        logger.info(f">> Package:           {pkg_path}")
        logger.info(f">> Results Directory: {results_dir.absolute()}")
        logger.info("\n" + "=" * 80 + "\n")

    except RiserMLException as e:
        print(f"\n[FATAL ERROR] {e.__class__.__name__}: {str(e)}")
        if 'logger' in locals():
            logger.critical(f"{e.__class__.__name__}: {str(e)}")
        sys.exit(1)

    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        if 'logger' in locals():
            logger.critical(f"Unexpected: {str(e)}", exc_info=True)
        sys.exit(1)


# =====================================================================
#                   H E L P E R   F U N C T I O N S
# =====================================================================
def _prepare_hpo_artifacts(results_dir: Path, enabled: bool, logger) -> dict:
    """Collect hyperparameter optimization results and plots."""
    artifacts = {'enabled': enabled, 'models': {}, 'tables': {}}

    if not enabled:
        return artifacts

    analysis_dir = results_dir / "04_HYPERPARAMETER_ANALYSIS"

    # Collect plots
    viz_dir = analysis_dir / "visualizations"
    if viz_dir.exists():
        for model_dir in viz_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            plots_by_metric = {}
            for metric_dir in model_dir.iterdir():
                if not metric_dir.is_dir():
                    continue

                all_pngs = [str(p) for p in metric_dir.rglob("*.png")]
                if all_pngs:
                    plots_by_metric[metric_dir.name] = sorted(all_pngs)

            if plots_by_metric:
                artifacts['models'][model_name] = plots_by_metric

    # Collect Excel tables
    excel_path = analysis_dir / "optimal_ranges.xlsx"
    if excel_path.exists():
        try:
            xls = pd.read_excel(excel_path, sheet_name=None)
            for sheet_name, df in xls.items():
                df_clean = df.fillna("N/A")
                artifacts['tables'][sheet_name] = [df_clean.columns.tolist()] + df_clean.values.tolist()
        except Exception as e:
            logger.warning(f"Could not load HPO Excel: {e}")

    return artifacts


def _collect_diagnostic_plots(results_dir: Path):
    """Collect standard diagnostic plot paths."""
    diag_dir = results_dir / "07_DIAGNOSTICS"
    plots = []

    preferred = [
        diag_dir / "scatter_plots/actual_vs_pred_test.png",
    diag_dir / "distribution_plots/error_hist_test.png",
    results_dir / "10_ADVANCED_ANALYTICS/bootstrapping/test/bootstrap_dist_cmae.png"
    ]

    for p in preferred:
        if p.exists():
            plots.append(str(p))

    return plots


if __name__ == "__main__":
    main()
