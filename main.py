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

def main():
    try:
        # =================================================================
        # PHASE 1: FOUNDATION
        # =================================================================
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
        
        logger.info(f"Pipeline Started. Run ID: {run_id}")
        
        # =================================================================
        # PHASE 2: DATA LAYER
        # =================================================================
        logger.info("-" * 40)
        logger.info("Phase 2: Data Layer")
        
        data_manager = DataManager(config, logger)
        validated_data = data_manager.execute(run_id)
        
        split_engine = SplitEngine(config, logger)
        train_df, val_df, test_df = split_engine.execute(validated_data, run_id)
        
        # =================================================================
        # PHASE 3: MODEL DEVELOPMENT
        # =================================================================
        logger.info("-" * 40)
        logger.info("Phase 3: Model Development")
        
        # 1. HPO
        hpo_enabled = config.get('hyperparameters', {}).get('enabled', False)
        if hpo_enabled:
            hpo_engine = HPOSearchEngine(config, logger)
            best_config = hpo_engine.execute(train_df, run_id)
        else:
            best_config = {'model': config['models']['native'][0], 'params': {}}
            logger.info("HPO disabled. Using default model configuration.")
            
        # 2. Train Final
        training_engine = TrainingEngine(config, logger)
        final_model = training_engine.train(train_df, best_config, run_id)
        
        # 3. Predict
        prediction_engine = PredictionEngine(config, logger)
        preds_val = prediction_engine.predict(final_model, val_df, "val", run_id)
        preds_test = prediction_engine.predict(final_model, test_df, "test", run_id)

        # =================================================================
        # PHASE 4: HYPERPARAMETER ANALYSIS
        # =================================================================
        logger.info("-" * 40)
        logger.info("Phase 4: Hyperparameter Analysis")
        
        hpo_artifacts = {'enabled': False, 'models': {}, 'tables': {}}
        
        if hpo_enabled:
            hpo_analyzer = HyperparameterAnalyzer(config, logger)
            hpo_analyzer.analyze(run_id)
            
            hpo_artifacts['enabled'] = True
            hpo_analysis_dir = results_dir / "04_HYPERPARAMETER_SEARCH" / "ANALYSIS"
            
            # 1. Collect Plots per Model (Iterate subdirectories)
            if hpo_analysis_dir.exists():
                for model_dir in [d for d in hpo_analysis_dir.iterdir() if d.is_dir()]:
                    model_name = model_dir.name
                    # Find all pngs in this model's folder
                    plots = sorted([str(p) for p in model_dir.glob("*.png")])
                    if plots:
                        hpo_artifacts['models'][model_name] = plots
                        logger.info(f"Found {len(plots)} plots for {model_name}")

                # 2. Load Excel Tables per Model (Read all sheets)
                excel_path = hpo_analysis_dir / "optimal_parameter_ranges.xlsx"
                if excel_path.exists():
                    try:
                        xls = pd.read_excel(excel_path, sheet_name=None)
                        for sheet_name, df in xls.items():
                            # Convert to list of lists [Header, Row1, Row2...] for ReportLab
                            # Fill NaNs to avoid report crashes
                            df_clean = df.fillna('N/A')
                            hpo_artifacts['tables'][sheet_name] = [df_clean.columns.tolist()] + df_clean.values.tolist()
                    except Exception as e:
                        logger.warning(f"Could not load HPO excel: {e}")
        else:
            logger.info("Skipping Phase 4 (HPO Disabled)")
        
        # =================================================================
        # PHASE 5: EVALUATION & DIAGNOSTICS
        # =================================================================
        logger.info("-" * 40)
        logger.info("Phase 5: Evaluation & Diagnostics")
        
        eval_engine = EvaluationEngine(config, logger)
        metrics_val = eval_engine.evaluate(preds_val, "val", run_id)
        metrics_test = eval_engine.evaluate(preds_test, "test", run_id)
        
        diag_engine = DiagnosticsEngine(config, logger)
        diag_engine.generate_all(preds_test, "test", run_id)
        diag_engine.generate_all(preds_val, "val", run_id)

        # =================================================================
        # PHASE 6: ERROR ANALYSIS
        # =================================================================
        logger.info("-" * 40)
        logger.info("Phase 6: Error Analysis")
        
        ea_engine = ErrorAnalysisEngine(config, logger)
        ea_engine.analyze(preds_test, test_df, "test", run_id)
        ea_engine.analyze(preds_val, val_df, "val", run_id)

        # =================================================================
        # PHASE 7: ADVANCED ANALYTICS
        # =================================================================
        logger.info("-" * 40)
        logger.info("Phase 7: Advanced Analytics")
        
        # 1. Bootstrap
        boot_engine = BootstrappingEngine(config, logger)
        boot_results = boot_engine.bootstrap(preds_test, "test", run_id)
        
        # 2. Ensemble (Optional)
        ens_engine = EnsemblingEngine(config, logger)
        ens_engine.ensemble([preds_test], [metrics_test], "test", run_id)
        
        # 3. Stability (Optional)
        stab_engine = StabilityEngine(config, logger)
        stab_results = stab_engine.run_stability_analysis(validated_data, run_id)
        
        # =================================================================
        # PHASE 8: OUTPUT LAYER
        # =================================================================
        logger.info("-" * 40)
        logger.info("Phase 8: Output Layer")
        
        # 1. Generate PDF Report
        report_data = {
            'run_metadata': {'id': run_id},
            'model_info': best_config,
            'metrics': {'val': metrics_val, 'test': metrics_test},
            'hpo_analysis': hpo_artifacts, 
            'plots': [
                str(Path(results_dir) / "08_DIAGNOSTICS/scatter_plots/actual_vs_pred_test.png"),
                str(Path(results_dir) / "08_DIAGNOSTICS/distribution_plots/error_hist_test.png"),
                str(Path(results_dir) / "09_ADVANCED_ANALYTICS/bootstrapping/test/bootstrap_dist_cmae.png")
            ]
        }
        
        rep_engine = ReportingEngine(config, logger)
        report_path = rep_engine.generate_report(report_data, run_id)
        
        # 2. Reproducibility Package
        repro_engine = ReproducibilityEngine(config, logger)
        pkg_path = repro_engine.package(run_id)
        
        logger.info("="*60)
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Final Report: {report_path}")
        logger.info(f"Package:      {pkg_path}")
        logger.info("="*60)
        
    except RiserMLException as e:
        print(f"\n[FATAL ERROR] {e.__class__.__name__}: {str(e)}")
        if 'logger' in locals(): logger.critical(f"{e.__class__.__name__}: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        if 'logger' in locals(): logger.critical(f"Unexpected: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()