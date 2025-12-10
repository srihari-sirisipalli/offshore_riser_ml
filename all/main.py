import sys
import logging
import json
import pandas as pd
import numpy as np
import traceback
import random
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# --- Core Infrastructure ---
from modules.config_manager import ConfigurationManager
from modules.logging_config import LoggingConfigurator
from modules.data_manager import DataManager
from modules.split_engine import SplitEngine
from modules.reproducibility_engine import ReproducibilityEngine
from utils.exceptions import RiserMLException

# --- The New Circular Engine ---
from modules.rfe.rfe_controller import RFEController

def _setup_global_determinism(config: dict, logger: logging.Logger):
    """
    Enforces the Global Seed across all libraries to ensure 100% reproducibility.
    """
    seed = config.get('splitting', {}).get('seed', 456)
    logger.info(f"Setting Global Deterministic Seed: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        # If using joblib directly for parallel backends that need seeding
        import joblib
        # (Joblib usually relies on numpy seed within workers, but explicit setting is good practice)
    except ImportError:
        pass

def main():
    logger = None
    try:
        # Enable copy-on-write for performance
        pd.options.mode.copy_on_write = True

        print("\n" + "=" * 80)
        print("    OFFSHORE RISER CIRCULAR RFE PIPELINE (PRODUCTION)")
        print("=" * 80 + "\n")

        # ---------------------------------------------------------------
        # PHASE 0: INITIALIZATION
        # ---------------------------------------------------------------
        # 1. Load Config
        config_manager = ConfigurationManager()
        config = config_manager.load_and_validate()

        # 2. Setup Logging
        logging_configurator = LoggingConfigurator(config)
        logging_configurator.setup()
        logger = logging_configurator.get_logger('pipeline')

        # 3. Setup Folder Structure (No Timestamps)
        # The output directory is taken directly from the config.
        run_dir = Path(config.get('outputs', {}).get('base_results_dir', 'results'))
        
        # Use the folder name as a static run ID for consistency
        run_id = run_dir.name
        config_manager.run_id = run_id
        
        # Override config path to ensure all modules use the correct, absolute path
        config['outputs']['base_results_dir'] = str(run_dir.absolute())
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # 4. Save Initial Config Artifacts
        config_manager.save_artifacts(str(run_dir))
        
        # 5. Set Global Seed
        _setup_global_determinism(config, logger)

        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output Directory: {run_dir.absolute()}")

        # ---------------------------------------------------------------
        # PHASE 1: DATA INGESTION & SPLITTING
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: DATA INGESTION & SPLITTING")
        logger.info("=" * 60)

        # Data Loading & Validation
        data_manager = DataManager(config, logger)
        validated_data = data_manager.execute(run_id)
        
        # Smart Stratified Split (Angle x Hs)
        split_engine = SplitEngine(config, logger)
        train_df, val_df, test_df = split_engine.execute(validated_data, run_id)
        
        logger.info(f"Master Splits Created: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # ---------------------------------------------------------------
        # PHASE 2: CIRCULAR RECURSIVE FEATURE ELIMINATION
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: CIRCULAR RECURSIVE FEATURE ELIMINATION")
        logger.info("=" * 60)

        # The RFE Controller takes over control flow here.
        # It manages HPO, Baseline Training, LOFO, and Feature Dropping loops.
        rfe_controller = RFEController(config, logger)
        rfe_controller.run(train_df, val_df, test_df)

        # ---------------------------------------------------------------
        # PHASE 3: FINALIZATION & REPRODUCIBILITY
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: FINALIZATION & REPRODUCIBILITY")
        logger.info("=" * 60)

        # Generate Reproducibility Package
        # Bundles the 'model_reconstruction_mapping.xlsx' and final artifacts
        repro_engine = ReproducibilityEngine(config, logger)
        pkg_path = repro_engine.package(run_id)
        
        logger.info("-" * 60)
        logger.info(f"Pipeline Completed Successfully.")
        logger.info(f"Reproducibility Package: {pkg_path}")
        logger.info("-" * 60)

    except RiserMLException as e:
        msg = f"Pipeline Error: {str(e)}"
        print(f"\n[FATAL] {msg}")
        if logger: logger.critical(msg)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n[USER] Pipeline interrupted by user.")
        if logger: logger.warning("Pipeline interrupted by user.")
        sys.exit(130)

    except Exception as e:
        msg = f"Unexpected System Error: {str(e)}"
        print(f"\n[CRITICAL] {msg}")
        if logger: logger.critical(msg, exc_info=True)
        else: traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()