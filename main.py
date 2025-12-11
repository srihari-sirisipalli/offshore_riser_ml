#!/usr/bin/env python
"""
Offshore Riser ML Pipeline - Main Entry Point
Orchestrates the complete machine learning pipeline for offshore riser angle prediction.
"""
import sys
import logging
import argparse
import traceback
import random
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Core Infrastructure
from modules.config_manager import ConfigurationManager
from modules.logging_config import LoggingConfigurator
from modules.data_manager import DataManager
from modules.split_engine import SplitEngine
from modules.reproducibility_engine import ReproducibilityEngine
from utils.results_layout import ResultsLayoutManager
from utils.exceptions import RiserMLException

# RFE Controller
from modules.rfe.rfe_controller import RFEController


def parse_arguments():
    """
    Parse command-line arguments for configurable pipeline execution.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Offshore Riser ML Pipeline - RFE & Hyperparameter Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="Path to the configuration JSON file"
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier (defaults to timestamp if not provided)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing run directory (requires --run-id or existing base_results_dir in config)"
    )

    parser.add_argument(
        "--skip-rfe",
        action="store_true",
        help="Skip the RFE phase and only run baseline model training"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and setup without running the pipeline"
    )

    return parser.parse_args()


def setup_global_determinism(config: dict, logger: logging.Logger):
    """
    Enforce global determinism across all libraries for 100%% reproducibility.

    Args:
        config: Configuration dictionary containing seed settings.
        logger: Logger instance for recording seed information.
    """
    seed = config.get('splitting', {}).get('seed', 456)
    logger.info(f"Setting Global Deterministic Seed: {seed}")

    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # Pandas copy-on-write for performance and safety
    pd.options.mode.copy_on_write = True

    # Environment variables for additional libraries (if used)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Try to set seeds for joblib if available
    try:
        import joblib
        # Joblib relies on numpy seed within workers
    except ImportError:
        pass


def validate_environment(logger: logging.Logger):
    """
    Validate the runtime environment and dependencies.

    Args:
        logger: Logger instance for recording validation results.

    Raises:
        RuntimeError: If critical dependencies are missing or incompatible.
    """
    logger.info("Validating environment...")

    # Check Python version
    if sys.version_info < (3, 8):
        raise RuntimeError(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")

    # Check critical imports
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'joblib', 'jsonschema'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        raise RuntimeError(f"Missing required packages: {', '.join(missing)}")

    logger.info("Environment validation passed")


def setup_run_directory(config: dict, run_id: str = None, resume: bool = False, logger: logging.Logger = None):
    """
    Setup or resume the run directory structure.

    Args:
        config: Configuration dictionary.
        run_id: Optional run identifier.
        resume: Whether to resume from existing directory.
        logger: Logger instance.

    Returns:
        tuple: (run_dir Path, run_id string)
    """
    base_results_dir = config.get('outputs', {}).get('base_results_dir', 'results')

    if resume:
        # Resume mode: use existing directory
        run_dir = Path(base_results_dir).absolute()
        if not run_dir.exists():
            raise RiserMLException(f"Cannot resume: directory '{run_dir}' does not exist")
        run_id = run_dir.name
        if logger:
            logger.info(f"Resuming from existing run: {run_id}")
    else:
        # New run mode
        if run_id:
            # User-specified run ID
            run_dir = Path(f"{base_results_dir}_{run_id}").absolute()
        else:
            # Use directory name from config or create timestamped one
            run_dir = Path(base_results_dir).absolute()
            run_id = run_dir.name

        # Create directory
        run_dir.mkdir(parents=True, exist_ok=True)
        if logger:
            logger.info(f"Created new run directory: {run_id}")

    return run_dir, run_id


def main():
    """
    Main pipeline orchestration function.

    Handles configuration loading, environment validation, logging setup,
    and sequential execution of all pipeline phases.

    Returns:
        int: Exit code (0 for success, 1 for errors)
    """
    logger = None
    args = None

    try:
        # Parse command-line arguments
        args = parse_arguments()

        # Print banner
        print("\n" + "=" * 80)
        print("    OFFSHORE RISER CIRCULAR RFE PIPELINE (PRODUCTION)")
        print("=" * 80 + "\n")

        # ---------------------------------------------------------------
        # PHASE 0: INITIALIZATION & VALIDATION
        # ---------------------------------------------------------------

        # 1. Load and validate configuration
        config_manager = ConfigurationManager(config_path=args.config)
        config = config_manager.load_and_validate()

        # Override config settings from CLI if provided
        if args.verbose:
            config['logging']['level'] = 'DEBUG'

        # 2. Setup logging
        logging_configurator = LoggingConfigurator(config)
        logging_configurator.setup()
        logger = logging_configurator.get_logger('pipeline')

        logger.info("Pipeline initialization started")
        logger.info(f"Configuration loaded from: {args.config}")

        # 3. Validate environment
        validate_environment(logger)

        # 4. Setup run directory
        run_dir, run_id = setup_run_directory(
            config,
            run_id=args.run_id,
            resume=args.resume,
            logger=logger
        )

        # Update config with final run directory
        config_manager.run_id = run_id
        config['outputs']['base_results_dir'] = str(run_dir)

        # 5. Initialize results layout manager
        excel_copy = config.get("outputs", {}).get("save_excel_copy", False)
        layout_manager = ResultsLayoutManager(run_dir, excel_copy=excel_copy, logger=logger)
        layout_manager.ensure_base_structure()

        # 6. Save configuration artifacts
        config_manager.save_artifacts(str(run_dir))

        # 7. Set global determinism
        setup_global_determinism(config, logger)

        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output Directory: {run_dir.absolute()}")

        # Log visualization settings
        viz_cfg = config.get("visualization", {})
        logger.info(
            f"Visualization settings - Advanced suite: {viz_cfg.get('run_advanced_suite', False)}, "
            f"Dashboard: {viz_cfg.get('run_dashboard', False)}, "
            f"Parallel: {viz_cfg.get('parallel_plots', False)}"
        )

        # Dry run mode: exit after validation
        if args.dry_run:
            logger.info("Dry run mode: validation complete. Exiting without running pipeline.")
            print("\n[SUCCESS] Configuration validated successfully.")
            return 0

        # ---------------------------------------------------------------
        # PHASE 1: DATA INGESTION & SPLITTING
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: DATA INGESTION & SPLITTING")
        logger.info("=" * 60)

        # Data loading and validation
        data_manager = DataManager(config, logger)
        validated_data = data_manager.execute(run_id)

        logger.info(f"Data loaded: {len(validated_data)} samples")

        # Stratified splitting
        split_engine = SplitEngine(config, logger)
        train_df, val_df, test_df = split_engine.execute(validated_data, run_id)

        logger.info(
            f"Master splits created - Train: {len(train_df)}, "
            f"Val: {len(val_df)}, Test: {len(test_df)}"
        )

        # ---------------------------------------------------------------
        # PHASE 2: RECURSIVE FEATURE ELIMINATION (RFE)
        # ---------------------------------------------------------------

        if args.skip_rfe:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 2: RFE SKIPPED (--skip-rfe flag set)")
            logger.info("=" * 60)
        elif not config.get('iterative', {}).get('enabled', True):
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 2: RFE DISABLED (config: iterative.enabled = false)")
            logger.info("=" * 60)
        else:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 2: CIRCULAR RECURSIVE FEATURE ELIMINATION")
            logger.info("=" * 60)

            # RFE Controller manages HPO, training, LOFO, and feature dropping
            rfe_controller = RFEController(config, logger)
            rfe_controller.run(train_df, val_df, test_df)

        # ---------------------------------------------------------------
        # PHASE 3: FINALIZATION & REPRODUCIBILITY
        # ---------------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: FINALIZATION & REPRODUCIBILITY")
        logger.info("=" * 60)

        # Generate reproducibility package
        repro_engine = ReproducibilityEngine(config, logger)
        pkg_path = repro_engine.execute(run_id)

        # Mirroring disabled to avoid duplicate folder trees

        # ---------------------------------------------------------------
        # COMPLETION
        # ---------------------------------------------------------------
        logger.info("\n" + "-" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output Directory: {run_dir.absolute()}")
        logger.info(f"Reproducibility Package: {pkg_path}")
        logger.info("-" * 60 + "\n")

        print(f"\n[SUCCESS] Pipeline completed. Results saved to: {run_dir}")

        return 0

    except RiserMLException as e:
        # Known pipeline errors
        msg = f"Pipeline Error: {str(e)}"
        print(f"\n[ERROR] {msg}")
        if logger:
            logger.critical(msg, exc_info=True)
        else:
            traceback.print_exc()
        return 1

    except KeyboardInterrupt:
        # User interruption
        print("\n[INTERRUPTED] Pipeline interrupted by user.")
        if logger:
            logger.warning("Pipeline interrupted by user (Ctrl+C)")
        return 130

    except Exception as e:
        # Unexpected errors
        msg = f"Unexpected Error: {str(e)}"
        print(f"\n[CRITICAL] {msg}")
        if logger:
            logger.critical(msg, exc_info=True)
        else:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
