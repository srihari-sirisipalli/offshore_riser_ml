# Offshore Riser ML Prediction System

## Overview

This project is a comprehensive Machine Learning pipeline designed to predict the heading of offshore risers based on gyro data. It provides an end-to-end workflow including data validation, hyperparameter optimization, model training, evaluation, and reporting. The system is built with a modular architecture, allowing for easy extension and maintenance of individual pipeline phases.

## Features

- **End-to-End ML Pipeline:** A complete workflow from raw data to final model and reports.
- **Robust Data Validation:** Ensures data quality by checking for missing columns, NaN/Inf values, and circular constraints (sin/cos). Includes path traversal safety checks to ensure data is loaded only from designated directories.
- **Smart Data Splitting:** Stratified sampling to ensure balanced train, validation, and test sets, with improved handling of rare data bins (either dropping them or moving them to the training set).
- **Hyperparameter Optimization (HPO):** Automated grid search with K-Fold cross-validation to find the best model parameters.
- **Model Training & Prediction:** Robust training and prediction workflows that validate feature consistency and handle potential save errors gracefully.
- **Comprehensive Evaluation:** Computes a suite of metrics, including circular MAE/RMSE and accuracy bands.
- **Advanced Diagnostics:** Generates detailed diagnostic plots for error analysis. Plotting styles are now managed to prevent leaking into other modules.
- **Stability Analysis:** Assesses model and metric stability by running the pipeline multiple times with different random seeds, with improved isolation between runs.
- **Reproducibility:** Creates a self-contained reproducibility package with all necessary artifacts, pinned dependencies, configurations, and environment details.
- **Automated Reporting:** Generates PDF reports for executive summaries and technical deep-dives, with improved handling of potential image loading errors.
- **Informative Logging:** Provides clear progress indicators for long-running operations and improved UTF-8 support on Windows.

## Prerequisites

- Python 3.8+
- `pip` for installing packages

## Installation

1.  Clone the repository to your local machine.
2.  Navigate to the project root directory.
3.  Install the required Python packages using the `requirements.txt` file. Using a virtual environment is highly recommended.

    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install dependencies
    pip install -r requirements.txt
    ```

## Usage

The entire pipeline can be run from the command line using the `main.py` script.

```bash
python main.py
```

The pipeline will execute all configured phases in sequence, starting from data loading and ending with reporting and reproducibility packaging. Progress and logs will be displayed on the console and saved to the `logs/` directory.

## Running Tests

The test suite can be run using `pytest`. From the project root directory, run:

```bash
pytest
```

This will discover and run all tests in the `tests/` directory, ensuring the integrity and correctness of the individual pipeline modules.

## Configuration

The pipeline's behavior is controlled by the `config/config.json` file. This file allows you to enable/disable specific phases, define file paths, set model parameters, and configure the HPO search.

The structure and allowed values for the configuration are defined in `config/schema.json`. This schema is the single source of truth for default values.

### Key Configuration Sections:

-   `data`: Specifies the input data file path, target columns, and features to drop.
-   `splitting`: Controls the train/validation/test split ratios, stratification bins, and the master random seed.
-   `hyperparameters`: Enables/disables HPO and defines the grid search space for different models.
-   `stability`: Enables/disables stability analysis and configures the number of runs.
-   `outputs`: Defines the base directory for all output artifacts.

## Output Structure

All outputs from the pipeline are saved to the `results/` directory (or the directory specified in `base_results_dir` in the config). The output is organized into subdirectories corresponding to each phase of the pipeline, following a consistent numbering scheme:

```
results/
├── 00_CONFIG/                  # Configuration artifacts for the run (Phase 1)
├── 01_DATA_VALIDATION/         # Validated data and reports (Phase 2)
├── 02_SMART_SPLIT/             # Train, validation, and test set files (Phase 2)
├── 03_HYPERPARAMETER_OPTIMIZATION/ # HPO results and progress (Phase 3)
├── 03_HPO_SEARCH/              # HPO tracking snapshots (Phase 3)
├── 04_GLOBAL_FAILURE_TRACKING/ # Analysis of persistent failures across HPO trials (Phase 4)
├── 05_HYPERPARAMETER_ANALYSIS/ # Visualizations and reports for HPO analysis (Phase 5)
├── 06_FINAL_MODEL/             # The final trained model object (.pkl) and metadata (Phase 6)
├── 07_PREDICTIONS/             # Predictions for val and test sets (Phase 7)
├── 08_EVALUATION/              # Performance metrics (Phase 8)
├── 09_DIAGNOSTICS/             # Diagnostic plots (Phase 9)
├── 10_ERROR_ANALYSIS/          # Detailed error analysis reports (Phase 10)
├── 11_ADVANCED_ANALYTICS/      # Stability, Ensembling, and Bootstrapping results (Phase 11)
├── 12_REPORTING/               # Final PDF reports (Phase 12)
└── 13_REPRODUCIBILITY_PACKAGE/ # A self-contained package for the run (Phase 13)
```

---
This README provides a basic guide to the project. For more detailed information on the methodology and results of a specific run, please refer to the PDF reports and artifacts generated in the `results/` directory.
