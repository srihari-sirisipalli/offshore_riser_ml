# Offshore Riser ML Pipeline - Circular RFE

## Overview

A production-grade machine learning pipeline for predicting offshore riser angles using circular regression techniques. The system implements a comprehensive **Circular Recursive Feature Elimination (RFE)** workflow with automated hyperparameter optimization, advanced diagnostics, and robust data integrity tracking.

## Key Features

### Core Pipeline
- **Circular Recursive Feature Elimination (RFE):** Iterative feature selection using Leave-One-Feature-Out (LOFO) evaluation with circular angle metrics
- **Automated Hyperparameter Optimization:** Grid search with stratified K-Fold cross-validation and automatic resumption
- **Circular Angle Handling:** Specialized metrics (Circular MAE, RMSE) and sin/cos transformations for angle predictions
- **Data Integrity Tracking:** Comprehensive lineage tracking across splits and RFE rounds with checksum validation
- **Smart Stratified Splitting:** Multi-level stratification (combined_bin, hs_bin, angle_bin) with rare bin handling

### Analysis & Diagnostics
- **Advanced Visualizations:** 3D error surfaces, optimal performance zones, circular error plots, boundary analysis
- **Error Analysis:** Statistical outlier detection, prediction explanations via correlation analysis, safety gate validation
- **Model Comparison:** Detailed baseline vs. reduced model comparisons across all metrics
- **Bootstrapping:** Confidence interval estimation for all performance metrics
- **Stability Analysis:** Multi-seed validation for model robustness assessment

### Production Features
- **Parallel Execution:** Multi-threaded HPO, LOFO evaluation, and visualization generation
- **Caching System:** Smart caching for splits, HPO configurations, and predictions
- **Resource Monitoring:** CPU, memory, and disk usage tracking throughout the pipeline
- **Reproducibility:** Complete artifact packages with pinned dependencies and environment snapshots
- **Comprehensive Logging:** Structured logging with progress indicators and detailed error traces

## Project Structure

```
offshore_riser_ml/
├── config/
│   ├── config.json           # Main configuration file
│   └── config_template.json  # Template with all options
├── data/
│   └── raw/                  # Input data files
├── modules/
│   ├── bootstrapping_engine/
│   ├── config_manager/
│   ├── data_integrity/       # Data lineage and quality tracking
│   ├── data_manager/
│   ├── diagnostics_engine/
│   ├── ensembling_engine/
│   ├── error_analysis_engine/
│   ├── evaluation_engine/
│   ├── global_error_tracking/ # Cross-round metric evolution
│   ├── hpo_search_engine/
│   ├── hyperparameter_analyzer/
│   ├── logging_config/
│   ├── model_factory/
│   ├── prediction_engine/
│   ├── reporting_engine/
│   ├── reproducibility_engine/
│   ├── rfe/                  # Circular RFE controller and feature evaluation
│   ├── split_engine/
│   ├── stability_engine/
│   ├── training_engine/
│   └── visualization/        # Advanced viz and interactive dashboards
├── utils/
│   ├── cache.py
│   ├── constants.py
│   ├── error_handling.py
│   ├── exceptions.py
│   ├── file_io.py
│   ├── resource_monitor.py
│   └── results_layout.py
├── tests/                    # Comprehensive test suite
├── main.py                   # Pipeline entry point
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd offshore_riser_ml
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Pipeline

Execute the complete pipeline from the command line:

```bash
python main.py
```

The pipeline executes the following phases:

1. **Data Ingestion & Splitting:** Load data, validate quality, create stratified train/val/test splits
2. **Circular RFE Rounds:** For each round:
   - Hyperparameter optimization with grid search
   - Baseline model training and evaluation
   - LOFO feature evaluation (parallel execution)
   - Feature elimination based on test set performance
   - Comprehensive diagnostics and visualizations
   - Round comparison and evolution tracking
3. **Final Analysis:** Global tracking, ensembling, reproducibility packaging

### Configuration

Edit `config/config.json` to customize pipeline behavior:

```json
{
  "data": {
    "file_path": "data/raw/your_data.xlsx",
    "target_column": "RiserAngle",
    "features_to_drop": ["timestamp", "id"]
  },
  "splitting": {
    "test_size": 0.1,
    "val_size": 0.1,
    "seed": 123,
    "cache_enabled": true
  },
  "rfe": {
    "enabled": true,
    "min_features": 1,
    "strategy": "lofo_test",
    "stopping_criteria": {
      "patience_rounds": 5,
      "min_improvement_pct": 0.5
    }
  },
  "hyperparameters": {
    "enable_hpo": true,
    "parallel_configs": false,
    "models": ["ExtraTreesRegressor"]
  },
  "execution": {
    "n_jobs": -1
  },
  "visualization": {
    "run_advanced_suite": true,
    "run_dashboard": false,
    "parallel_plots": true
  }
}
```

See `config/config_template.json` for all available options.

## Output Structure

Results are organized in a standardized directory structure under `results_<target>_<model>_RFE_<features>/`:

```
results_RiserAngle_ExtraTrees_RFE_167_to_1/
├── 00_CONFIG/                      # Run configuration and metadata
├── 02_DataQualityChecks/           # Data integrity tracking
├── 03_MasterDataSplits/            # Train/val/test splits
├── ROUND_000/                      # First RFE round (all features)
│   ├── 01_RoundDatasets/
│   ├── 02_Hyperparameter_GridSearch/
│   ├── 03_Hyperparameter_Analysis/
│   ├── 04_BaseModel_WithAllFeatures/
│   │   ├── ErrorAnalysis/
│   │   ├── DiagnosticPlots/
│   │   └── 10_DiagnosticPlots_Advanced/
│   ├── 05_FeatureEvaluation_LOFO/
│   ├── 06_ReducedModel_FeatureDropped/
│   ├── 07_ModelComparison_BaseVsReduced/
│   ├── 08_Bootstrapping/
│   └── 10_StabilityAnalysis/
├── ROUND_001/                      # Second RFE round (one feature dropped)
│   └── ... (same structure)
├── ROUND_NNN/                      # Subsequent rounds
├── 01_GlobalTracking/              # Cross-round evolution metrics
├── 96_ReproducibilityPackage/      # Complete reproducibility bundle
└── 99_RFE_Summary/                 # Final RFE analysis
```

All data artifacts are saved in Parquet format with optional Excel exports enabled via `outputs.save_excel_copy`.

## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=modules --cov-report=html

# Run specific test module
pytest tests/rfe/test_rfe_controller.py

# Run with verbose output
pytest -v
```

## Performance Tuning

### Parallel Execution
- **HPO Grid Search:** Set `hyperparameters.parallel_configs: true` for parallel configuration evaluation (high memory usage)
- **LOFO Evaluation:** Automatically parallelized, respects `execution.n_jobs`
- **Diagnostics Plots:** Set `diagnostics.parallel_plots: true` (default)
- **Advanced Visualizations:** Set `visualization.parallel_plots: true`
- **Global Jobs Setting:** `execution.n_jobs: -1` uses all CPU cores

### Caching
- **Split Caching:** `splitting.cache_enabled: true` caches train/val/test splits based on data signature
- **HPO Caching:** Automatically resumes interrupted HPO runs
- **Cache Directory:** `.cache/` (configurable)

### Resource Management
- Monitor resource usage in `00_DataQualityChecks/resource_utilization_dashboard.png`
- Adjust `execution.n_jobs` based on available memory
- Enable/disable advanced visualizations and dashboard generation to reduce processing time

## Advanced Features

### Circular RFE Strategy
The pipeline supports multiple RFE strategies via `rfe.strategy`:
- `lofo_test`: Drop feature with best test performance when removed (default)
- `lofo_val`: Drop feature with best validation performance when removed
- `importance`: Drop feature with lowest model importance (not yet implemented)

### Stopping Criteria
RFE automatically stops when:
- Minimum features reached (`rfe.min_features`)
- No improvement for N rounds (`rfe.stopping_criteria.patience_rounds`)
- Improvement below threshold (`rfe.stopping_criteria.min_improvement_pct`)

### Advanced Visualizations
Enable comprehensive visualization suite:
- 3D error surface plots
- Optimal performance zone maps
- Circular error vs. angle plots
- Boundary gradient analysis
- Faceted error distributions
- Operating envelope overlays

Set `visualization.run_advanced_suite: true` in config.

### Interactive Dashboard
Generate interactive Plotly dashboards:
```python
from modules.visualization.interactive_dashboard import build_dashboard
import pandas as pd
from pathlib import Path

predictions = pd.read_parquet("results/.../predictions_test.parquet")
build_dashboard(predictions, Path("dashboard.html"), hs_col="Hs_ft")
```

## Reproducibility

The pipeline generates a complete reproducibility package in `96_ReproducibilityPackage/`:
- Frozen dependency list (`requirements_frozen.txt`)
- Complete configuration snapshot
- System information and environment details
- Data checksums and split signatures
- Model artifacts and predictions
- Deployment readiness checklist

## Troubleshooting

### Common Issues

**Issue:** `ValueError: The truth value of a Series is ambiguous`
- **Fix:** Ensure you're using the latest version with the fix in `error_analysis_engine.py`

**Issue:** Thousands of duplicate index warnings
- **Fix:** Clear split cache with `rm -rf .cache/splits/*` and rerun

**Issue:** Out of memory during parallel HPO
- **Fix:** Set `hyperparameters.parallel_configs: false` or reduce `execution.n_jobs`

**Issue:** Missing plotly for dashboard
- **Fix:** `pip install plotly` or disable dashboard with `visualization.run_dashboard: false`

### Logging
- Console output: Real-time progress and status
- File logs: `logs/pipeline.log` with detailed timestamps and stack traces
- Log level: Configurable in `logging_config/logging_config.py`

## Contributing

When contributing to this project:
1. Run tests before submitting: `pytest`
2. Follow the existing code structure and naming conventions
3. Update relevant documentation
4. Add tests for new features

## License

[Add your license information here]

## Citation

If you use this pipeline in your research, please cite:
[Add citation information here]

## Recent Fixes

### Version 1.1.0
- Fixed error analysis DataFrame/Series ambiguity issue
- Fixed duplicate index warnings in data integrity tracking
- Improved split caching to preserve original indices
- Enhanced error handling in visualization modules
