# Offshore Riser ML Pipeline - Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Command-Line Arguments](#command-line-arguments)
3. [Configuration Guide](#configuration-guide)
4. [Directory Structure](#directory-structure)
5. [Running the Pipeline](#running-the-pipeline)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Usage

Run the pipeline with default configuration:

```bash
python main.py
```

### With Custom Configuration

```bash
python main.py --config path/to/custom_config.json
```

### Dry Run (Validation Only)

Validate configuration without running the pipeline:

```bash
python main.py --dry-run
```

---

## Command-Line Arguments

### Required Arguments
None - all arguments have defaults.

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | string | `config/config.json` | Path to configuration JSON file |
| `--run-id` | string | `<timestamp>` | Custom run identifier |
| `--resume` | flag | `False` | Resume from existing run directory |
| `--skip-rfe` | flag | `False` | Skip RFE phase, run baseline only |
| `--verbose` | flag | `False` | Enable DEBUG-level logging |
| `--dry-run` | flag | `False` | Validate config without running |

### Examples

#### Custom Run ID
```bash
python main.py --run-id experiment_001
```

#### Resume Previous Run
```bash
python main.py --resume --run-id experiment_001
```

#### Verbose Logging
```bash
python main.py --verbose
```

#### Skip RFE (Baseline Only)
```bash
python main.py --skip-rfe
```

---

## Configuration Guide

### Configuration File Structure

The pipeline uses a hierarchical JSON configuration file with the following sections:

#### 1. Data Configuration
```json
{
  "data": {
    "file_path": "data.xlsx",
    "target_sin": "target_wave_direction_deg_sin",
    "target_cos": "target_wave_direction_deg_cos",
    "hs_column": "sea_elevation_significant_height_Hs_m",
    "include_hs_in_features": true,
    "drop_columns": [],
    "precision": "float32",
    "validate_sin_cos_circle": true,
    "circle_tolerance": 0.01
  }
}
```

**Key Parameters:**
- `file_path`: Input data file (Excel/CSV)
- `target_sin/target_cos`: Circular target columns
- `hs_column`: Wave height column name
- `include_hs_in_features`: Whether to use Hs as a feature
- `precision`: Data type for memory efficiency

#### 2. Data Quality Gates
```json
{
  "data_quality_gates": {
    "enabled": true,
    "min_split_rows": 10,
    "max_missing_pct": 5.0,
    "min_feature_variance": 0.0
  }
}
```

#### 3. Splitting Configuration
```json
{
  "splitting": {
    "angle_bins": 12,
    "hs_bins": 6,
    "test_size": 0.10,
    "val_size": 0.10,
    "seed": 123,
    "shuffle": true,
    "drop_incomplete_bins": false,
    "hs_binning_method": "quantile"
  }
}
```

**Key Parameters:**
- `angle_bins`: Number of angular stratification bins
- `hs_bins`: Number of wave height bins
- `test_size/val_size`: Fraction for test/validation sets
- `seed`: Random seed for reproducibility

#### 4. Model Selection
```json
{
  "models": {
    "native": ["ExtraTreesRegressor"],
    "wrapped": []
  }
}
```

#### 5. Hyperparameter Optimization (HPO)
```json
{
  "hyperparameters": {
    "enabled": true,
    "cv_strategy": "kfold_on_train",
    "cv_folds": 5,
    "max_trials": null,
    "grids": {
      "ExtraTreesRegressor": {
        "n_estimators": [100, 200],
        "max_depth": [null, 10, 20],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"]
      }
    }
  }
}
```

**Key Parameters:**
- `enabled`: Toggle HPO on/off
- `cv_folds`: Number of cross-validation folds
- `grids`: Parameter grid for each model

#### 6. Iterative RFE
```json
{
  "iterative": {
    "enabled": true,
    "max_rounds": 50,
    "min_features": 1,
    "run_hpo_analysis": true,
    "stopping_criteria": "combined"
  }
}
```

**Key Parameters:**
- `enabled`: Enable/disable RFE
- `max_rounds`: Maximum RFE iterations
- `min_features`: Stop when this many features remain
- `stopping_criteria`: When to stop RFE

#### 7. Execution Settings
```json
{
  "execution": {
    "n_jobs": -1,
    "deterministic": true,
    "max_hours": null,
    "parallel_hpo_configs": true
  }
}
```

**Key Parameters:**
- `n_jobs`: Number of parallel jobs (-1 = all cores)
- `deterministic`: Enable reproducibility
- `parallel_hpo_configs`: Parallelize HPO across configurations

#### 8. Evaluation Settings
```json
{
  "evaluation": {
    "bootstrap_samples": 0,
    "bootstrap_alpha": 0.05,
    "accuracy_bands": [0, 5, 10, 20],
    "compute_naive_baseline": true,
    "industry_baseline_cmae": null
  }
}
```

#### 9. Error Analysis
```json
{
  "error_analysis": {
    "enabled": true,
    "threshold_bins": [0, 5, 10, 20],
    "safety_gates": {
      "enabled": true,
      "max_critical_pct": 25.0,
      "max_warning_pct": 50.0
    }
  }
}
```

#### 10. Resource Limits
```json
{
  "resources": {
    "max_hpo_configs": 1000,
    "max_file_size_mb": 1000,
    "max_memory_mb": 8192,
    "operation_timeout_sec": 3600
  }
}
```

**Safety Guardrails:**
- `max_hpo_configs`: Prevent grid explosion
- `max_memory_mb`: Memory limit (auto-detected)
- `operation_timeout_sec`: Per-operation timeout

---

## Directory Structure

### Semantic Output Structure

The pipeline creates a structured output directory with semantic names (no numeric prefixes):

```
results_<run_name>/
├── Configuration/
│   ├── config_used.json
│   ├── config_hash.txt
│   └── run_metadata.json
├── DataIntegrity/
│   └── integrity_report.parquet
├── DataValidation/
│   └── validation_summary.parquet
├── MasterSplits/
│   ├── train.parquet
│   ├── val.parquet
│   ├── test.parquet
│   └── split_summary.parquet
├── ROUND_000/
│   ├── RoundDatasets/
│   ├── GridSearch/
│   ├── HPOAnalysis/
│   ├── BaseModelResults/
│   ├── FeatureEvaluation/
│   ├── DroppedFeatureResults/
│   ├── Comparison/
│   ├── ErrorAnalysis/
│   ├── Diagnostics/
│   └── AdvancedVisualizations/
├── ROUND_001/
│   └── ...
├── RFESummary/
│   ├── all_rounds_metrics.parquet
│   └── feature_elimination_history.parquet
├── ReproducibilityPackage/
│   ├── reproducibility_manifest.json
│   └── environment.yml
└── Reporting/
    ├── final_report.pdf
    └── technical_deep_dive.pdf
```

### Legacy Compatibility

The pipeline maintains backward compatibility with numeric folder prefixes through automatic aliasing.

---

## Running the Pipeline

### Standard Workflow

1. **Prepare Configuration**
   ```bash
   cp config/config_template.json config/my_experiment.json
   # Edit my_experiment.json as needed
   ```

2. **Validate Configuration**
   ```bash
   python main.py --config config/my_experiment.json --dry-run
   ```

3. **Run Pipeline**
   ```bash
   python main.py --config config/my_experiment.json --run-id exp_v1
   ```

4. **Review Results**
   - Check `results_exp_v1/Reporting/final_report.pdf`
   - Explore `results_exp_v1/RFESummary/` for feature evolution
   - Examine `results_exp_v1/ROUND_XXX/` for detailed round analysis

---

## Advanced Features

### 1. Parallel HPO Across Configurations

Enable in config for large grids:

```json
{
  "execution": {
    "parallel_hpo_configs": true,
    "n_jobs": -1
  }
}
```

**Benefits:**
- Faster HPO for large parameter grids
- Better CPU utilization
- Automatic CV fold serialization to avoid over-parallelization

### 2. Resume Capability

Resume interrupted runs:

```bash
python main.py --resume
```

The pipeline automatically:
- Detects completed rounds
- Skips finished configurations
- Resumes from the last incomplete round

### 3. Reproducibility Package

Automatically generated at the end of each run:

- Environment snapshot (packages, versions)
- Configuration hash
- Data checksums
- Model reconstruction mapping

### 4. Safety Gates

Configure safety thresholds in error analysis:

```json
{
  "error_analysis": {
    "safety_gates": {
      "enabled": true,
      "max_critical_pct": 25.0,
      "max_warning_pct": 50.0
    }
  }
}
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors

**Solution:**
- Reduce `execution.n_jobs`
- Disable `execution.parallel_hpo_configs`
- Reduce `hyperparameters.grids` complexity
- Decrease `resources.max_memory_mb`

#### 2. Configuration Validation Errors

**Solution:**
- Run `--dry-run` to validate before execution
- Check JSON syntax
- Ensure all required fields are present
- Verify data file paths exist

#### 3. HPO Grid Explosion

**Solution:**
- Reduce parameter grid size
- Increase `resources.max_hpo_configs` if intentional
- Use focused parameter ranges

#### 4. Slow Performance

**Solution:**
- Enable `execution.parallel_hpo_configs`
- Increase `execution.n_jobs`
- Reduce `hyperparameters.cv_folds`
- Disable expensive visualizations temporarily

### Logging

**View Logs:**
```bash
tail -f logs/pipeline.log
```

**Increase Verbosity:**
```bash
python main.py --verbose
```

---

## Best Practices

1. **Start Small**: Test with a small parameter grid before scaling up
2. **Use Dry Run**: Always validate configuration with `--dry-run` first
3. **Name Your Runs**: Use meaningful `--run-id` values
4. **Monitor Resources**: Check CPU/memory usage during large runs
5. **Save Results**: Keep reproducibility packages for important experiments
6. **Document Changes**: Track configuration changes across experiments

---

## Getting Help

- Configuration issues: Check `config/config_template.json` for reference
- Pipeline errors: Review `logs/pipeline.log`
- Feature requests: Submit to project repository
- Bug reports: Include run_id and error logs

---

**Last Updated:** 2025-12-11
**Pipeline Version:** 2.0 (Production)
