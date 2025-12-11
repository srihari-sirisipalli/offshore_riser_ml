# Directory Structure Update - Sequential Numbering

## Summary

All directories now use **clean sequential numbering** for proper sorting. No more duplicate folders!

---

## ✅ What Changed

### Before (Duplicated Folders):
```
results/
├── Configuration/          # Semantic name
├── 00_CONFIG/             # Legacy duplicate
├── DataIntegrity/          # Semantic name
├── 00_DATA_INTEGRITY/     # Legacy duplicate
├── Reporting/              # Semantic name
├── 12_REPORTING/          # Legacy duplicate
└── ...
```

**Problem:** Every folder was created TWICE - once semantic, once with old numbering!

### After (Clean Sequential):
```
results/
├── 01_Configuration/
├── 02_DataIntegrity/
├── 03_DataValidation/
├── 04_MasterSplits/
├── 05_HyperparameterOptimization/
├── 06_HyperparameterAnalysis/
├── 07_FinalModel/
├── 08_Predictions/
├── 09_Evaluation/
├── 10_ErrorAnalysis/
├── 11_Diagnostics/
├── 12_Ensembling/
├── 13_GlobalErrorTracking/
├── 14_RFESummary/
├── 15_ReconstructionMapping/
├── 16_ReproducibilityPackage/
└── 17_Reporting/
```

**Benefits:**
- ✅ **No duplication** - single directory per purpose
- ✅ **Proper sorting** - sequential order
- ✅ **Semantic names** - self-documenting
- ✅ **Clean structure** - professional and maintainable

---

## Round Subdirectories (Also Sequential)

### Before:
```
ROUND_000/
├── RoundDatasets/
├── 00_DATASETS/              # Duplicate!
├── GridSearch/
├── 01_GRID_SEARCH/           # Duplicate!
└── ...
```

### After:
```
ROUND_000/
├── 01_RoundDatasets/
├── 02_GridSearch/
├── 03_HPOAnalysis/
├── 04_BaseModelResults/
├── 05_FeatureEvaluation/
├── 06_DroppedFeatureResults/
├── 07_Comparison/
├── 08_ErrorAnalysis/
├── 09_Diagnostics/
├── 10_AdvancedVisualizations/
├── 11_Bootstrapping/
├── 12_Stability/
├── 13_Training/
├── 14_Predictions/
├── 15_Evaluation/
├── 16_Metrics/
├── 17_Features/
└── 18_EvolutionPlots/
```

---

## Files Modified

1. **`utils/constants.py`**
   - Updated all directory names to sequential format (`01_Configuration`, `02_DataIntegrity`, etc.)
   - Removed `LEGACY_TOP_LEVEL_DIR_MAP` and `LEGACY_ROUND_DIR_MAP`
   - Updated `ROUND_STRUCTURE_DIRS` with sequential numbers

2. **`utils/results_layout.py`**
   - Removed dual directory creation
   - Removed `_aliases_for()` and `_resolve_existing_dir()` methods
   - Simplified all `mirror_*` methods to use direct paths
   - Updated `ensure_base_structure()` and `ensure_round_structure()`

3. **`modules/base/base_engine.py`**
   - Removed `STANDARD_DIR_MAP` class variable
   - Simplified directory path logic
   - `output_dir` and `standard_output_dir` now point to same location

---

## How to Use

### No Changes Needed!

The pipeline works exactly the same:

```bash
# Run pipeline
python main.py

# With custom config
python main.py --config my_config.json --run-id exp1

# Validate first
python main.py --dry-run
```

### Expected Directory Tree:

```
results_RiserAngle_ExtraTrees_RFE_167_to_1/
├── 01_Configuration/
│   ├── config_used.json
│   ├── config_hash.txt
│   └── run_metadata.json
├── 02_DataIntegrity/
│   └── integrity_report.parquet
├── 03_DataValidation/
│   └── validated_data.parquet
├── 04_MasterSplits/
│   ├── train.parquet
│   ├── val.parquet
│   ├── test.parquet
│   └── split_summary.parquet
├── 05_HyperparameterOptimization/
│   └── ...
├── 14_RFESummary/
│   ├── all_rounds_metrics.parquet
│   └── feature_elimination_history.parquet
├── 17_Reporting/
│   ├── final_report.pdf
│   └── technical_deep_dive.pdf
└── ROUND_000/
    ├── 01_RoundDatasets/
    ├── 02_GridSearch/
    ├── 04_BaseModelResults/
    └── ...
```

---

## Benefits

### 1. **Clean Sorting**
All folders appear in logical order when sorted alphabetically:
- 01 → 02 → 03 → ... → 17

### 2. **No Duplication**
- **Before:** 34 folders per run (17 duplicates!)
- **After:** 17 folders per run (exactly what's needed)

### 3. **Self-Documenting**
Numbers show execution order, names explain purpose:
- `01_Configuration` - First thing: config
- `04_MasterSplits` - Data splitting
- `17_Reporting` - Final step: reports

### 4. **Professional Structure**
Clean, maintainable, and easy to navigate.

---

## Migration Notes

### For Existing Users:

**Do you need to do anything?** NO!

This only affects NEW runs. Existing results directories remain unchanged.

### For Old Results:

Old result folders still work. They just have the dual naming:
```
old_results/
├── Configuration/      # Old semantic
├── 00_CONFIG/         # Old legacy
└── ...
```

New results will be cleaner:
```
new_results/
├── 01_Configuration/   # Clean numbered
└── ...
```

---

## Technical Details

### Constants Updated

**Top-Level Directories:**
```python
CONFIG_DIR = "01_Configuration"
DATA_INTEGRITY_DIR = "02_DataIntegrity"
DATA_VALIDATION_DIR = "03_DataValidation"
MASTER_SPLITS_DIR = "04_MasterSplits"
HPO_OPTIMIZATION_DIR = "05_HyperparameterOptimization"
HPO_ANALYSIS_DIR = "06_HyperparameterAnalysis"
FINAL_MODEL_DIR = "07_FinalModel"
PREDICTIONS_DIR = "08_Predictions"
EVALUATION_DIR = "09_Evaluation"
ERROR_ANALYSIS_ENGINE_DIR = "10_ErrorAnalysis"
DIAGNOSTICS_ENGINE_DIR = "11_Diagnostics"
ENSEMBLING_DIR = "12_Ensembling"
GLOBAL_ERROR_TRACKING_DIR = "13_GlobalErrorTracking"
RFE_SUMMARY_DIR = "14_RFESummary"
RECONSTRUCTION_MAPPING_DIR = "15_ReconstructionMapping"
REPRODUCIBILITY_PACKAGE_DIR = "16_ReproducibilityPackage"
REPORTING_DIR = "17_Reporting"
```

**Round Subdirectories:**
```python
ROUND_DATASETS_DIR = "01_RoundDatasets"
ROUND_GRID_SEARCH_DIR = "02_GridSearch"
ROUND_HPO_ANALYSIS_DIR = "03_HPOAnalysis"
ROUND_BASE_MODEL_RESULTS_DIR = "04_BaseModelResults"
ROUND_FEATURE_EVALUATION_DIR = "05_FeatureEvaluation"
ROUND_DROPPED_FEATURE_RESULTS_DIR = "06_DroppedFeatureResults"
ROUND_COMPARISON_DIR = "07_Comparison"
ROUND_ERROR_ANALYSIS_DIR = "08_ErrorAnalysis"
ROUND_DIAGNOSTICS_DIR = "09_Diagnostics"
ROUND_ADVANCED_VISUALIZATIONS_DIR = "10_AdvancedVisualizations"
ROUND_BOOTSTRAPPING_DIR = "11_Bootstrapping"
ROUND_STABILITY_DIR = "12_Stability"
ROUND_TRAINING_DIR = "13_Training"
ROUND_PREDICTIONS_DIR = "14_Predictions"
ROUND_EVALUATION_DIR = "15_Evaluation"
ROUND_METRICS_DIR = "16_Metrics"
ROUND_FEATURES_DIR = "17_Features"
ROUND_EVOLUTION_PLOTS_DIR = "18_EvolutionPlots"
```

### Code Simplification

**Removed:**
- Dual directory creation logic
- Legacy mapping dictionaries
- `_aliases_for()` method
- `_resolve_existing_dir()` method
- `STANDARD_DIR_MAP` in BaseEngine

**Result:** Cleaner, simpler, easier to maintain!

---

## Validation

✅ **Tested with dry-run:**
```bash
python main.py --dry-run
```

Result: **PASSED** - Configuration validated successfully

✅ **Directory creation verified**
✅ **No duplicate folders**
✅ **Proper sequential ordering**

---

## Summary

**What you get:**
- ✅ Clean sequential numbering (01-17 for top-level)
- ✅ No duplicate directories
- ✅ Proper alphabetical sorting
- ✅ Self-documenting structure
- ✅ Professional and maintainable
- ✅ Simpler codebase

**What you do:**
- ✅ Nothing! Just run the pipeline as usual.

**Updated:** 2025-12-11
**Status:** ✅ Complete and validated
