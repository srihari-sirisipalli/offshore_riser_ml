# Results Folder Organization Guide

## Naming Convention

All results folders use **self-explanatory, descriptive names WITHOUT timestamps** for clarity and reproducibility.

**Parquet-first outputs:** All core artifacts are saved as `.parquet`. Optional Excel sidecars (`.xlsx`) are only written when `outputs.save_excel_copy` is `True` (default: `False`). Filenames below use the Parquet extension; if Excel copies are enabled they share the same base name.

### Root Results Folder Format

```
results_<TargetVariable>_<ModelType>_<Strategy>_<FeatureRange>/
```

**Example:**
```
results_RiserAngle_ExtraTrees_RFE_167_to_1/
```

**Components:**
- `TargetVariable`: What is being predicted (e.g., RiserAngle)
- `ModelType`: ML algorithm used (e.g., ExtraTrees, RandomForest, GradientBoosting)
- `Strategy`: Pipeline type (e.g., RFE, HPO, GridSearch, FullPipeline)
- `FeatureRange`: Feature count progression (e.g., 167_to_1, 100_to_10)

---

## Directory Structure (Complete)

```
results_RiserAngle_ExtraTrees_RFE_167_to_1/
│
├── 00_CONFIG/                              # Configuration & Metadata
│   ├── config_used.json                    # Full configuration snapshot
│   ├── config_hash.txt                     # SHA256 hash for uniqueness
│   ├── run_metadata.json                   # Execution metadata
│   └── schema.json                         # Configuration schema
│
├── 01_DATA_VALIDATION/                     # Data Quality Checks
│   ├── validated_data.parquet              # Validated dataset (optional Excel sidecar)
│   ├── column_stats.parquet                # Statistical summary (optional Excel sidecar)
│   ├── sin_cos_validation.parquet          # Circle constraint check (optional Excel sidecar)
│   ├── angle_distribution.png              # Angle histogram
│   ├── hs_distribution.png                 # Wave height histogram
│   └── data_quality_report.txt             # Quality summary
│
├── 01_GLOBAL_TRACKING/                     # Cross-Round Evolution
│   ├── 01_metrics/
│   │   ├── global_evolution_metrics.parquet   # All rounds metrics (optional Excel sidecar)
│   │   └── cumulative_dropped_features.parquet   # Optional Excel sidecar
│   ├── 02_features/
│   │   ├── feature_importance_evolution.parquet   # Optional Excel sidecar
│   │   └── feature_retention_timeline.csv
│   └── 04_evolution_plots/
│       ├── metric_evolution.png
│       ├── feature_count_progression.png
│       └── performance_vs_complexity.png
│
├── 02_SMART_SPLIT/                         # Train/Val/Test Splits
│   ├── train.parquet                       # Training set (70%) (optional Excel sidecar)
│   ├── val.parquet                         # Validation set (15%) (optional Excel sidecar)
│   ├── test.parquet                        # Test set (15%) (optional Excel sidecar)
│   ├── split_balance_report.parquet        # Stratification quality (optional Excel sidecar)
│   ├── split_hs_dist.png                   # Hs distribution comparison
│   └── split_angle_dist.png                # Angle distribution comparison
│
├── 13_REPRODUCIBILITY_PACKAGE/             # Self-Contained Package
│   ├── README.md                           # Reproduction instructions
│   ├── config.json                         # Configuration used
│   ├── requirements_frozen.txt             # Exact dependencies
│   ├── system_info.json                    # Hardware/OS details
│   ├── run_metadata.json                   # Execution info
│   └── diagnostics/                        # System diagnostics
│
├── ROUND_000/                              # First RFE Round (167 features)
├── ROUND_001/                              # Second RFE Round (166 features)
├── ROUND_002/                              # Third RFE Round (165 features)
│   ...                                     # (continues until stopping criteria)
└── ROUND_NNN/                              # Final RFE Round (N features)
```

---

## Per-Round Structure (Detailed)

Each `ROUND_XXX/` directory contains:

```
ROUND_000/
│
├── 00_DATASETS/                            # Feature Lists & Data
│   ├── feature_list.json                   # Active features (JSON format)
│   ├── feature_list_checksum.txt           # SHA256 checksum
│   └── round_summary.txt                   # Human-readable summary
│
├── 01_GRID_SEARCH/                         # Hyperparameter Optimization
│   ├── 01_INITIAL_TRIALS/
│   │   ├── trial_results.parquet              # Optional Excel sidecar
│   │   └── trial_predictions/
│   ├── 02_DETAILED_SEARCH/
│   │   └── search_results.parquet             # Optional Excel sidecar
│   └── 03_HYPERPARAMETER_OPTIMIZATION/
│       ├── results/
│       │   ├── all_configurations.parquet      # All HPO trials (optional Excel sidecar)
│       │   ├── best_configuration.json      # Optimal parameters
│       │   └── optimization_summary.parquet    # Optional Excel sidecar
│       ├── predictions/
│       │   └── config_XXXX_predictions.parquet # Optional Excel sidecar
│       └── visualizations/
│           ├── param_importance.png
│           └── optimization_history.png
│
├── 02_HYPERPARAMETER_ANALYSIS/             # HPO Analysis
│   ├── parameter_importance.parquet           # Optional Excel sidecar
│   ├── interaction_matrix.parquet             # Optional Excel sidecar
│   ├── sensitivity_plots/
│   │   ├── param_A_response.png
│   │   └── param_B_response.png
│   └── correlation_heatmap.png
│
├── 03_BASE_MODEL_RESULTS/                  # Baseline Model (All Features)
│   ├── baseline_predictions_val.parquet      # Optional Excel sidecar
│   ├── baseline_predictions_test.parquet     # Optional Excel sidecar
│   ├── baseline_metrics_val.parquet          # Optional Excel sidecar
│   ├── baseline_metrics_test.parquet         # Optional Excel sidecar
│   ├── error_analysis_test/
│   │   ├── error_by_angle_bin.parquet        # Optional Excel sidecar
│   │   ├── error_by_hs_bin.parquet           # Optional Excel sidecar
│   │   ├── error_statistics.parquet          # Optional Excel sidecar
│   │   └── worst_predictions.parquet         # Optional Excel sidecar
│   └── diagnostics/
│       ├── residual_plots.png
│       ├── qq_plot.png
│       ├── error_distribution.png
│       └── prediction_scatter.png
│
├── 04_FEATURE_EVALUATION/                  # LOFO Analysis
│   ├── lofo_summary.parquet                   # Complete LOFO results (optional Excel sidecar)
│   └── feature_evaluation_plots/
│       ├── lofo_impact_barplot.png
│       ├── lofo_delta_cmae.png
│       └── lofo_error_heatmap.png
│
├── 05_DROPPED_FEATURE_RESULTS/             # Model Without Selected Feature
│   ├── dropped_predictions_val.parquet       # Optional Excel sidecar
│   ├── dropped_predictions_test.parquet      # Optional Excel sidecar
│   ├── error_analysis/
│   │   ├── error_patterns.parquet            # Optional Excel sidecar
│   │   └── error_distribution.png
│   └── diagnostics/
│       ├── residual_plots.png
│       └── prediction_scatter.png
│
├── 06_COMPARISON/                          # Baseline vs Dropped Comparison
│   ├── comparison_summary.parquet             # Metric differences (optional Excel sidecar)
│   └── comparison_plots/
│       ├── metric_comparison_barplot.png
│       ├── prediction_scatter_comparison.png
│       ├── error_distribution_comparison.png
│       └── residual_comparison.png
│
└── _ROUND_COMPLETE.flag                    # Round Completion Marker
```

---

## File Naming Conventions

### 1. Sequenced Directories

All directories use **zero-padded numbers** for proper alphabetical sorting:

```
00_CONFIG              # First
01_DATA_VALIDATION     # Second
02_SMART_SPLIT         # Third
...
13_REPRODUCIBILITY_PACKAGE  # Last
```

**Benefits:**
- Alphabetical order = execution order
- Easy to navigate in file explorer
- Clear progression

### 2. Round Directories

```
ROUND_000   # First round (all features)
ROUND_001   # Second round (one feature dropped)
ROUND_002   # Third round (two features dropped)
...
ROUND_167   # Final round (only 1 feature remaining)
```

**Format:** `ROUND_XXX` where XXX is zero-padded to 3 digits

### 3. Data Files

**Excel Files:**
```
<descriptive_name>_<split>.parquet  # Optional Excel sidecar when enabled

Examples:
- baseline_predictions_val.parquet (optional Excel sidecar)
- baseline_predictions_test.parquet (optional Excel sidecar)
- lofo_summary.parquet (optional Excel sidecar)
- comparison_summary.parquet (optional Excel sidecar)
```

**JSON Files:**
```
<descriptive_name>.json

Examples:
- config_used.json
- feature_list.json
- best_configuration.json
- run_metadata.json
```

**Plot Files:**
```
<descriptive_name>.png

Examples:
- metric_evolution.png
- lofo_impact_barplot.png
- error_distribution.png
```

---

## Phase Numbering System

```
00_CONFIG                   # Phase 0: Configuration
01_DATA_VALIDATION          # Phase 1: Data Quality
01_GLOBAL_TRACKING          # Phase 1: Cross-round tracking (parallel)
02_SMART_SPLIT              # Phase 2: Data splitting
03-12_ENGINE_OUTPUTS        # Phases 3-12: Various engines
13_REPRODUCIBILITY          # Phase 13: Final package
```

**Why start at 00?**
- 00 = Pre-execution (configuration)
- 01+ = Execution phases
- Makes sorting clearer

---

## Special Files

### Completion Markers

**`_ROUND_COMPLETE.flag`**
- JSON file marking round completion
- Contains:
  - Timestamp
  - Round number
  - Dropped feature
  - Features remaining
  - Stopping status
  - Next round checksum

**Format:**
```json
{
  "timestamp": "2025-12-10T14:23:45",
  "round": 0,
  "dropped_feature": "feature_X",
  "features_remaining": 166,
  "stopping_status": "CONTINUE",
  "next_round_feature_checksum": "abc123...",
  "next_round_feature_count": 166
}
```

### Checksum Files

**`feature_list_checksum.txt`**
- SHA256 hash of feature list
- One line containing hex digest
- Validates feature list integrity

**`config_hash.txt`**
- SHA256 hash of configuration
- Ensures unique run identification
- Prevents configuration mix-ups

---

## Size Estimates

### Typical Run (167→1 features, 100K rows)

| Component | Size | Notes |
|-----------|------|-------|
| **Configuration** | 1-5 MB | JSON files, metadata |
| **Data Splits** | 50-100 MB | Excel format (train/val/test) |
| **Per Round (avg)** | 200-500 MB | HPO, predictions, plots |
| **8 Rounds Total** | 1.6-4 GB | Full RFE execution |
| **Global Tracking** | 10-20 MB | Evolution metrics |
| **Reproducibility** | 5-10 MB | Package files |
| **TOTAL** | **~2-4.5 GB** | Complete pipeline run |

### Large Runs (1M rows, 1000 features)

| Component | Size | Notes |
|-----------|------|-------|
| **Data Splits** | 500 MB - 1 GB | Large datasets |
| **Per Round** | 1-2 GB | More predictions |
| **50+ Rounds** | 50-100 GB | Full feature elimination |
| **TOTAL** | **~50-100 GB** | Enterprise-scale run |

**Recommendation:** Use Parquet format for >100K rows (5-10x smaller)

---

## File Organization Best Practices

### 1. Descriptive Over Cryptic

❌ BAD:
```
results_20251210_143022/
  r0/
    hp/
      cfg_001.parquet  # Optional Excel sidecar
```

✅ GOOD:
```
results_RiserAngle_ExtraTrees_RFE_167_to_1/
  ROUND_000/
    01_GRID_SEARCH/
      all_configurations.parquet  # Optional Excel sidecar
```

### 2. Sequential Numbering

All numbered directories use the format: `XX_DESCRIPTION`

- Always 2 digits minimum
- Zero-padded for sorting
- Description in UPPERCASE or Title Case

### 3. Consistent Naming

**Splits:**
- `train.parquet` (optional Excel sidecar)
- `val.parquet` (optional Excel sidecar)
- `test.parquet` (optional Excel sidecar)

**Metrics:**
- `metrics_val.parquet` (optional Excel sidecar)
- `metrics_test.parquet` (optional Excel sidecar)

**Predictions:**
- `predictions_val.parquet` (optional Excel sidecar)
- `predictions_test.parquet` (optional Excel sidecar)

### 4. Metadata Everywhere

Every major directory should contain:
- `README.txt` or `README.md` - Description
- `metadata.json` - Structured info
- `summary.parquet` (optional Excel sidecar) or `summary.txt` - Human-readable summary

---

## Navigation Tips

### Finding Specific Information

**"What were the best hyperparameters for Round 5?"**
```
→ results_*/ROUND_005/01_GRID_SEARCH/03_HYPERPARAMETER_OPTIMIZATION/results/best_configuration.json
```

**"What feature was dropped in Round 10?"**
```
→ results_*/ROUND_010/_ROUND_COMPLETE.flag
  (look for "dropped_feature" field)
```

**"How did performance evolve across rounds?"**
```
→ results_*/01_GLOBAL_TRACKING/01_metrics/global_evolution_metrics.parquet
```

**"What was the final model configuration?"**
```
→ results_*/ROUND_NNN/01_GRID_SEARCH/.../best_configuration.json
  (where NNN is the last round)
```

### Using Command Line

**List all rounds:**
```bash
ls -d results_*/ROUND_*
```

**Find all completion flags:**
```bash
find results_* -name "_ROUND_COMPLETE.flag"
```

**Check which features were dropped:**
```bash
grep "dropped_feature" results_*/ROUND_*/_ROUND_COMPLETE.flag
```

**Find all LOFO summaries:**
```bash
find results_* -name "lofo_summary.parquet"
```

---

## Maintenance & Cleanup

### Archive Old Runs

```bash
# Compress runs older than 30 days
find results_* -type d -mtime +30 -exec tar -czf {}.tar.gz {} \;

# Remove original directories after archiving
find results_* -type d -mtime +30 -exec rm -rf {} \;
```

### Selective Cleanup

**Remove large intermediate files:**
```bash
# Remove individual prediction files (keep summaries)
find results_* -path "*/predictions/config_*.parquet" -delete

# Remove individual plots (keep summary plots)
find results_* -path "*/sensitivity_plots/*.png" -delete
```

**Keep only essentials:**
- Configuration files
- Final model
- Metric summaries
- Completion flags
- Feature lists

---

## Verification Scripts

### Check Completeness

```python
import json
from pathlib import Path

def verify_run(results_dir):
    """Verify all expected files exist."""
    results_dir = Path(results_dir)

    # Check required top-level directories
    required_dirs = [
        '00_CONFIG',
        '01_DATA_VALIDATION',
        '02_SMART_SPLIT',
        '13_REPRODUCIBILITY_PACKAGE'
    ]

    for dir_name in required_dirs:
        if not (results_dir / dir_name).exists():
            print(f"❌ Missing: {dir_name}")

    # Check rounds
    rounds = sorted(results_dir.glob('ROUND_*'))
    print(f"✓ Found {len(rounds)} rounds")

    # Verify each round is complete
    for round_dir in rounds:
        flag_file = round_dir / '_ROUND_COMPLETE.flag'
        if flag_file.exists():
            with open(flag_file) as f:
                data = json.load(f)
                print(f"✓ {round_dir.name}: {data['stopping_status']}")
        else:
            print(f"❌ {round_dir.name}: INCOMPLETE")
```

### Calculate Sizes

```python
from pathlib import Path

def calculate_sizes(results_dir):
    """Calculate disk usage by component."""
    results_dir = Path(results_dir)

    sizes = {}
    for item in results_dir.iterdir():
        if item.is_dir():
            size_mb = sum(f.stat().st_size for f in item.rglob('*')) / (1024**2)
            sizes[item.name] = size_mb

    # Print sorted by size
    for name, size in sorted(sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:40s} {size:>10.1f} MB")

    print(f"{'='*50}")
    print(f"{'TOTAL':40s} {sum(sizes.values()):>10.1f} MB")
```

---

## FAQ

**Q: Why no timestamps in folder names?**
A: Timestamps make folders hard to read and compare. Use descriptive names + metadata files instead.

**Q: Why zero-padded numbers (ROUND_000 vs ROUND_0)?**
A: Zero-padding ensures proper alphabetical sorting in all file explorers and command-line tools.

**Q: Can I change the numbering scheme?**
A: Yes, but be consistent. Update the constants in `utils/constants.py`.

**Q: Why Excel instead of CSV?**
A: Excel preserves formatting and is easier for non-technical users. Use Parquet for performance-critical workflows.

**Q: How do I resume after a crash?**
A: The pipeline automatically detects `_ROUND_COMPLETE.flag` files and resumes from the next incomplete round.

**Q: What if I want custom folder names?**
A: Modify the phase constants in `utils/constants.py` and update the RESULTS_STRUCTURE.md documentation.

---

## Related Documentation

- [RESULTS_STRUCTURE.md](./RESULTS_STRUCTURE.md) - Detailed structure guide
- [README.md](./README.md) - Main project documentation
- [audit/00_INDEX.md](./audit/00_INDEX.md) - Technical audit
- [AUDIT_IMPLEMENTATION_STATUS.md](./AUDIT_IMPLEMENTATION_STATUS.md) - Implementation tracking

---

**Last Updated:** 2025-12-10
**Maintained By:** Offshore Riser ML Team
