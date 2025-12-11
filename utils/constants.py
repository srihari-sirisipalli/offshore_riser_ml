# utils/constants.py

# --- Top-Level Result Directories ---
# Only pipeline-wide artifacts - NOT per-round results
# Sequentially numbered for proper sorting with clear, self-explanatory names

CONFIG_DIR = "01_RunConfiguration"                    # Run config, metadata, seeds
DATA_INTEGRITY_DIR = "02_DataQualityChecks"           # Data validation, integrity tracking
MASTER_SPLITS_DIR = "03_MasterDataSplits"             # Single train/val/test split
RFE_SUMMARY_DIR = "04_RFE_AggregatedResults"          # Summary across all RFE rounds
GLOBAL_ERROR_TRACKING_DIR = "05_ModelEvolutionTracking"  # Evolution across rounds
ENSEMBLING_DIR = "06_FinalEnsemble"                   # Final ensemble results (if enabled)
RECONSTRUCTION_MAPPING_DIR = "07_BestModelReconstruction"  # Best model info
REPRODUCIBILITY_PACKAGE_DIR = "08_DeploymentPackage"  # Deployment artifacts
REPORTING_DIR = "09_FinalReports"                     # PDF reports

# Canonical list used when building the base results structure (in order).
TOP_LEVEL_RESULT_DIRS = [
    CONFIG_DIR,
    DATA_INTEGRITY_DIR,
    MASTER_SPLITS_DIR,
    RFE_SUMMARY_DIR,
    GLOBAL_ERROR_TRACKING_DIR,
    ENSEMBLING_DIR,
    RECONSTRUCTION_MAPPING_DIR,
    REPRODUCIBILITY_PACKAGE_DIR,
    REPORTING_DIR,
]

# Legacy names kept for backward compatibility in engines
# These should NOT be created at top level - only used inside ROUND_XXX
DATA_VALIDATION_DIR = DATA_INTEGRITY_DIR  # Alias for legacy code
HPO_OPTIMIZATION_DIR = "HPO_GridSearch"   # Only in rounds
HPO_ANALYSIS_DIR = "HPO_Analysis"         # Only in rounds
FINAL_MODEL_DIR = "06_TrainedModel"       # Only in rounds or stability runs; numbered for ordering
PREDICTIONS_DIR = "ModelPredictions"      # Only in rounds
EVALUATION_DIR = "PerformanceMetrics"     # Only in rounds
ERROR_ANALYSIS_ENGINE_DIR = "ErrorAnalysis"  # Only in rounds
DIAGNOSTICS_ENGINE_DIR = "DiagnosticPlots"   # Only in rounds

# Legacy mapping removed - using sequential numbering only

# --- RFE Round Sub-Directories ---
# Clear, self-explanatory names for artifacts within each ROUND_XXX folder
# Sequentially numbered for proper alphabetical sorting

ROUND_DATASETS_DIR = "01_FeatureSet_CurrentRound"               # Features used this round
ROUND_GRID_SEARCH_DIR = "02_Hyperparameter_GridSearch"          # HPO search results
ROUND_HPO_ANALYSIS_DIR = "03_Hyperparameter_Analysis"           # HPO analysis plots
ROUND_BASE_MODEL_RESULTS_DIR = "04_BaseModel_WithAllFeatures"   # Baseline model results
ROUND_FEATURE_EVALUATION_DIR = "05_Feature_ImportanceRanking"   # LOFO evaluation
ROUND_DROPPED_FEATURE_RESULTS_DIR = "06_ReducedModel_FeatureDropped"  # Model after dropping worst feature
ROUND_COMPARISON_DIR = "07_ModelComparison_BaseVsReduced"       # Baseline vs dropped comparison
ROUND_ERROR_ANALYSIS_DIR = "08_ErrorAnalysis_ByConditions"      # Error breakdown by Hs, angle
ROUND_DIAGNOSTICS_DIR = "09_DiagnosticPlots_Standard"           # Residuals, scatter plots
ROUND_ADVANCED_VISUALIZATIONS_DIR = "10_DiagnosticPlots_Advanced"  # 3D surfaces, contours
ROUND_BOOTSTRAPPING_DIR = "11_UncertaintyAnalysis_Bootstrap"    # Bootstrap confidence intervals
ROUND_STABILITY_DIR = "12_ModelStability_CrossValidation"       # Stability analysis
ROUND_TRAINING_DIR = "13_ModelTraining_Details"                 # Training artifacts
ROUND_PREDICTIONS_DIR = "14_Predictions_AllSplits"              # Predictions with Hs
ROUND_EVALUATION_DIR = "15_PerformanceMetrics_AllSplits"        # Metrics by split
ROUND_METRICS_DIR = "16_RoundMetrics_Summary"                   # Round summary metrics
ROUND_FEATURES_DIR = "17_FeatureList_Evolution"                 # Feature timeline
ROUND_EVOLUTION_PLOTS_DIR = "18_EvolutionPlots_RoundProgress"   # Progress visualization

# Canonical round directory ordering for structure creation (in sequence).
ROUND_STRUCTURE_DIRS = [
    ROUND_DATASETS_DIR,
    ROUND_GRID_SEARCH_DIR,
    ROUND_HPO_ANALYSIS_DIR,
    ROUND_BASE_MODEL_RESULTS_DIR,
    ROUND_FEATURE_EVALUATION_DIR,
    ROUND_DROPPED_FEATURE_RESULTS_DIR,
    ROUND_COMPARISON_DIR,
    ROUND_ERROR_ANALYSIS_DIR,
    ROUND_DIAGNOSTICS_DIR,
    ROUND_ADVANCED_VISUALIZATIONS_DIR,
    ROUND_BOOTSTRAPPING_DIR,
    ROUND_STABILITY_DIR,
    ROUND_TRAINING_DIR,
    ROUND_PREDICTIONS_DIR,
    ROUND_EVALUATION_DIR,
    ROUND_METRICS_DIR,
    ROUND_FEATURES_DIR,
    ROUND_EVOLUTION_PLOTS_DIR,
]

# Legacy mapping removed - using sequential numbering only

# --- File Names ---
CONFIG_USED_FILE = "config_used.json"
CONFIG_HASH_FILE = "config_hash.txt"
RUN_METADATA_FILE = "run_metadata.json"
OPTIMAL_FEATURE_SET_FILE = "optimal_feature_set.json"
DEPLOYMENT_CHECKLIST_FILE = "deployment_readiness_checklist.parquet"
SYSTEM_INFO_FILE = "system_info.json"
REQUIREMENTS_FROZEN_FILE = "requirements_frozen.txt"
README_FILE = "README.md"

# --- Stratification Bins (for data management and splitting) ---
COMBINED_BIN = "combined_bin"
HS_BIN = "hs_bin"
ANGLE_BIN = "angle_bin"
