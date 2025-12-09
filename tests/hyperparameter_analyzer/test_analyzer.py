import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from unittest.mock import MagicMock
from modules.hyperparameter_analyzer import HyperparameterAnalyzer

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def sample_results(tmp_path):
    # Create fake HPO results
    data = {
        'model': ['ModelA'] * 10,
        'param_n_estimators': [10, 50, 100, 10, 50, 100, 10, 50, 100, 50],
        'param_max_depth': [5, 5, 5, 10, 10, 10, None, None, None, 5],
        'cv_cmae_deg_mean': [10, 8, 7, 6, 5, 4, 9, 8, 7, 7.5], # Lower is better
        'cv_crmse_deg_mean': [12, 10, 9, 8, 7, 6, 11, 10, 9, 9.5],
        'cv_max_error_deg_mean': [20, 18, 17, 16, 15, 14, 19, 18, 17, 17.5]
    }
    df = pd.DataFrame(data)
    
    # Save to mock location
    hpo_dir = tmp_path / "04_HYPERPARAMETER_SEARCH"
    hpo_dir.mkdir(parents=True)
    df.to_excel(hpo_dir / "all_config_results.xlsx", index=False)
    
    return str(tmp_path)

@pytest.fixture
def analyzer_config(sample_results):
    return {
        'outputs': {'base_results_dir': sample_results}
    }

def test_preprocess_data(analyzer_config, mock_logger):
    analyzer = HyperparameterAnalyzer(analyzer_config, mock_logger)
    df = pd.DataFrame({'param_x': [1, None, np.nan]})
    clean_df = analyzer._preprocess_data(df)
    
    assert "None" in clean_df['param_x'].values
    assert pd.isna(clean_df['param_x']).sum() == 0

def test_optimal_range_generation(analyzer_config, mock_logger, sample_results):
    analyzer = HyperparameterAnalyzer(analyzer_config, mock_logger)
    # Mocking analysis call
    analyzer.output_dir = Path(sample_results) / "04_HYPERPARAMETER_SEARCH" / "ANALYSIS"
    analyzer.output_dir.mkdir()
    
    df = pd.read_excel(Path(sample_results) / "04_HYPERPARAMETER_SEARCH" / "all_config_results.xlsx")
    df = analyzer._preprocess_data(df)
    
    top_df = analyzer._generate_optimal_ranges_report(df)

    # Top 10% of 10 rows = 1 row. Best is CMAE=4 (param_n=100, depth=10)
    assert len(top_df) >= 1
    assert top_df.iloc[0]['cv_cmae_deg_mean'] == 4
    
    report_path = analyzer.output_dir / "optimal_parameter_ranges.xlsx"
    assert report_path.exists()

def test_plotting_logic(analyzer_config, mock_logger, sample_results):
    analyzer = HyperparameterAnalyzer(analyzer_config, mock_logger)
    analyzer.analyze("test_run")
    
    # FIX: Update path to match implementation's nested structure
    # Implementation uses: base_results_dir / "05_HYPERPARAMETER_ANALYSIS"
    output_dir = Path(sample_results) / "05_HYPERPARAMETER_ANALYSIS"
    
    # Structure: visualizations / {model} / {metric_type} / {metric} / heatmap / {style} / {p1}_vs_{p2}.png
    # Model: ModelA
    # Metric Type: CV (implied by columns in sample_results)
    # Metric: cv_cmae_deg_mean (primary)
    # Style: box (default first style)
    # Params: n_estimators vs max_depth
    heatmap_file = output_dir / "visualizations" / "ModelA" / "CV" / "cv_cmae_deg_mean" / "heatmap" / "box" / "n_estimators_vs_max_depth.png"