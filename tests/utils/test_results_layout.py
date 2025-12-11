import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import json
import pandas as pd
from utils.results_layout import ResultsLayoutManager

# Mocking file_io functions
@pytest.fixture(autouse=True)
def mock_file_io():
    with patch('utils.results_layout.save_dataframe') as mock_save_df, \
         patch('utils.results_layout.read_dataframe') as mock_read_df:
        yield mock_save_df, mock_read_df

@pytest.fixture
def mock_logger():
    """Provides a mock logger instance."""
    return Mock()

@pytest.fixture
def manager(tmp_path, mock_logger):
    """Provides a ResultsLayoutManager instance with a temporary base directory."""
    return ResultsLayoutManager(base_dir=tmp_path, logger=mock_logger)

def create_dummy_parquet(file_path: Path, df_content: dict = None):
    """Helper to create a dummy parquet file with minimal content."""
    df_content = df_content if df_content is not None else {'col1': [1, 2], 'col2': ['a', 'b']}
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(df_content).to_parquet(file_path)

def create_dummy_json(file_path: Path, content: dict = None):
    """Helper to create a dummy JSON file."""
    content = content if content is not None else {'key': 'value'}
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(content))

def create_dummy_txt(file_path: Path, content: str = "test content"):
    """Helper to create a dummy text file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)

# Test cases

def test_ensure_base_structure(manager, tmp_path):
    """Tests that ensure_base_structure creates all top-level folders."""
    manager.ensure_base_structure()
    expected_folders = [
        "00_CONFIG", "00_DATA_INTEGRITY", "01_DATA_VALIDATION",
        "02_MASTER_SPLITS", "03_HYPERPARAMETER_OPTIMIZATION",
        "04_HYPERPARAMETER_ANALYSIS", "98_ENSEMBLING",
        "97_RECONSTRUCTION_MAPPING", "96_REPRODUCIBILITY_PACKAGE",
        "99_RFE_SUMMARY",
    ]
    for folder in expected_folders:
        assert (tmp_path / folder).is_dir()

def test_mirror_config_artifacts(manager, tmp_path):
    """Tests mirroring of config-related artifacts."""
    # Create dummy source files
    create_dummy_json(tmp_path / "config_used.json")
    create_dummy_txt(tmp_path / "config_hash.txt")
    create_dummy_json(tmp_path / "run_metadata.json")

    manager.mirror_config_artifacts()

    # Assert destination files exist
    assert (tmp_path / "00_CONFIG" / "config_used.json").is_file()
    assert (tmp_path / "00_CONFIG" / "config_hash.txt").is_file()
    assert (tmp_path / "00_CONFIG" / "run_metadata.json").is_file()

def test_mirror_splits(manager, tmp_path):
    """Tests mirroring of smart split outputs."""
    src_dir = tmp_path / "02_SMART_SPLIT"
    src_dir.mkdir()
    create_dummy_parquet(src_dir / "train.parquet")
    create_dummy_parquet(src_dir / "val.parquet")

    manager.mirror_splits()

    dest_dir = tmp_path / "02_MASTER_SPLITS"
    assert (dest_dir / "train.parquet").is_file()
    assert (dest_dir / "val.parquet").is_file()
    assert not (dest_dir / "test.parquet").is_file() # Only created sources should be mirrored

def test_mirror_global_tracking(manager, tmp_path):
    """Tests mirroring of global error tracking artifacts."""
    src_dir = tmp_path / "01_GLOBAL_TRACKING"
    (src_dir / "01_metrics").mkdir(parents=True)
    (src_dir / "02_features").mkdir(parents=True)
    (src_dir / "04_evolution_plots").mkdir(parents=True)

    create_dummy_parquet(src_dir / "01_metrics" / "metrics_all_rounds.parquet")
    create_dummy_parquet(src_dir / "02_features" / "features_eliminated_timeline.parquet")
    create_dummy_txt(src_dir / "04_evolution_plots" / "plot1.png")
    create_dummy_parquet(tmp_path / "safety_threshold_summary_all_rounds.parquet")
    create_dummy_parquet(tmp_path / "significance_baseline_vs_dropped_over_rounds.parquet")
    create_dummy_parquet(tmp_path / "safety_gate_status_all_rounds.parquet")


    manager.mirror_global_tracking()

    dest_dir = tmp_path / "99_RFE_SUMMARY"
    assert (dest_dir / "all_rounds_metrics.parquet").is_file()
    assert (dest_dir / "feature_elimination_history.parquet").is_file()
    assert (dest_dir / "plots" / "plot1.png").is_file()
    assert (dest_dir / "safety_threshold_summary_all_rounds.parquet").is_file()
    assert (dest_dir / "statistical_tests_round_comparisons.parquet").is_file()
    assert (dest_dir / "safety_gate_status_all_rounds.parquet").is_file()


def test_mirror_ensembling(manager, tmp_path):
    """Tests mirroring of ensembling outputs."""
    src_dir = tmp_path / "09_ADVANCED_ANALYTICS" / "ensembling"
    src_dir.mkdir(parents=True)
    create_dummy_parquet(src_dir / "ensemble_predictions.parquet")
    create_dummy_txt(src_dir / "ensemble_report.txt")

    manager.mirror_ensembling()

    dest_dir = tmp_path / "98_ENSEMBLING"
    assert (dest_dir / "ensemble_predictions.parquet").is_file()
    assert (dest_dir / "ensemble_report.txt").is_file()

def test_mirror_reconstruction_mapping(manager, tmp_path):
    """Tests mirroring of reconstruction mapping artifacts."""
    create_dummy_parquet(tmp_path / "model_reconstruction_mapping.parquet")
    create_dummy_parquet(tmp_path / "reporting" / "model_reconstruction_summary.parquet") # rglob test

    manager.mirror_reconstruction_mapping()

    dest_dir = tmp_path / "97_RECONSTRUCTION_MAPPING"
    assert (dest_dir / "model_reconstruction_mapping.parquet").is_file()
    assert (dest_dir / "model_reconstruction_summary.parquet").is_file()


def test_mirror_reproducibility_package(manager, tmp_path):
    """Tests mirroring of reproducibility package."""
    src_dir = tmp_path / "13_REPRODUCIBILITY_PACKAGE"
    src_dir.mkdir(parents=True)
    create_dummy_txt(src_dir / "package.zip")

    manager.mirror_reproducibility_package()

    dest_dir = tmp_path / "96_REPRODUCIBILITY_PACKAGE"
    assert (dest_dir / "package.zip").is_file()

def test_mirror_run(manager, tmp_path):
    """Tests that mirror_run orchestrates all top-level mirroring."""
    # Create some source files for various mirrors
    create_dummy_json(tmp_path / "config_used.json")
    src_splits_dir = tmp_path / "02_SMART_SPLIT"
    src_splits_dir.mkdir()
    create_dummy_parquet(src_splits_dir / "train.parquet")
    (tmp_path / "01_GLOBAL_TRACKING" / "01_metrics").mkdir(parents=True)
    create_dummy_parquet(tmp_path / "01_GLOBAL_TRACKING" / "01_metrics" / "metrics_all_rounds.parquet")
    (tmp_path / "09_ADVANCED_ANALYTICS" / "ensembling").mkdir(parents=True)
    create_dummy_parquet(tmp_path / "09_ADVANCED_ANALYTICS" / "ensembling" / "ensemble_predictions.parquet")
    create_dummy_parquet(tmp_path / "model_reconstruction_mapping.parquet")
    (tmp_path / "13_REPRODUCIBILITY_PACKAGE").mkdir()
    create_dummy_txt(tmp_path / "13_REPRODUCIBILITY_PACKAGE" / "repro.zip")

    manager.mirror_run()

    # Assert top-level structure is created
    assert (tmp_path / "02_MASTER_SPLITS").is_dir()
    assert (tmp_path / "99_RFE_SUMMARY").is_dir()
    assert (tmp_path / "98_ENSEMBLING").is_dir()
    assert (tmp_path / "97_RECONSTRUCTION_MAPPING").is_dir()
    assert (tmp_path / "96_REPRODUCIBILITY_PACKAGE").is_dir()

    # Assert specific mirrored files
    assert (tmp_path / "00_CONFIG" / "config_used.json").is_file()
    assert (tmp_path / "02_MASTER_SPLITS" / "train.parquet").is_file()
    assert (tmp_path / "99_RFE_SUMMARY" / "all_rounds_metrics.parquet").is_file()
    assert (tmp_path / "98_ENSEMBLING" / "ensemble_predictions.parquet").is_file()
    assert (tmp_path / "97_RECONSTRUCTION_MAPPING" / "model_reconstruction_mapping.parquet").is_file()
    assert (tmp_path / "96_REPRODUCIBILITY_PACKAGE" / "repro.zip").is_file()


def test_ensure_round_structure(manager, tmp_path):
    """Tests that ensure_round_structure creates subdirectories within a round."""
    round_dir = tmp_path / "round_1"
    manager.ensure_round_structure(round_dir)
    expected_subdirs = [
        "01_TRAINING", "02_PREDICTIONS", "03_EVALUATION",
        "04_FEATURE_EVALUATION", "05_ERROR_ANALYSIS", "06_COMPARISON",
        "07_DIAGNOSTICS", "08_ADVANCED_VISUALIZATIONS", "09_BOOTSTRAPPING",
        "10_STABILITY",
    ]
    for subdir in expected_subdirs:
        assert (round_dir / subdir).is_dir()

def test_mirror_baseline_outputs(manager, tmp_path, mock_file_io):
    """Tests mirroring of baseline outputs for a specific round."""
    mock_save_df, mock_read_df = mock_file_io

    round_dir = tmp_path / "round_1"
    legacy_base_dir = round_dir / "03_BASE_MODEL_RESULTS"
    legacy_base_dir.mkdir(parents=True)

    # Create dummy legacy files
    create_dummy_parquet(legacy_base_dir / "baseline_predictions_val.parquet", {'pred': [1]})
    create_dummy_parquet(legacy_base_dir / "baseline_predictions_test.parquet", {'pred': [2]})
    create_dummy_parquet(legacy_base_dir / "baseline_metrics_val.parquet", {'metric': [0.5]})
    create_dummy_parquet(legacy_base_dir / "baseline_metrics_test.parquet", {'metric': [0.6]})
    create_dummy_parquet(legacy_base_dir / "safety_threshold_summary.parquet")
    (legacy_base_dir / "09_ERROR_ANALYSIS").mkdir()
    create_dummy_txt(legacy_base_dir / "09_ERROR_ANALYSIS" / "error.txt")
    (legacy_base_dir / "08_DIAGNOSTICS").mkdir()
    create_dummy_txt(legacy_base_dir / "08_DIAGNOSTICS" / "diag.txt")
    (legacy_base_dir / "08_ADVANCED_VISUALIZATIONS").mkdir()
    create_dummy_txt(legacy_base_dir / "08_ADVANCED_VISUALIZATIONS" / "viz.png")

    # Mock read_dataframe to return simple DataFrames
    mock_read_df.side_effect = [
        pd.DataFrame({'metric': [0.5]}), # for val metrics
        pd.DataFrame({'metric': [0.6]})
    ]

    manager.mirror_baseline_outputs(round_dir)

    # Assert round structure is created
    assert (round_dir / "02_PREDICTIONS").is_dir()
    assert (round_dir / "03_EVALUATION").is_dir()
    assert (round_dir / "05_ERROR_ANALYSIS").is_dir()
    assert (round_dir / "07_DIAGNOSTICS").is_dir()
    assert (round_dir / "08_ADVANCED_VISUALIZATIONS").is_dir()

    # Assert predictions
    assert (round_dir / "02_PREDICTIONS" / "predictions_val.parquet").is_file()
    assert (round_dir / "02_PREDICTIONS" / "predictions_test.parquet").is_file()

    # Assert metrics
    assert (round_dir / "03_EVALUATION" / "metrics_val.parquet").is_file()
    assert (round_dir / "03_EVALUATION" / "metrics_test.parquet").is_file()
    mock_save_df.assert_called_with(
        mock_save_df.call_args[0][0],  # The DataFrame argument
        round_dir / "03_EVALUATION" / "combined_metrics.parquet",
        excel_copy=manager.excel_copy,
        index=False
    )
    assert mock_save_df.called # Check if save_dataframe was called for combined metrics

    # Assert error analysis
    assert (round_dir / "05_ERROR_ANALYSIS" / "safety_threshold_summary.parquet").is_file()
    assert (round_dir / "05_ERROR_ANALYSIS" / "error.txt").is_file()

    # Assert diagnostics
    assert (round_dir / "07_DIAGNOSTICS" / "diag.txt").is_file()

    # Assert advanced visualizations
    assert (round_dir / "08_ADVANCED_VISUALIZATIONS" / "viz.png").is_file()

    # Assert round summary
    assert (round_dir / "round_summary.json").is_file()
    summary_content = json.loads((round_dir / "round_summary.json").read_text())
    assert summary_content["mirrored"] is True
    assert summary_content["artifacts"]["predictions_val"] is True
    assert summary_content["artifacts"]["metrics_test"] is True


def test_write_combined_metrics_no_data(manager, tmp_path, mock_file_io):
    """Tests _write_combined_metrics when no metric files exist."""
    mock_save_df, mock_read_df = mock_file_io
    dest_dir = tmp_path / "eval"
    dest_dir.mkdir()
    val_path = tmp_path / "non_existent_val.parquet"
    test_path = tmp_path / "non_existent_test.parquet"

    manager._write_combined_metrics(dest_dir, val_path, test_path)

    mock_read_df.assert_not_called()
    mock_save_df.assert_not_called()
    assert not (dest_dir / "combined_metrics.parquet").is_file()

def test_write_combined_metrics_only_val(manager, tmp_path, mock_file_io):
    """Tests _write_combined_metrics when only val metrics exist."""
    mock_save_df, mock_read_df = mock_file_io
    dest_dir = tmp_path / "eval"
    dest_dir.mkdir()
    val_path = tmp_path / "val.parquet"
    create_dummy_parquet(val_path, {'metric': [0.7]})
    test_path = tmp_path / "non_existent_test.parquet"

    mock_read_df.return_value = pd.DataFrame({'metric': [0.7]})

    manager._write_combined_metrics(dest_dir, val_path, test_path)

    mock_read_df.assert_called_once_with(val_path)
    mock_save_df.assert_called_once()
    mock_save_df.assert_called_with(
        mock_save_df.call_args[0][0],  # The DataFrame argument
        dest_dir / "combined_metrics.parquet",
        excel_copy=manager.excel_copy,
        index=False
    )

def test_copy_if_exists_src_not_exists(manager, tmp_path):
    """Tests _copy_if_exists when source file does not exist."""
    src = tmp_path / "non_existent.txt"
    dest = tmp_path / "dest" / "file.txt"
    manager._copy_if_exists(src, dest)
    assert not dest.is_file()
    manager.logger.warning.assert_not_called() # No warning if src doesn't exist

def test_copy_if_exists_src_exists(manager, tmp_path):
    """Tests _copy_if_exists when source file exists."""
    src = tmp_path / "source.txt"
    create_dummy_txt(src)
    dest = tmp_path / "dest" / "file.txt"
    manager._copy_if_exists(src, dest)
    assert dest.is_file()
    assert dest.read_text() == "test content"
    manager.logger.warning.assert_not_called()

def test_copy_tree_src_not_exists(manager, tmp_path):
    """Tests _copy_tree when source directory does not exist."""
    src = tmp_path / "non_existent_dir"
    dest = tmp_path / "dest_dir"
    manager._copy_tree(src, dest)
    assert not dest.is_dir()
    manager.logger.warning.assert_not_called()

def test_copy_tree_src_exists(manager, tmp_path):
    """Tests _copy_tree when source directory exists."""
    src = tmp_path / "source_dir"
    (src / "subdir").mkdir(parents=True)
    create_dummy_txt(src / "file1.txt")
    create_dummy_txt(src / "subdir" / "file2.txt")

    dest = tmp_path / "dest_dir"
    manager._copy_tree(src, dest)

    assert (dest / "file1.txt").is_file()
    assert (dest / "subdir" / "file2.txt").is_file()
    assert (dest / "file1.txt").read_text() == "test content"
    assert (dest / "subdir" / "file2.txt").read_text() == "test content"
