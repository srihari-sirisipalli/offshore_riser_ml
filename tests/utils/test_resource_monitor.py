import pytest
from unittest.mock import Mock, patch
import json
from pathlib import Path
import os
import time

# Mock psutil for testing purposes
# We cannot directly import psutil here as it might not be installed in all environments
# and we want to test its absence as well.
# We will patch _safe_import_psutil directly.
from utils.resource_monitor import capture_resource_snapshot, write_resource_dashboard

@pytest.fixture
def mock_psutil_available():
    """Mocks psutil as available with predictable return values."""
    mock_psutil = Mock()
    mock_psutil.cpu_percent.return_value = 50.0
    
    mock_vm = Mock()
    mock_vm.total = 1024 * (1024 ** 2) # 1024 MB
    mock_vm.available = 512 * (1024 ** 2) # 512 MB
    mock_vm.used = 512 * (1024 ** 2) # 512 MB
    mock_vm.percent = 50.0
    mock_psutil.virtual_memory.return_value = mock_vm

    mock_part = Mock()
    mock_part.mountpoint = "/mnt/data"
    mock_part.device = "/dev/sda1"
    mock_psutil.disk_partitions.return_value = [mock_part]

    mock_usage = Mock()
    mock_usage.total = 1024 ** 3
    mock_usage.used = 256 ** 3
    mock_usage.free = 768 ** 3
    mock_usage.percent = 25.0
    mock_psutil.disk_usage.return_value = mock_usage

    mock_process = Mock()
    mock_mem_info = Mock()
    mock_mem_info.rss = 100 * (1024 ** 2) # 100 MB
    mock_mem_info.vms = 200 * (1024 ** 2) # 200 MB
    mock_process.memory_info.return_value = mock_mem_info
    mock_process.num_threads.return_value = 10
    mock_process.open_files.return_value = ["file1", "file2"] # Simulate open files
    mock_context_manager = Mock()
    mock_context_manager.__enter__ = Mock(return_value=mock_process) # Assign a callable mock for __enter__
    mock_context_manager.__exit__ = Mock(return_value=None) # Assign a callable mock for __exit__
    mock_process.oneshot.return_value = mock_context_manager
    mock_psutil.Process.return_value = mock_process

    with patch('utils.resource_monitor._safe_import_psutil', return_value=mock_psutil):
        yield mock_psutil

@pytest.fixture
def mock_psutil_unavailable():
    """Mocks psutil as unavailable."""
    with patch('utils.resource_monitor._safe_import_psutil', return_value=None):
        yield

@pytest.fixture(autouse=True)
def mock_os_cpu_count():
    """Mocks os.cpu_count to return a fixed value."""
    with patch('os.cpu_count', return_value=4):
        yield

@pytest.fixture(autouse=True)
def mock_time_strftime():
    """Mocks time.strftime to return a fixed timestamp."""
    with patch('time.strftime', return_value="2023-10-27T10:00:00"):
        yield

# --- capture_resource_snapshot tests ---

def test_capture_resource_snapshot_psutil_available(mock_psutil_available):
    """
    Tests capture_resource_snapshot when psutil is available.
    """
    snapshot = capture_resource_snapshot()

    assert snapshot["timestamp"] == "2023-10-27T10:00:00"
    assert snapshot["cpu_count"] == 4
    assert "psutil_available" not in snapshot # Should not be false if available
    assert snapshot["cpu_percent"] == 50.0

    assert snapshot["memory"]["total_mb"] == 1024.0
    assert snapshot["memory"]["available_mb"] == 512.0
    assert snapshot["memory"]["used_mb"] == 512.0
    assert snapshot["memory"]["percent"] == 50.0

    assert len(snapshot["disks"]) == 1
    assert snapshot["disks"][0]["mountpoint"] == "/mnt/data"
    assert snapshot["disks"][0]["percent"] == 25.0

    assert snapshot["process"]["rss_mb"] == 100.0
    assert snapshot["process"]["vms_mb"] == 200.0
    assert snapshot["process"]["num_threads"] == 10
    assert snapshot["process"]["open_files"] == 2 # len(["file1", "file2"])

    mock_psutil_available.cpu_percent.assert_called_once()
    mock_psutil_available.virtual_memory.assert_called_once()
    mock_psutil_available.disk_partitions.assert_called_once()
    mock_psutil_available.disk_usage.assert_called_once_with("/mnt/data")
    mock_psutil_available.Process.assert_called_once()

def test_capture_resource_snapshot_psutil_unavailable(mock_psutil_unavailable):
    """
    Tests capture_resource_snapshot when psutil is unavailable.
    """
    snapshot = capture_resource_snapshot()

    assert snapshot["timestamp"] == "2023-10-27T10:00:00"
    assert snapshot["cpu_count"] == 4
    assert snapshot["psutil_available"] is False
    assert "cpu_percent" not in snapshot # Should not be present

# --- write_resource_dashboard tests ---

@pytest.fixture
def mock_psutil_for_dashboard(mock_psutil_available):
    """Provides a filled snapshot for write_resource_dashboard tests."""
    return capture_resource_snapshot()

@pytest.fixture
def mock_pandas_available():
    """Mocks pandas components used in write_resource_dashboard."""
    mock_pd = Mock()
    mock_dataframe_instance = Mock()
    mock_pd.DataFrame.return_value = mock_dataframe_instance
    
    with patch.dict('sys.modules', {'pandas': mock_pd}):
        yield mock_pd, mock_dataframe_instance

@pytest.fixture
def mock_pandas_unavailable():
    """Mocks pandas import to simulate unavailability."""
    with patch.dict('sys.modules', {'pandas': None}): # Make 'import pandas' fail
        yield

def test_write_resource_dashboard_with_pandas_and_psutil_available(tmp_path, mock_psutil_available, mock_pandas_available, mock_time_strftime):
    """
    Tests write_resource_dashboard when both pandas and psutil are available.
    """
    mock_pd, mock_dataframe_instance = mock_pandas_available

    json_path_returned = write_resource_dashboard(tmp_path)

    out_dir = tmp_path / "00_DATA_INTEGRITY"
    expected_json_path = out_dir / "resource_utilization_dashboard.json"
    expected_parquet_path = out_dir / "resource_utilization_dashboard.parquet"

    # Assert JSON file is written
    assert expected_json_path.is_file()
    assert json_path_returned == expected_json_path
    json_content = json.loads(expected_json_path.read_text())
    assert json_content["cpu_percent"] == 50.0
    assert json_content["timestamp"] == "2023-10-27T10:00:00"

    # Assert pandas DataFrame creation and parquet writing
    mock_pd.DataFrame.assert_called_once()
    mock_dataframe_instance.to_parquet.assert_called_once_with(expected_parquet_path, index=False)
    assert not mock_dataframe_instance.to_excel.called # excel_copy is False by default

    assert expected_parquet_path.parent.is_dir() # Ensure parent dir was created

def test_write_resource_dashboard_with_excel_copy(tmp_path, mock_psutil_available, mock_pandas_available):
    """
    Tests write_resource_dashboard when excel_copy is True.
    """
    _ , mock_dataframe_instance = mock_pandas_available

    write_resource_dashboard(tmp_path, excel_copy=True)

    out_dir = tmp_path / "00_DATA_INTEGRITY"
    expected_parquet_path = out_dir / "resource_utilization_dashboard.parquet"
    expected_excel_path = expected_parquet_path.with_suffix(".xlsx")

    mock_dataframe_instance.to_parquet.assert_called_once()
    mock_dataframe_instance.to_excel.assert_called_once_with(expected_excel_path, index=False)


def test_write_resource_dashboard_pandas_unavailable(tmp_path, mock_psutil_available, mock_pandas_unavailable):
    """
    Tests write_resource_dashboard when pandas is not available.
    Should write JSON only.
    """
    json_path_returned = write_resource_dashboard(tmp_path)

    out_dir = tmp_path / "00_DATA_INTEGRITY"
    expected_json_path = out_dir / "resource_utilization_dashboard.json"
    expected_parquet_path = out_dir / "resource_utilization_dashboard.parquet"

    assert expected_json_path.is_file()
    assert json_path_returned == expected_json_path # Should return json path
    assert not expected_parquet_path.is_file() # Parquet should not be written

    # Ensure pandas-related functions were not called (since pandas is unavailable)
    # This requires patching the pandas module itself in _safe_import_psutil, but
    # for `write_resource_dashboard`, the try-except block handles it.
    # We can't directly assert `pd.DataFrame.called` if pd itself couldn't be imported.
    # The main assertion is that the .parquet file is not created.
