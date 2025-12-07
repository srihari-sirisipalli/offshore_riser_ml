import pytest
import os
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import MagicMock
from modules.reporting_engine import ReportingEngine

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def report_config(tmp_path):
    return {
        'outputs': {
            'base_results_dir': str(tmp_path),
            'save_reports_pdf': True
        }
    }

@pytest.fixture
def dummy_data(tmp_path):
    # Create a dummy plot
    plot_path = tmp_path / "test_plot.png"
    plt.figure()
    plt.plot([1, 2], [1, 2])
    plt.title("Test Plot")
    plt.savefig(plot_path)
    plt.close()
    
    return {
        'model_info': {'model': 'TestModel'},
        'metrics': {
            'val': {'cmae': 1.0, 'accuracy_5deg': 90.0},
            'test': {'cmae': 1.1, 'accuracy_5deg': 88.0}
        },
        'plots': [str(plot_path)]
    }

def test_pdf_generation(report_config, mock_logger, dummy_data):
    engine = ReportingEngine(report_config, mock_logger)
    
    pdf_path = engine.generate_report(dummy_data, "test_run_id")
    
    # Check if file returned path is correct
    assert pdf_path.endswith("final_report.pdf")
    
    # Check existence
    assert os.path.exists(pdf_path)
    
    # Check size > 0
    assert os.path.getsize(pdf_path) > 0

def test_missing_plot_handling(report_config, mock_logger, dummy_data):
    # Add a non-existent path
    dummy_data['plots'].append("non_existent_plot.png")
    
    engine = ReportingEngine(report_config, mock_logger)
    pdf_path = engine.generate_report(dummy_data, "test_run_id")
    
    # Should still succeed
    assert os.path.exists(pdf_path)
    
    # Logger should have warned
    mock_logger.warning.assert_called()

def test_disabled_reporting(report_config, mock_logger, dummy_data):
    report_config['outputs']['save_reports_pdf'] = False
    
    engine = ReportingEngine(report_config, mock_logger)
    result = engine.generate_report(dummy_data, "test_run_id")
    
    assert result == ""