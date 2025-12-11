import pytest
from unittest.mock import Mock
from pathlib import Path
from modules.base.base_engine import BaseEngine

# Concrete implementation for testing purposes
class ConcreteTestEngine(BaseEngine):
    def __init__(self, config, logger, engine_dir_name):
        self._engine_dir_name_value = engine_dir_name
        super().__init__(config, logger)

    def _get_engine_directory_name(self) -> str:
        return self._engine_dir_name_value

    def execute(self, *args, **kwargs):
        pass # Not relevant for BaseEngine tests

@pytest.fixture
def mock_logger():
    """Provides a mock logger instance."""
    return Mock()

@pytest.fixture
def base_config(tmp_path):
    """Provides a base configuration dictionary with a temporary results directory."""
    return {
        'outputs': {
            'base_results_dir': str(tmp_path)
        }
    }

def test_base_engine_directory_creation_no_mapping(base_config, mock_logger, tmp_path):
    """
    Tests that BaseEngine correctly creates the output directory when no mapping is present.
    output_dir and standard_output_dir should be the same.
    """
    engine_name = "UNMAPPED_ENGINE"
    engine = ConcreteTestEngine(base_config, mock_logger, engine_name)

    expected_dir = tmp_path / engine_name
    assert engine.output_dir == expected_dir
    assert engine.standard_output_dir == expected_dir
    assert expected_dir.is_dir()
    mock_logger.info.assert_called_with(f"Output directory for ConcreteTestEngine: {expected_dir}")

def test_base_engine_directory_creation_with_mapping(base_config, mock_logger, tmp_path):
    """
    Tests that BaseEngine correctly creates both legacy and standard directories
    when a mapping is defined in STANDARD_DIR_MAP.
    """
    legacy_engine_name = "02_SMART_SPLIT" # A key in STANDARD_DIR_MAP
    standard_mapped_name = BaseEngine.STANDARD_DIR_MAP[legacy_engine_name]

    engine = ConcreteTestEngine(base_config, mock_logger, legacy_engine_name)

    expected_legacy_dir = tmp_path / legacy_engine_name
    expected_standard_dir = tmp_path / standard_mapped_name

    assert engine.output_dir == expected_legacy_dir
    assert engine.standard_output_dir == expected_standard_dir
    assert expected_legacy_dir.is_dir()
    assert expected_standard_dir.is_dir()
    mock_logger.info.assert_called_with(
        f"Output directory for ConcreteTestEngine: {expected_legacy_dir} (legacy), {expected_standard_dir} (standard)"
    )

def test_base_engine_standard_dir_map_values():
    """
    Tests that the STANDARD_DIR_MAP contains expected mappings.
    This is a sanity check for the hardcoded map.
    """
    assert BaseEngine.STANDARD_DIR_MAP["02_SMART_SPLIT"] == "02_MASTER_SPLITS"
    assert BaseEngine.STANDARD_DIR_MAP["09_ADVANCED_ANALYTICS"] == "98_ENSEMBLING"
    assert BaseEngine.STANDARD_DIR_MAP["13_REPRODUCIBILITY_PACKAGE"] == "96_REPRODUCIBILITY_PACKAGE"
