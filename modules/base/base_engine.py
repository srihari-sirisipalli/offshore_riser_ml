import abc
import logging
from pathlib import Path
from typing import Dict, Any
from utils import constants

class BaseEngine(abc.ABC):
    """
    Abstract base class for all processing engines.

    Provides common functionality for:
    - Configuration and logger attachment.
    - Standardized output directory management with sequential numbering.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.base_dir = Path(self.config.get('outputs', {}).get('base_results_dir', 'results'))
        self.engine_dir_name = self._get_engine_directory_name()

        # Direct sequential directory path only
        self.output_dir = self.base_dir / self.engine_dir_name
        self.standard_output_dir = self.output_dir  # Same as output_dir now

        self._setup_directories()

    @abc.abstractmethod
    def _get_engine_directory_name(self) -> str:
        """
        Determines the directory name for the engine's output.
        e.g., '01_DATA_VALIDATION', '02_HPO_SEARCH'
        This should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_engine_directory_name.")

    def _setup_directories(self):
        """
        Creates the main output directory for the engine.
        """
        skip_dirs = self.config.get('outputs', {}).get('skip_dir_creation', False)
        if skip_dirs:
            # Directory creation explicitly disabled (used for compute-only helpers)
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.standard_output_dir != self.output_dir:
            self.standard_output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(
                f"Output directory for {self.__class__.__name__}: "
                f"{self.output_dir} (legacy), {self.standard_output_dir} (standard)"
            )
        else:
            self.logger.info(f"Output directory for {self.__class__.__name__}: {self.output_dir}")

    @abc.abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Main execution method for the engine.
        This must be implemented by all subclasses.
        """
        pass
