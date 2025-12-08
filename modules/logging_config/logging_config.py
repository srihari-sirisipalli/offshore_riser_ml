import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    COLORS = {
        'DEBUG': Fore.WHITE + Style.DIM if COLORAMA_AVAILABLE else '',
        'INFO': Fore.CYAN if COLORAMA_AVAILABLE else '',
        'WARNING': Fore.YELLOW if COLORAMA_AVAILABLE else '',
        'ERROR': Fore.RED if COLORAMA_AVAILABLE else '',
        'CRITICAL': Fore.RED + Style.BRIGHT if COLORAMA_AVAILABLE else ''
    }
    RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else ''

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)

class LoggingConfigurator:
    """Configures system-wide logging with Windows UTF-8 support."""
    
    def __init__(self, config: dict):
        self.config = config.get('logging', {})
        self.log_level = getattr(logging, self.config.get('level', 'INFO').upper())
        self.log_dir = Path("logs")
        
    def setup(self) -> None:
        """Setup all loggers and handlers with UTF-8 encoding for Windows."""
        self.log_dir.mkdir(exist_ok=True)
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # FIX #82: Iterate and remove handlers to avoid disrupting other modules' logging,
        # while preventing duplicate handlers if setup is called multiple times.
        for handler in root_logger.handlers[:]: # Iterate over a copy to safely modify list
            root_logger.removeHandler(handler)
        
        # Console Handler with UTF-8 support for Windows
        if self.config.get('log_to_console', True):
            # FIX: Reconfigure stdout/stderr to use UTF-8 on Windows
            if sys.platform == 'win32':
                try:
                    sys.stdout.reconfigure(encoding='utf-8')
                    # FIX #47: Also reconfigure stderr for full UTF-8 support.
                    sys.stderr.reconfigure(encoding='utf-8') 
                except AttributeError:
                    # FIX #47: Log a warning instead of silently passing for older Python versions.
                    logging.getLogger(__name__).warning("sys.stdout/stderr.reconfigure not available (Python < 3.7). Console encoding issues may occur on Windows.")
            
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            
            if self.config.get('colorful_console', True):
                formatter = ColoredFormatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            else:
                formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            
        # File Handler (always UTF-8)
        if self.config.get('log_to_file', True):
            self._add_file_handler(root_logger, "pipeline.log")
            
    def _add_file_handler(self, logger: logging.Logger, filename: str):
        file_path = self.log_dir / filename
        # FIX: Explicitly use UTF-8 encoding for file handler
        handler = RotatingFileHandler(
            file_path, 
            maxBytes=10*1024*1024, 
            backupCount=5,
            encoding='utf-8'  # Ensure UTF-8 for file
        )
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)